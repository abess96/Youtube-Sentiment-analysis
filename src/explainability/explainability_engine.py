"""
Explainability Engine for Model Interpretability.
Implements LIME and SHAP explanations for sentiment analysis models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mlflow
    from src.config.mlflow_config import setup_mlflow
    MLFLOW_AVAILABLE = True
    setup_mlflow()
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Structured explanation result."""
    method: str
    feature_importance: Dict[str, float]
    prediction: Any
    confidence: float
    visualization_data: Optional[Dict[str, Any]] = None


class ExplainabilityEngine:
    """
    Engine for generating model explanations using LIME and SHAP.
    Supports both traditional ML and deep learning models.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None, log_to_mlflow: bool = True):
        """
        Initialize the explainability engine.
        
        Args:
            model: Trained model with predict/predict_proba methods
            feature_names: Names of input features
            class_names: Names of output classes
            log_to_mlflow: Whether to automatically log to MLflow
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or ['negative', 'neutral', 'positive']
        self.log_to_mlflow = log_to_mlflow and MLFLOW_AVAILABLE
        
    def explain_with_lime(self, X: np.ndarray, instance_idx: int = 0,
                         num_features: int = 10, **kwargs) -> Explanation:
        """
        Generate LIME explanation for a prediction.
        
        Args:
            X: Input features
            instance_idx: Index of instance to explain
            num_features: Number of top features to show
            **kwargs: Additional LIME parameters
            
        Returns:
            Explanation object with feature importance
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(X[instance_idx:instance_idx+1])[0]
            prediction = np.argmax(prediction_proba)
            confidence = float(prediction_proba[prediction])
        else:
            prediction = self.model.predict(X[instance_idx:instance_idx+1])[0]
            confidence = 1.0
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            **kwargs
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            X[instance_idx],
            self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
            num_features=num_features
        )
        
        # Extract feature importance
        feature_importance = dict(exp.as_list())
        
        logger.info(f"Generated LIME explanation for instance {instance_idx}")
        
        return Explanation(
            method='LIME',
            feature_importance=feature_importance,
            prediction=prediction,
            confidence=confidence,
            visualization_data={'lime_exp': exp}
        )
    
    def explain_with_shap(self, X: np.ndarray, background_samples: int = 100,
                         instance_idx: Optional[int] = None) -> Union[Explanation, Dict[str, Any]]:
        """
        Generate SHAP explanation for predictions.
        
        Args:
            X: Input features
            background_samples: Number of background samples for SHAP
            instance_idx: Specific instance to explain (None for all)
            
        Returns:
            Explanation object or dict with SHAP values
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        # Select background data
        background = shap.sample(X, min(background_samples, len(X)))
        
        # Create SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.KernelExplainer(self.model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(self.model.predict, background)
        
        # Calculate SHAP values
        if instance_idx is not None:
            shap_values = explainer.shap_values(X[instance_idx:instance_idx+1])
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(X[instance_idx:instance_idx+1])[0]
                prediction = np.argmax(prediction_proba)
                confidence = float(prediction_proba[prediction])
            else:
                prediction = self.model.predict(X[instance_idx:instance_idx+1])[0]
                confidence = 1.0
            
            # Extract feature importance for predicted class
            if isinstance(shap_values, list):
                values = shap_values[prediction][0]
            else:
                values = shap_values[0]
            
            feature_importance = {
                self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}': float(values[i])
                for i in range(len(values))
            }
            
            logger.info(f"Generated SHAP explanation for instance {instance_idx}")
            
            return Explanation(
                method='SHAP',
                feature_importance=feature_importance,
                prediction=prediction,
                confidence=confidence,
                visualization_data={'shap_values': shap_values, 'base_value': explainer.expected_value}
            )
        else:
            shap_values = explainer.shap_values(X)
            
            logger.info(f"Generated SHAP values for {len(X)} instances")
            
            return {
                'shap_values': shap_values,
                'base_value': explainer.expected_value,
                'feature_names': self.feature_names
            }
    
    def get_global_feature_importance(self, X: np.ndarray, method: str = 'shap',
                                     top_k: int = 20) -> Dict[str, float]:
        """
        Calculate global feature importance across dataset.
        
        Args:
            X: Input features
            method: Method to use ('shap' or 'permutation')
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature importance scores
        """
        if method == 'shap':
            try:
                import shap
                
                # Get SHAP values for all instances
                result = self.explain_with_shap(X, background_samples=100)
                shap_values = result['shap_values']
                
                # Calculate mean absolute SHAP values
                if isinstance(shap_values, list):
                    # Multi-class: average across classes
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
                
                # Create importance dict
                importance = {
                    self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}': float(mean_shap[i])
                    for i in range(len(mean_shap))
                }
                
            except ImportError:
                logger.warning("SHAP not available, falling back to permutation importance")
                method = 'permutation'
        
        if method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model, X, self.model.predict(X),
                n_repeats=10, random_state=42
            )
            
            importance = {
                self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}': float(result.importances_mean[i])
                for i in range(len(result.importances_mean))
            }
        
        # Sort and return top k
        sorted_importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
        
        logger.info(f"Calculated global feature importance using {method}")
        
        # Log to MLflow
        if self.log_to_mlflow:
            try:
                for feature, importance in list(sorted_importance.items())[:10]:
                    mlflow.log_metric(f"importance_{feature}", importance)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return sorted_importance
    
    def visualize_explanation(self, explanation: Explanation, save_path: Optional[str] = None) -> None:
        """
        Visualize feature importance from explanation.
        
        Args:
            explanation: Explanation object to visualize
            save_path: Path to save visualization
        """
        # Sort features by absolute importance
        sorted_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]
        
        features, importances = zip(*sorted_features)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in importances]
        plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'{explanation.method} Explanation\nPrediction: {self.class_names[explanation.prediction]} (Confidence: {explanation.confidence:.2%})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved explanation visualization to {save_path}")
            
            # Log to MLflow
            if self.log_to_mlflow:
                try:
                    mlflow.log_artifact(save_path, artifact_path="explainability")
                except Exception as e:
                    logger.warning(f"Failed to log artifact to MLflow: {e}")
        
        plt.close()
    
    def generate_explanation_report(self, X: np.ndarray, y: np.ndarray,
                                   sample_size: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report.
        
        Args:
            X: Input features
            y: True labels
            sample_size: Number of samples to explain
            
        Returns:
            Dictionary with explanation metrics and insights
        """
        # Sample instances
        indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        
        explanations = []
        for idx in indices:
            try:
                exp = self.explain_with_lime(X, instance_idx=idx, num_features=10)
                explanations.append(exp)
            except Exception as e:
                logger.warning(f"Failed to explain instance {idx}: {e}")
        
        # Calculate global importance
        global_importance = self.get_global_feature_importance(X, method='permutation', top_k=20)
        
        # Aggregate insights
        report = {
            'num_explanations': len(explanations),
            'global_feature_importance': global_importance,
            'top_features': list(global_importance.keys())[:10],
            'sample_explanations': [
                {
                    'prediction': exp.prediction,
                    'confidence': exp.confidence,
                    'top_features': list(exp.feature_importance.keys())[:5]
                }
                for exp in explanations
            ]
        }
        
        logger.info(f"Generated explanation report for {len(explanations)} instances")
        
        # Log to MLflow
        if self.log_to_mlflow:
            try:
                mlflow.log_metric("num_explanations", len(explanations))
                for i, feature in enumerate(report['top_features'][:5]):
                    mlflow.log_param(f"top_feature_{i+1}", feature)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return report


class TextExplainer:
    """
    Specialized explainer for text-based sentiment models.
    Provides word-level explanations for text inputs.
    """
    
    def __init__(self, model: Any, vectorizer: Any, class_names: Optional[List[str]] = None):
        """
        Initialize text explainer.
        
        Args:
            model: Trained sentiment model
            vectorizer: Text vectorizer (TfidfVectorizer, CountVectorizer, etc.)
            class_names: Names of sentiment classes
        """
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ['negative', 'neutral', 'positive']
    
    def explain_text_prediction(self, text: str, num_features: int = 10) -> Explanation:
        """
        Explain prediction for a text input.
        
        Args:
            text: Input text to explain
            num_features: Number of top words to show
            
        Returns:
            Explanation with word-level importance
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        # Create text explainer
        explainer = LimeTextExplainer(class_names=self.class_names)
        
        # Define prediction function
        def predict_fn(texts):
            vectors = self.vectorizer.transform(texts)
            return self.model.predict_proba(vectors)
        
        # Generate explanation
        exp = explainer.explain_instance(text, predict_fn, num_features=num_features)
        
        # Get prediction
        vector = self.vectorizer.transform([text])
        prediction_proba = self.model.predict_proba(vector)[0]
        prediction = np.argmax(prediction_proba)
        confidence = float(prediction_proba[prediction])
        
        # Extract word importance
        word_importance = dict(exp.as_list())
        
        logger.info(f"Generated text explanation for: '{text[:50]}...'")
        
        return Explanation(
            method='LIME_Text',
            feature_importance=word_importance,
            prediction=prediction,
            confidence=confidence,
            visualization_data={'lime_exp': exp, 'text': text}
        )
    
    def visualize_text_explanation(self, explanation: Explanation, save_path: Optional[str] = None) -> None:
        """
        Visualize word-level importance for text.
        
        Args:
            explanation: Text explanation to visualize
            save_path: Path to save visualization
        """
        # Sort words by absolute importance
        sorted_words = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]
        
        words, importances = zip(*sorted_words)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in importances]
        plt.barh(range(len(words)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Word Importance')
        plt.title(f'Text Explanation\nPrediction: {self.class_names[explanation.prediction]} (Confidence: {explanation.confidence:.2%})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved text explanation to {save_path}")
        
        plt.close()
