"""
Model Debugging and Analysis Tools.
Provides tools for analyzing misclassifications, decision boundaries, and failure modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

try:
    import mlflow
    from src.utils.mlflow_config import setup_mlflow
    MLFLOW_AVAILABLE = True
    setup_mlflow()
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MisclassificationAnalysis:
    """Analysis result for misclassified examples."""
    total_misclassified: int
    misclassification_rate: float
    confusion_patterns: Dict[str, int]
    difficult_examples: List[Dict[str, Any]]
    class_specific_errors: Dict[str, Dict[str, Any]]


class ModelDebugger:
    """
    Comprehensive debugging tools for model analysis.
    Analyzes misclassifications, decision boundaries, and failure modes.
    """
    
    def __init__(self, model: Any, class_names: Optional[List[str]] = None, log_to_mlflow: bool = True):
        """
        Initialize model debugger.
        
        Args:
            model: Trained model
            class_names: Names of classes
            log_to_mlflow: Whether to automatically log to MLflow
        """
        self.model = model
        self.class_names = class_names or ['negative', 'neutral', 'positive']
        self.log_to_mlflow = log_to_mlflow and MLFLOW_AVAILABLE
    
    def analyze_misclassifications(self, X: np.ndarray, y_true: np.ndarray,
                                   feature_names: Optional[List[str]] = None,
                                   top_k: int = 10) -> MisclassificationAnalysis:
        """
        Analyze misclassified examples in detail.
        
        Args:
            X: Input features
            y_true: True labels
            feature_names: Names of features
            top_k: Number of top difficult examples to return
            
        Returns:
            Detailed misclassification analysis
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = None
        
        # Find misclassifications
        misclassified_mask = y_pred != y_true
        misclassified_indices = np.where(misclassified_mask)[0]
        
        total_misclassified = len(misclassified_indices)
        misclassification_rate = total_misclassified / len(y_true)
        
        # Analyze confusion patterns
        confusion_patterns = {}
        for idx in misclassified_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            pattern = f"{true_class} -> {pred_class}"
            confusion_patterns[pattern] = confusion_patterns.get(pattern, 0) + 1
        
        # Find most difficult examples (lowest confidence on correct class)
        difficult_examples = []
        if y_proba is not None:
            for idx in misclassified_indices:
                confidence = y_proba[idx, y_pred[idx]]
                true_class_prob = y_proba[idx, y_true[idx]]
                
                difficult_examples.append({
                    'index': int(idx),
                    'true_label': self.class_names[y_true[idx]],
                    'predicted_label': self.class_names[y_pred[idx]],
                    'prediction_confidence': float(confidence),
                    'true_class_probability': float(true_class_prob),
                    'features': X[idx].tolist() if feature_names is None else dict(zip(feature_names, X[idx]))
                })
            
            # Sort by prediction confidence (most confident mistakes)
            difficult_examples.sort(key=lambda x: x['prediction_confidence'], reverse=True)
            difficult_examples = difficult_examples[:top_k]
        
        # Class-specific error analysis
        class_specific_errors = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_true == class_idx
            class_misclassified = np.sum(misclassified_mask & class_mask)
            class_total = np.sum(class_mask)
            
            class_specific_errors[class_name] = {
                'total_samples': int(class_total),
                'misclassified': int(class_misclassified),
                'error_rate': float(class_misclassified / class_total) if class_total > 0 else 0.0
            }
        
        logger.info(f"Analyzed {total_misclassified} misclassifications ({misclassification_rate:.2%})")
        
        return MisclassificationAnalysis(
            total_misclassified=total_misclassified,
            misclassification_rate=misclassification_rate,
            confusion_patterns=confusion_patterns,
            difficult_examples=difficult_examples,
            class_specific_errors=class_specific_errors
        )
    
    def visualize_confusion_patterns(self, analysis: MisclassificationAnalysis,
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize confusion patterns.
        
        Args:
            analysis: Misclassification analysis result
            save_path: Path to save visualization
        """
        patterns = analysis.confusion_patterns
        
        if not patterns:
            logger.warning("No confusion patterns to visualize")
            return
        
        # Sort patterns by frequency
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        pattern_names, counts = zip(*sorted_patterns)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(pattern_names)), counts, color='coral', alpha=0.7)
        plt.xticks(range(len(pattern_names)), pattern_names, rotation=45, ha='right')
        plt.xlabel('Confusion Pattern (True -> Predicted)')
        plt.ylabel('Count')
        plt.title('Misclassification Patterns')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion pattern visualization to {save_path}")
            
            # Log to MLflow
            if self.log_to_mlflow:
                try:
                    mlflow.log_artifact(save_path, artifact_path="debugging")
                except Exception as e:
                    logger.warning(f"Failed to log artifact to MLflow: {e}")
        
        plt.close()
    
    def analyze_decision_boundary(self, X: np.ndarray, y: np.ndarray,
                                 feature_indices: Tuple[int, int] = (0, 1),
                                 resolution: int = 100) -> Dict[str, Any]:
        """
        Analyze decision boundary for 2D feature space.
        
        Args:
            X: Input features
            y: Labels
            feature_indices: Indices of two features to visualize
            resolution: Grid resolution for boundary
            
        Returns:
            Decision boundary analysis
        """
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features for decision boundary visualization")
        
        # Extract two features
        X_2d = X[:, feature_indices]
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict on grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Create full feature space (pad with zeros for other features)
        if X.shape[1] > 2:
            full_grid = np.zeros((len(grid_points), X.shape[1]))
            full_grid[:, feature_indices[0]] = grid_points[:, 0]
            full_grid[:, feature_indices[1]] = grid_points[:, 1]
            # Use mean values for other features
            for i in range(X.shape[1]):
                if i not in feature_indices:
                    full_grid[:, i] = X[:, i].mean()
        else:
            full_grid = grid_points
        
        Z = self.model.predict(full_grid)
        Z = Z.reshape(xx.shape)
        
        logger.info(f"Analyzed decision boundary for features {feature_indices}")
        
        return {
            'xx': xx,
            'yy': yy,
            'Z': Z,
            'X_2d': X_2d,
            'y': y,
            'feature_indices': feature_indices
        }
    
    def visualize_decision_boundary(self, boundary_data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize decision boundary.
        
        Args:
            boundary_data: Decision boundary data from analyze_decision_boundary
            save_path: Path to save visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Plot decision boundary
        plt.contourf(boundary_data['xx'], boundary_data['yy'], boundary_data['Z'],
                    alpha=0.3, cmap='viridis')
        
        # Plot data points
        scatter = plt.scatter(
            boundary_data['X_2d'][:, 0],
            boundary_data['X_2d'][:, 1],
            c=boundary_data['y'],
            cmap='viridis',
            edgecolors='black',
            alpha=0.7
        )
        
        plt.colorbar(scatter, label='Class')
        plt.xlabel(f'Feature {boundary_data["feature_indices"][0]}')
        plt.ylabel(f'Feature {boundary_data["feature_indices"][1]}')
        plt.title('Decision Boundary Visualization')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved decision boundary to {save_path}")
        
        plt.close()
    
    def identify_failure_modes(self, X: np.ndarray, y_true: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Identify common failure modes and patterns.
        
        Args:
            X: Input features
            y_true: True labels
            feature_names: Names of features
            
        Returns:
            Analysis of failure modes
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = None
        
        # Find misclassifications
        misclassified_mask = y_pred != y_true
        misclassified_indices = np.where(misclassified_mask)[0]
        
        failure_modes = {
            'low_confidence_errors': [],
            'high_confidence_errors': [],
            'boundary_cases': [],
            'systematic_biases': {}
        }
        
        if y_proba is not None:
            for idx in misclassified_indices:
                confidence = y_proba[idx, y_pred[idx]]
                
                if confidence < 0.6:
                    failure_modes['low_confidence_errors'].append(int(idx))
                elif confidence > 0.8:
                    failure_modes['high_confidence_errors'].append(int(idx))
                else:
                    failure_modes['boundary_cases'].append(int(idx))
        
        # Analyze systematic biases
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_true == class_idx
            class_predictions = y_pred[class_mask]
            
            # Count prediction distribution for this true class
            pred_distribution = Counter(class_predictions)
            
            failure_modes['systematic_biases'][class_name] = {
                'predicted_as': {
                    self.class_names[pred_class]: int(count)
                    for pred_class, count in pred_distribution.items()
                }
            }
        
        logger.info(f"Identified failure modes: {len(failure_modes['low_confidence_errors'])} low-confidence, "
                   f"{len(failure_modes['high_confidence_errors'])} high-confidence errors")
        
        return failure_modes
    
    def generate_debug_report(self, X: np.ndarray, y_true: np.ndarray,
                             feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive debugging report.
        
        Args:
            X: Input features
            y_true: True labels
            feature_names: Names of features
            
        Returns:
            Complete debugging report
        """
        # Analyze misclassifications
        misclass_analysis = self.analyze_misclassifications(X, y_true, feature_names)
        
        # Identify failure modes
        failure_modes = self.identify_failure_modes(X, y_true, feature_names)
        
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        report = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'misclassification_analysis': {
                'total_misclassified': misclass_analysis.total_misclassified,
                'misclassification_rate': misclass_analysis.misclassification_rate,
                'confusion_patterns': misclass_analysis.confusion_patterns,
                'class_specific_errors': misclass_analysis.class_specific_errors,
                'top_difficult_examples': misclass_analysis.difficult_examples[:5]
            },
            'failure_modes': failure_modes,
            'recommendations': self._generate_recommendations(misclass_analysis, failure_modes)
        }
        
        logger.info("Generated comprehensive debug report")
        
        # Log to MLflow
        if self.log_to_mlflow:
            try:
                for metric, value in report['overall_metrics'].items():
                    mlflow.log_metric(f"debug_{metric}", value)
                mlflow.log_metric("misclassification_rate", 
                                report['misclassification_analysis']['misclassification_rate'])
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return report
    
    def _generate_recommendations(self, misclass_analysis: MisclassificationAnalysis,
                                 failure_modes: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis.
        
        Args:
            misclass_analysis: Misclassification analysis
            failure_modes: Failure mode analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check misclassification rate
        if misclass_analysis.misclassification_rate > 0.3:
            recommendations.append("High misclassification rate detected. Consider collecting more training data or trying different model architectures.")
        
        # Check class-specific errors
        for class_name, errors in misclass_analysis.class_specific_errors.items():
            if errors['error_rate'] > 0.4:
                recommendations.append(f"Class '{class_name}' has high error rate ({errors['error_rate']:.2%}). Consider class-specific data augmentation or rebalancing.")
        
        # Check high-confidence errors
        if len(failure_modes['high_confidence_errors']) > 10:
            recommendations.append(f"Found {len(failure_modes['high_confidence_errors'])} high-confidence errors. Model may be overconfident - consider calibration.")
        
        # Check systematic biases
        for class_name, bias_info in failure_modes['systematic_biases'].items():
            pred_dist = bias_info['predicted_as']
            if len(pred_dist) > 0:
                most_common_pred = max(pred_dist.items(), key=lambda x: x[1])
                if most_common_pred[0] != class_name and most_common_pred[1] > 20:
                    recommendations.append(f"Systematic bias detected: '{class_name}' often predicted as '{most_common_pred[0]}'. Review feature engineering.")
        
        if not recommendations:
            recommendations.append("Model performance looks good. Continue monitoring on new data.")
        
        return recommendations


class ErrorAnalyzer:
    """
    Specialized analyzer for error patterns in text classification.
    """
    
    def __init__(self, model: Any, vectorizer: Any = None):
        """
        Initialize error analyzer.
        
        Args:
            model: Trained model
            vectorizer: Text vectorizer (optional)
        """
        self.model = model
        self.vectorizer = vectorizer
    
    def analyze_text_errors(self, texts: List[str], y_true: np.ndarray,
                           y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze errors in text classification.
        
        Args:
            texts: Input texts
            y_true: True labels
            y_pred: Predicted labels (optional)
            
        Returns:
            Text-specific error analysis
        """
        if y_pred is None:
            if self.vectorizer is not None:
                X = self.vectorizer.transform(texts)
                y_pred = self.model.predict(X)
            else:
                raise ValueError("Either provide y_pred or set vectorizer")
        
        # Find errors
        error_mask = y_pred != y_true
        error_indices = np.where(error_mask)[0]
        
        # Analyze text characteristics of errors
        error_texts = [texts[i] for i in error_indices]
        
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(texts),
            'avg_error_length': np.mean([len(t.split()) for t in error_texts]) if error_texts else 0,
            'error_examples': [
                {
                    'text': texts[i][:100],
                    'true_label': int(y_true[i]),
                    'predicted_label': int(y_pred[i])
                }
                for i in error_indices[:10]
            ]
        }
        
        logger.info(f"Analyzed {len(error_indices)} text classification errors")
        
        return error_analysis
