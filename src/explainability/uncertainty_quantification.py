"""
Uncertainty Quantification System for Model Predictions.
Implements confidence intervals, Monte Carlo dropout, and uncertainty-based filtering.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from scipy import stats

try:
    import mlflow
    from src.config.mlflow_config import setup_mlflow
    MLFLOW_AVAILABLE = True
    setup_mlflow()
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Structured uncertainty quantification result."""
    prediction: Any
    mean_confidence: float
    std_confidence: float
    confidence_interval: Tuple[float, float]
    uncertainty_score: float
    is_reliable: bool


class UncertaintyQuantifier:
    """
    System for quantifying prediction uncertainty.
    Supports bootstrap sampling and confidence interval estimation.
    """
    
    def __init__(self, model: Any, confidence_level: float = 0.95,
                 reliability_threshold: float = 0.7, log_to_mlflow: bool = True):
        """
        Initialize uncertainty quantifier.
        
        Args:
            model: Trained model with predict_proba method
            confidence_level: Confidence level for intervals (default: 0.95)
            reliability_threshold: Minimum confidence for reliable predictions
            log_to_mlflow: Whether to automatically log to MLflow
        """
        self.model = model
        self.confidence_level = confidence_level
        self.reliability_threshold = reliability_threshold
        self.log_to_mlflow = log_to_mlflow and MLFLOW_AVAILABLE
    
    def bootstrap_uncertainty(self, X: np.ndarray, n_iterations: int = 100,
                             sample_ratio: float = 0.8) -> List[UncertaintyResult]:
        """
        Estimate uncertainty using bootstrap sampling.
        
        Args:
            X: Input features
            n_iterations: Number of bootstrap iterations
            sample_ratio: Ratio of data to sample in each iteration
            
        Returns:
            List of uncertainty results for each instance
        """
        n_samples = len(X)
        sample_size = int(n_samples * sample_ratio)
        
        # Store predictions from each bootstrap iteration
        all_predictions = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_boot = X[indices]
            
            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_boot)
            else:
                # For models without predict_proba, use binary predictions
                pred = self.model.predict(X_boot)
                proba = np.eye(len(np.unique(pred)))[pred]
            
            all_predictions.append(proba)
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_proba = all_predictions.mean(axis=0)
        std_proba = all_predictions.std(axis=0)
        
        # Generate results
        results = []
        for i in range(sample_size):
            prediction = np.argmax(mean_proba[i])
            mean_conf = float(mean_proba[i, prediction])
            std_conf = float(std_proba[i, prediction])
            
            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            ci_lower = max(0, mean_conf - z_score * std_conf)
            ci_upper = min(1, mean_conf + z_score * std_conf)
            
            # Uncertainty score (higher = more uncertain)
            uncertainty = std_conf / (mean_conf + 1e-10)
            
            results.append(UncertaintyResult(
                prediction=prediction,
                mean_confidence=mean_conf,
                std_confidence=std_conf,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_score=uncertainty,
                is_reliable=mean_conf >= self.reliability_threshold
            ))
        
        logger.info(f"Completed bootstrap uncertainty estimation with {n_iterations} iterations")
        
        return results
    
    def predict_with_uncertainty(self, X: np.ndarray, method: str = 'bootstrap',
                                **kwargs) -> List[UncertaintyResult]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features
            method: Uncertainty estimation method ('bootstrap' or 'ensemble')
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of predictions with uncertainty
        """
        if method == 'bootstrap':
            return self.bootstrap_uncertainty(X, **kwargs)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def filter_by_uncertainty(self, X: np.ndarray, results: List[UncertaintyResult],
                             max_uncertainty: float = 0.3) -> Tuple[np.ndarray, List[UncertaintyResult]]:
        """
        Filter predictions based on uncertainty threshold.
        
        Args:
            X: Input features
            results: Uncertainty results
            max_uncertainty: Maximum allowed uncertainty score
            
        Returns:
            Filtered features and results
        """
        reliable_indices = [
            i for i, r in enumerate(results)
            if r.uncertainty_score <= max_uncertainty
        ]
        
        filtered_X = X[reliable_indices]
        filtered_results = [results[i] for i in reliable_indices]
        
        logger.info(f"Filtered {len(filtered_results)}/{len(results)} predictions (uncertainty <= {max_uncertainty})")
        
        return filtered_X, filtered_results
    
    def get_uncertainty_statistics(self, results: List[UncertaintyResult]) -> Dict[str, Any]:
        """
        Calculate statistics about uncertainty across predictions.
        
        Args:
            results: List of uncertainty results
            
        Returns:
            Dictionary with uncertainty statistics
        """
        uncertainties = [r.uncertainty_score for r in results]
        confidences = [r.mean_confidence for r in results]
        reliable_count = sum(1 for r in results if r.is_reliable)
        
        stats_dict = {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'min_uncertainty': float(np.min(uncertainties)),
            'mean_confidence': float(np.mean(confidences)),
            'reliable_predictions': reliable_count,
            'reliability_rate': reliable_count / len(results),
            'total_predictions': len(results)
        }
        
        # Log to MLflow
        if self.log_to_mlflow:
            try:
                mlflow.log_metric("mean_uncertainty", stats_dict['mean_uncertainty'])
                mlflow.log_metric("mean_confidence", stats_dict['mean_confidence'])
                mlflow.log_metric("reliability_rate", stats_dict['reliability_rate'])
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return stats_dict


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation in neural networks.
    Requires models with dropout layers.
    """
    
    def __init__(self, model: Any, n_iterations: int = 50):
        """
        Initialize MC Dropout.
        
        Args:
            model: Neural network model with dropout
            n_iterations: Number of forward passes
        """
        self.model = model
        self.n_iterations = n_iterations
    
    def predict_with_uncertainty(self, X: np.ndarray) -> List[UncertaintyResult]:
        """
        Predict with uncertainty using MC Dropout.
        
        Args:
            X: Input features
            
        Returns:
            List of predictions with uncertainty estimates
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for MC Dropout")
        
        # Enable dropout during inference
        self.model.train()
        
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iterations):
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = X
                
                output = self.model(X_tensor)
                
                if hasattr(output, 'softmax'):
                    proba = torch.softmax(output, dim=1).numpy()
                else:
                    proba = output.numpy()
                
                all_predictions.append(proba)
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_proba = all_predictions.mean(axis=0)
        std_proba = all_predictions.std(axis=0)
        
        # Generate results
        results = []
        for i in range(len(X)):
            prediction = np.argmax(mean_proba[i])
            mean_conf = float(mean_proba[i, prediction])
            std_conf = float(std_proba[i, prediction])
            
            # Confidence interval
            z_score = 1.96  # 95% confidence
            ci_lower = max(0, mean_conf - z_score * std_conf)
            ci_upper = min(1, mean_conf + z_score * std_conf)
            
            uncertainty = std_conf / (mean_conf + 1e-10)
            
            results.append(UncertaintyResult(
                prediction=prediction,
                mean_confidence=mean_conf,
                std_confidence=std_conf,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_score=uncertainty,
                is_reliable=mean_conf >= 0.7
            ))
        
        # Restore model to eval mode
        self.model.eval()
        
        logger.info(f"Completed MC Dropout with {self.n_iterations} iterations")
        
        return results


class EnsembleUncertainty:
    """
    Uncertainty estimation using ensemble disagreement.
    """
    
    def __init__(self, models: List[Any]):
        """
        Initialize ensemble uncertainty estimator.
        
        Args:
            models: List of trained models
        """
        self.models = models
    
    def predict_with_uncertainty(self, X: np.ndarray) -> List[UncertaintyResult]:
        """
        Predict with uncertainty based on ensemble disagreement.
        
        Args:
            X: Input features
            
        Returns:
            List of predictions with uncertainty
        """
        all_predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.eye(n_classes)[pred]
            
            all_predictions.append(proba)
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_proba = all_predictions.mean(axis=0)
        std_proba = all_predictions.std(axis=0)
        
        # Generate results
        results = []
        for i in range(len(X)):
            prediction = np.argmax(mean_proba[i])
            mean_conf = float(mean_proba[i, prediction])
            std_conf = float(std_proba[i, prediction])
            
            # Confidence interval
            z_score = 1.96
            ci_lower = max(0, mean_conf - z_score * std_conf)
            ci_upper = min(1, mean_conf + z_score * std_conf)
            
            # Uncertainty based on ensemble disagreement
            uncertainty = std_conf / (mean_conf + 1e-10)
            
            results.append(UncertaintyResult(
                prediction=prediction,
                mean_confidence=mean_conf,
                std_confidence=std_conf,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_score=uncertainty,
                is_reliable=mean_conf >= 0.7
            ))
        
        logger.info(f"Completed ensemble uncertainty estimation with {len(self.models)} models")
        
        return results
    
    def get_disagreement_rate(self, X: np.ndarray) -> float:
        """
        Calculate rate of disagreement among ensemble models.
        
        Args:
            X: Input features
            
        Returns:
            Disagreement rate (0-1)
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate disagreement
        disagreements = 0
        for i in range(len(X)):
            if len(np.unique(predictions[:, i])) > 1:
                disagreements += 1
        
        disagreement_rate = disagreements / len(X)
        
        logger.info(f"Ensemble disagreement rate: {disagreement_rate:.2%}")
        
        return disagreement_rate


class CalibrationAnalyzer:
    """
    Analyze and improve model calibration.
    """
    
    def __init__(self, model: Any):
        """
        Initialize calibration analyzer.
        
        Args:
            model: Trained model
        """
        self.model = model
        self.calibrator = None
    
    def analyze_calibration(self, X: np.ndarray, y: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        Analyze model calibration using reliability diagrams.
        
        Args:
            X: Input features
            y: True labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Calibration metrics
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        
        # Get predictions
        y_proba = self.model.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Calculate calibration for each class
        calibration_results = {}
        
        for class_idx in range(y_proba.shape[1]):
            y_binary = (y == class_idx).astype(int)
            prob_true, prob_pred = calibration_curve(
                y_binary, y_proba[:, class_idx], n_bins=n_bins
            )
            
            brier = brier_score_loss(y_binary, y_proba[:, class_idx])
            
            calibration_results[f'class_{class_idx}'] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist(),
                'brier_score': float(brier)
            }
        
        # Overall metrics
        overall_brier = brier_score_loss(
            (y == y_pred).astype(int),
            y_proba[np.arange(len(y)), y_pred]
        )
        
        logger.info(f"Calibration analysis complete. Overall Brier score: {overall_brier:.4f}")
        
        return {
            'calibration_curves': calibration_results,
            'overall_brier_score': float(overall_brier),
            'n_bins': n_bins
        }
    
    def calibrate_model(self, X: np.ndarray, y: np.ndarray, method: str = 'isotonic'):
        """
        Calibrate model probabilities.
        
        Args:
            X: Input features
            y: True labels
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        self.calibrator = CalibratedClassifierCV(
            self.model, method=method, cv='prefit'
        )
        self.calibrator.fit(X, y)
        
        logger.info(f"Model calibrated using {method} method")
    
    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """
        Make calibrated predictions.
        
        Args:
            X: Input features
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Model not calibrated. Call calibrate_model first.")
        
        return self.calibrator.predict_proba(X)
