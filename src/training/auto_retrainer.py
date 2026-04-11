"""
Automated retraining pipeline with drift detection.
Task 8.3: Implement automated retraining pipeline
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional, Callable, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """Automated model retraining with scheduling and triggers."""
    
    def __init__(self, model: BaseEstimator, drift_detector=None,
                 retrain_threshold: float = 0.05, min_samples: int = 100):
        self.model = model
        self.drift_detector = drift_detector
        self.retrain_threshold = retrain_threshold
        self.min_samples = min_samples
        self.retrain_history = []
        self.last_retrain_time = None
        self.performance_baseline = None
        
    def should_retrain(self, X_new, y_new=None, y_pred=None) -> Tuple[bool, str]:
        """Determine if retraining is needed."""
        reasons = []
        
        # Check sample count
        if len(X_new) < self.min_samples:
            return False, "Insufficient samples for retraining"
        
        # Check drift
        if self.drift_detector and hasattr(self.drift_detector, 'reference_data'):
            drift_report = self.drift_detector.monitor(X_new, y_pred, y_new)
            
            if drift_report['overall_status'] in ['warning', 'critical']:
                reasons.append(f"Drift detected: {drift_report['overall_status']}")
        
        # Check performance degradation
        if y_new is not None and y_pred is not None:
            current_accuracy = accuracy_score(y_new, y_pred)
            
            if self.performance_baseline:
                performance_drop = self.performance_baseline - current_accuracy
                if performance_drop > self.retrain_threshold:
                    reasons.append(f"Performance drop: {performance_drop:.3f}")
        
        # Check time-based trigger
        if self.last_retrain_time:
            days_since_retrain = (datetime.now() - self.last_retrain_time).days
            if days_since_retrain > 30:  # Monthly retraining
                reasons.append(f"Scheduled retrain: {days_since_retrain} days since last")
        
        should_retrain = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should_retrain, reason_str
    
    def validate_data(self, X, y) -> Tuple[bool, str]:
        """Validate data quality before retraining."""
        # Check for missing values
        if np.isnan(X).any():
            return False, "Data contains NaN values"
        
        # Check label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        min_class_samples = counts.min()
        
        if min_class_samples < 10:
            return False, f"Insufficient samples in minority class: {min_class_samples}"
        
        # Check class imbalance
        max_imbalance = counts.max() / counts.min()
        if max_imbalance > 100:
            logger.warning(f"High class imbalance detected: {max_imbalance:.1f}:1")
        
        return True, "Data validation passed"
    
    def retrain(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """Retrain model with new data."""
        logger.info("Starting model retraining...")
        
        # Validate data
        is_valid, message = self.validate_data(X_train, y_train)
        if not is_valid:
            logger.error(f"Data validation failed: {message}")
            return {'success': False, 'error': message}
        
        # Split validation set if not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Train model
        try:
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            val_accuracy = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val, average='weighted')
            
            # Update baseline
            self.performance_baseline = val_accuracy
            self.last_retrain_time = datetime.now()
            
            # Record history
            retrain_record = {
                'timestamp': self.last_retrain_time.isoformat(),
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'val_f1': float(val_f1),
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val)
            }
            self.retrain_history.append(retrain_record)
            
            logger.info(f"Retraining completed - Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
            
            return {
                'success': True,
                'metrics': retrain_record
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def auto_retrain_pipeline(self, X_new, y_new, X_val=None, y_val=None) -> Dict[str, Any]:
        """Complete automated retraining pipeline."""
        # Make predictions on new data
        y_pred = self.model.predict(X_new)
        
        # Check if retraining needed
        should_retrain, reason = self.should_retrain(X_new, y_new, y_pred)
        
        if not should_retrain:
            logger.info(f"Retraining not triggered: {reason}")
            return {
                'retrained': False,
                'reason': reason
            }
        
        logger.info(f"Retraining triggered: {reason}")
        
        # Perform retraining
        result = self.retrain(X_new, y_new, X_val, y_val)
        
        if result['success']:
            return {
                'retrained': True,
                'reason': reason,
                'metrics': result['metrics']
            }
        else:
            return {
                'retrained': False,
                'reason': f"Retraining failed: {result['error']}"
            }
    
    def save_retrain_history(self, path: str):
        """Save retraining history."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.retrain_history, f, indent=2)
        logger.info(f"Retrain history saved to {path}")
    
    def save_model(self, path: str):
        """Save retrained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")
