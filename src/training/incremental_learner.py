"""
Incremental learning system for online model updates.
Task 8.1: Create incremental learning system
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from typing import Dict, Any, Optional
import logging
import pickle
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class IncrementalLearner:
    """Incremental learning for online model updates."""
    
    def __init__(self, base_model: Optional[BaseEstimator] = None, 
                 learning_rate: str = 'optimal', batch_size: int = 32):
        self.base_model = base_model or SGDClassifier(
            loss='log_loss', learning_rate=learning_rate, random_state=42
        )
        self.batch_size = batch_size
        self.update_count = 0
        self.performance_history = []
        
    def partial_fit(self, X, y, classes=None):
        """Update model with new batch of data."""
        if not hasattr(self.base_model, 'partial_fit'):
            raise ValueError("Base model must support partial_fit for incremental learning")
        
        if classes is None and self.update_count == 0:
            classes = np.unique(y)
        
        self.base_model.partial_fit(X, y, classes=classes)
        self.update_count += 1
        logger.info(f"Model updated with batch {self.update_count}")
        
    def fit_streaming(self, X_stream, y_stream, classes=None):
        """Fit model on streaming data in batches."""
        n_samples = len(X_stream)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            X_batch = X_stream[start_idx:end_idx]
            y_batch = y_stream[start_idx:end_idx]
            
            self.partial_fit(X_batch, y_batch, classes=classes)
            
        logger.info(f"Streaming fit completed: {n_batches} batches processed")
        
    def predict(self, X):
        """Make predictions."""
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        return None
    
    def adapt_to_drift(self, X_new, y_new, drift_severity: str = 'moderate'):
        """Adapt model based on detected drift severity."""
        if drift_severity == 'severe':
            # Reset and retrain for severe drift
            logger.warning("Severe drift detected - resetting model")
            self.base_model = SGDClassifier(loss='log_loss', random_state=42)
            self.update_count = 0
            
        # Update with new data
        self.partial_fit(X_new, y_new)
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model': self.base_model,
            'update_count': self.update_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.base_model = checkpoint['model']
        self.update_count = checkpoint['update_count']
        logger.info(f"Checkpoint loaded from {path}")
