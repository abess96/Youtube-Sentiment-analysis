"""
Active learning framework for intelligent sample selection.
Task 8.2: Build active learning framework
"""

import numpy as np
from sklearn.base import BaseEstimator
from typing import List, Tuple, Dict, Any
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class ActiveLearner:
    """Active learning for intelligent sample selection."""
    
    def __init__(self, model: BaseEstimator, strategy: str = 'uncertainty'):
        self.model = model
        self.strategy = strategy
        self.labeled_indices = []
        self.query_history = []
        
    def uncertainty_sampling(self, X_pool, n_samples: int = 10) -> np.ndarray:
        """Select samples with highest prediction uncertainty."""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must support predict_proba for uncertainty sampling")
        
        probas = self.model.predict_proba(X_pool)
        
        # Calculate uncertainty (entropy or margin)
        uncertainties = entropy(probas.T)
        
        # Select top uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        return uncertain_indices
    
    def margin_sampling(self, X_pool, n_samples: int = 10) -> np.ndarray:
        """Select samples with smallest margin between top predictions."""
        probas = self.model.predict_proba(X_pool)
        
        # Sort probabilities
        sorted_probas = np.sort(probas, axis=1)
        
        # Calculate margin (difference between top 2)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        
        # Select samples with smallest margins
        margin_indices = np.argsort(margins)[:n_samples]
        return margin_indices
    
    def diversity_sampling(self, X_pool, n_samples: int = 10) -> np.ndarray:
        """Select diverse samples using k-means clustering."""
        from sklearn.cluster import KMeans
        
        # Cluster pool samples
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(X_pool)
        
        # Select samples closest to cluster centers
        diverse_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(X_pool - center, axis=1)
            diverse_indices.append(np.argmin(distances))
        
        return np.array(diverse_indices)
    
    def query(self, X_pool, n_samples: int = 10) -> np.ndarray:
        """Query samples based on selected strategy."""
        if self.strategy == 'uncertainty':
            indices = self.uncertainty_sampling(X_pool, n_samples)
        elif self.strategy == 'margin':
            indices = self.margin_sampling(X_pool, n_samples)
        elif self.strategy == 'diversity':
            indices = self.diversity_sampling(X_pool, n_samples)
        else:
            # Random sampling as fallback
            indices = np.random.choice(len(X_pool), n_samples, replace=False)
        
        self.query_history.append({
            'strategy': self.strategy,
            'n_samples': n_samples,
            'indices': indices.tolist()
        })
        
        logger.info(f"Queried {n_samples} samples using {self.strategy} strategy")
        return indices
    
    def teach(self, X_new, y_new):
        """Update model with newly labeled samples."""
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            # Retrain if partial_fit not available
            logger.warning("Model doesn't support partial_fit, full retraining required")
        
        self.labeled_indices.extend(range(len(X_new)))
        logger.info(f"Model updated with {len(X_new)} new labeled samples")
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about query history."""
        return {
            'total_queries': len(self.query_history),
            'total_samples_queried': sum(q['n_samples'] for q in self.query_history),
            'strategies_used': list(set(q['strategy'] for q in self.query_history))
        }
