"""
Bias detection and fairness evaluation for sentiment analysis models.
Task 4.2: Implement bias detection and fairness evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BiasDetector:
    """Detect demographic bias and evaluate fairness in model predictions."""
    
    def __init__(self):
        self.protected_attributes = []
        self.bias_metrics = {}
    
    def detect_sentiment_bias(self, y_true, y_pred, texts: List[str]) -> Dict[str, Any]:
        """Detect bias in sentiment predictions based on text characteristics."""
        bias_report = {}
        
        # Length bias
        lengths = [len(text.split()) for text in texts]
        short_mask = np.array(lengths) < np.median(lengths)
        long_mask = ~short_mask
        
        bias_report['length_bias'] = {
            'short_accuracy': float(np.mean(y_true[short_mask] == y_pred[short_mask])),
            'long_accuracy': float(np.mean(y_true[long_mask] == y_pred[long_mask])),
            'disparity': float(abs(np.mean(y_true[short_mask] == y_pred[short_mask]) - 
                                  np.mean(y_true[long_mask] == y_pred[long_mask])))
        }
        
        # Sentiment distribution bias
        for sentiment_class in np.unique(y_true):
            mask = y_true == sentiment_class
            if mask.sum() > 0:
                bias_report[f'class_{sentiment_class}_accuracy'] = float(np.mean(y_true[mask] == y_pred[mask]))
        
        return bias_report
    
    def calculate_demographic_parity(self, y_pred, protected_attr) -> float:
        """Calculate demographic parity difference."""
        groups = np.unique(protected_attr)
        if len(groups) < 2:
            return 0.0
        
        positive_rates = []
        for group in groups:
            mask = protected_attr == group
            positive_rate = np.mean(y_pred[mask] == 1) if mask.sum() > 0 else 0
            positive_rates.append(positive_rate)
        
        return float(max(positive_rates) - min(positive_rates))
    
    def calculate_equalized_odds(self, y_true, y_pred, protected_attr) -> Dict[str, float]:
        """Calculate equalized odds metrics."""
        groups = np.unique(protected_attr)
        if len(groups) < 2:
            return {'tpr_disparity': 0.0, 'fpr_disparity': 0.0}
        
        tprs, fprs = [], []
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # True Positive Rate
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tprs.append(tpr)
            
            # False Positive Rate
            fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
            tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs.append(fpr)
        
        return {
            'tpr_disparity': float(max(tprs) - min(tprs)) if tprs else 0.0,
            'fpr_disparity': float(max(fprs) - min(fprs)) if fprs else 0.0
        }
    
    def evaluate_fairness(self, y_true, y_pred, texts: List[str], 
                         protected_attr=None) -> Dict[str, Any]:
        """Comprehensive fairness evaluation."""
        fairness_metrics = {}
        
        # Sentiment bias
        fairness_metrics['sentiment_bias'] = self.detect_sentiment_bias(y_true, y_pred, texts)
        
        # Protected attribute fairness (if provided)
        if protected_attr is not None:
            fairness_metrics['demographic_parity'] = self.calculate_demographic_parity(y_pred, protected_attr)
            fairness_metrics['equalized_odds'] = self.calculate_equalized_odds(y_true, y_pred, protected_attr)
        
        # Overall fairness score (0-1, higher is better)
        length_disparity = fairness_metrics['sentiment_bias']['length_bias']['disparity']
        fairness_metrics['overall_fairness_score'] = float(1.0 - min(length_disparity, 1.0))
        
        return fairness_metrics
    
    def generate_mitigation_recommendations(self, fairness_metrics: Dict[str, Any]) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []
        
        # Check length bias
        if 'sentiment_bias' in fairness_metrics:
            length_bias = fairness_metrics['sentiment_bias']['length_bias']
            if length_bias['disparity'] > 0.1:
                recommendations.append(
                    f"High length bias detected (disparity: {length_bias['disparity']:.3f}). "
                    "Consider: 1) Augmenting training data with varied text lengths, "
                    "2) Using length-normalized features"
                )
        
        # Check demographic parity
        if 'demographic_parity' in fairness_metrics:
            dp = fairness_metrics['demographic_parity']
            if dp > 0.1:
                recommendations.append(
                    f"Demographic parity violation detected ({dp:.3f}). "
                    "Consider: 1) Reweighting training samples, 2) Using fairness constraints"
                )
        
        # Check equalized odds
        if 'equalized_odds' in fairness_metrics:
            eo = fairness_metrics['equalized_odds']
            if eo['tpr_disparity'] > 0.1 or eo['fpr_disparity'] > 0.1:
                recommendations.append(
                    f"Equalized odds violation detected (TPR: {eo['tpr_disparity']:.3f}, "
                    f"FPR: {eo['fpr_disparity']:.3f}). "
                    "Consider: 1) Post-processing calibration, 2) Adversarial debiasing"
                )
        
        if not recommendations:
            recommendations.append("No significant bias detected. Model appears fair.")
        
        return recommendations
