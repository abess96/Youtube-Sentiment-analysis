"""
Comprehensive model evaluation framework with advanced metrics and cross-validation.
Task 4.1: Create comprehensive model evaluation framework
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import logging
from typing import Dict, Any, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with multiple metrics and statistical testing."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def evaluate(self, model, X, y) -> Dict[str, Any]:
        """Evaluate model with comprehensive metrics."""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': float(accuracy_score(y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
            'matthews_corrcoef': float(matthews_corrcoef(y, predictions)),
            'cohen_kappa': float(cohen_kappa_score(y, predictions))
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y, predictions, average=None)
        metrics['per_class'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        }
        
        # Weighted averages
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y, predictions, average='weighted')
        metrics['weighted'] = {
            'precision': float(p_w),
            'recall': float(r_w),
            'f1_score': float(f1_w)
        }
        
        # Macro averages
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y, predictions, average='macro')
        metrics['macro'] = {
            'precision': float(p_m),
            'recall': float(r_m),
            'f1_score': float(f1_m)
        }
        
        # ROC AUC for multiclass
        if probabilities is not None:
            try:
                metrics['roc_auc_ovr'] = float(roc_auc_score(y, probabilities, multi_class='ovr', average='weighted'))
            except:
                metrics['roc_auc_ovr'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, predictions).tolist()
        
        return metrics
    
    def cross_validate(self, model, X, y) -> Dict[str, Any]:
        """Perform stratified cross-validation with statistical significance testing."""
        cv_scores = {
            'accuracy': [],
            'f1_weighted': [],
            'precision_weighted': [],
            'recall_weighted': []
        }
        
        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and train model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_clone.predict(X_val)
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            
            p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
            cv_scores['f1_weighted'].append(f1)
            cv_scores['precision_weighted'].append(p)
            cv_scores['recall_weighted'].append(r)
        
        # Calculate statistics
        results = {}
        for metric, scores in cv_scores.items():
            results[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'scores': [float(s) for s in scores]
            }
            
            # 95% confidence interval
            ci = stats.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=stats.sem(scores))
            results[metric]['ci_95'] = [float(ci[0]), float(ci[1])]
        
        return results
    
    def compare_models(self, model1, model2, X, y) -> Dict[str, Any]:
        """Compare two models with statistical significance testing."""
        scores1, scores2 = [], []
        
        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            from sklearn.base import clone
            m1 = clone(model1)
            m2 = clone(model2)
            
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)
            
            scores1.append(accuracy_score(y_val, m1.predict(X_val)))
            scores2.append(accuracy_score(y_val, m2.predict(X_val)))
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'model1_mean': float(np.mean(scores1)),
            'model2_mean': float(np.mean(scores2)),
            'difference': float(np.mean(scores1) - np.mean(scores2)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str = None) -> str:
        """Generate evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nOverall Metrics:")
        report.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        report.append(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        report.append(f"  Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.4f}")
        report.append(f"  Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}")
        
        if 'weighted' in metrics:
            report.append(f"\nWeighted Averages:")
            report.append(f"  Precision: {metrics['weighted']['precision']:.4f}")
            report.append(f"  Recall: {metrics['weighted']['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['weighted']['f1_score']:.4f}")
        
        report_text = "\n".join(report)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
