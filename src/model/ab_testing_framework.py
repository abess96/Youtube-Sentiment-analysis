"""
A/B testing framework for model comparison and evaluation.
Provides statistical significance testing and experiment result analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from datetime import datetime
from pathlib import Path
import json
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """
    Framework for conducting A/B tests between model versions.
    Includes statistical significance testing and result analysis.
    """
    
    def __init__(self, model_a: Any, model_b: Any,
                 model_a_name: str = "Model A",
                 model_b_name: str = "Model B"):
        """
        Initialize A/B testing framework.
        
        Args:
            model_a: First model (control)
            model_b: Second model (treatment)
            model_a_name: Name for model A
            model_b_name: Name for model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        
        self.results = {
            'model_a': {'name': model_a_name, 'metrics': {}},
            'model_b': {'name': model_b_name, 'metrics': {}},
            'comparison': {},
            'statistical_tests': {}
        }
        
        logger.info(f"A/B test initialized: {model_a_name} vs {model_b_name}")
    
    def run_ab_test(self, X_test: np.ndarray, y_test: np.ndarray,
                   metrics: Optional[List[str]] = None,
                   confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Run complete A/B test.
        
        Args:
            X_test: Test features
            y_test: Test labels
            metrics: List of metrics to evaluate
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Complete test results
        """
        logger.info("Running A/B test...")
        
        # Get predictions
        pred_a = self.model_a.predict(X_test)
        pred_b = self.model_b.predict(X_test)
        
        # Get probabilities if available
        try:
            proba_a = self.model_a.predict_proba(X_test)
            proba_b = self.model_b.predict_proba(X_test)
        except:
            proba_a = None
            proba_b = None
        
        # Calculate metrics
        default_metrics = ['accuracy', 'precision', 'recall', 'f1']
        metrics = metrics or default_metrics
        
        for metric in metrics:
            self.results['model_a']['metrics'][metric] = self._calculate_metric(
                y_test, pred_a, proba_a, metric
            )
            self.results['model_b']['metrics'][metric] = self._calculate_metric(
                y_test, pred_b, proba_b, metric
            )
        
        # Statistical comparison
        self._perform_statistical_tests(
            y_test, pred_a, pred_b, 
            confidence_level
        )
        
        # Generate comparison summary
        self._generate_comparison_summary()
        
        # Add metadata
        self.results['metadata'] = {
            'test_samples': len(X_test),
            'confidence_level': confidence_level,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("A/B test completed")
        return self.results
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray], 
                         metric: str) -> float:
        """Calculate a specific metric."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, log_loss
        )
        
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'roc_auc' and y_proba is not None:
            try:
                return roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                return 0.0
        elif metric == 'log_loss' and y_proba is not None:
            try:
                return log_loss(y_true, y_proba)
            except:
                return 0.0
        else:
            return 0.0
    
    def _perform_statistical_tests(self, y_true: np.ndarray,
                                   pred_a: np.ndarray, pred_b: np.ndarray,
                                   confidence_level: float) -> None:
        """Perform statistical significance tests."""
        
        # McNemar's test for paired predictions
        mcnemar_result = self._mcnemar_test(y_true, pred_a, pred_b)
        self.results['statistical_tests']['mcnemar'] = mcnemar_result
        
        # Bootstrap confidence intervals
        bootstrap_result = self._bootstrap_comparison(
            y_true, pred_a, pred_b, confidence_level
        )
        self.results['statistical_tests']['bootstrap'] = bootstrap_result
        
        # Effect size (Cohen's h)
        effect_size = self._calculate_effect_size(y_true, pred_a, pred_b)
        self.results['statistical_tests']['effect_size'] = effect_size
    
    def _mcnemar_test(self, y_true: np.ndarray, 
                     pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
        """
        Perform McNemar's test for paired predictions.
        Tests if there's a significant difference between two models.
        """
        # Create contingency table
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)
        
        # Count cases
        both_correct = np.sum(correct_a & correct_b)
        both_wrong = np.sum(~correct_a & ~correct_b)
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        b_correct_a_wrong = np.sum(~correct_a & correct_b)
        
        # McNemar's test statistic
        n = a_correct_b_wrong + b_correct_a_wrong
        if n == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'interpretation': 'No difference between models'
            }
        
        statistic = (abs(a_correct_b_wrong - b_correct_a_wrong) - 1) ** 2 / n
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'contingency_table': {
                'both_correct': int(both_correct),
                'both_wrong': int(both_wrong),
                'a_correct_b_wrong': int(a_correct_b_wrong),
                'b_correct_a_wrong': int(b_correct_a_wrong)
            },
            'interpretation': self._interpret_mcnemar(p_value, a_correct_b_wrong, b_correct_a_wrong)
        }
    
    def _interpret_mcnemar(self, p_value: float, 
                          a_correct: int, b_correct: int) -> str:
        """Interpret McNemar's test results."""
        if p_value >= 0.05:
            return "No significant difference between models"
        elif a_correct > b_correct:
            return f"Model A significantly better (p={p_value:.4f})"
        else:
            return f"Model B significantly better (p={p_value:.4f})"
    
    def _bootstrap_comparison(self, y_true: np.ndarray,
                             pred_a: np.ndarray, pred_b: np.ndarray,
                             confidence_level: float,
                             n_iterations: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap confidence intervals for accuracy difference.
        """
        differences = []
        n_samples = len(y_true)
        
        for _ in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            acc_a = np.mean(pred_a[indices] == y_true[indices])
            acc_b = np.mean(pred_b[indices] == y_true[indices])
            
            differences.append(acc_b - acc_a)
        
        differences = np.array(differences)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(differences, alpha/2 * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)
        
        mean_diff = np.mean(differences)
        
        return {
            'mean_difference': float(mean_diff),
            'confidence_interval': {
                'lower': float(lower),
                'upper': float(upper),
                'level': confidence_level
            },
            'significant': not (lower <= 0 <= upper),
            'interpretation': self._interpret_bootstrap(mean_diff, lower, upper)
        }
    
    def _interpret_bootstrap(self, mean_diff: float, 
                            lower: float, upper: float) -> str:
        """Interpret bootstrap results."""
        if lower <= 0 <= upper:
            return "No significant difference (CI includes 0)"
        elif mean_diff > 0:
            return f"Model B significantly better (mean diff: {mean_diff:.4f})"
        else:
            return f"Model A significantly better (mean diff: {mean_diff:.4f})"
    
    def _calculate_effect_size(self, y_true: np.ndarray,
                              pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
        """Calculate effect size (Cohen's h) for the difference."""
        acc_a = np.mean(pred_a == y_true)
        acc_b = np.mean(pred_b == y_true)
        
        # Cohen's h for proportions
        h = 2 * (np.arcsin(np.sqrt(acc_b)) - np.arcsin(np.sqrt(acc_a)))
        
        # Interpret effect size
        if abs(h) < 0.2:
            magnitude = "negligible"
        elif abs(h) < 0.5:
            magnitude = "small"
        elif abs(h) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return {
            'cohens_h': float(h),
            'magnitude': magnitude,
            'interpretation': f"Effect size is {magnitude} (h={h:.4f})"
        }
    
    def _generate_comparison_summary(self) -> None:
        """Generate comparison summary."""
        comparison = {}
        
        for metric in self.results['model_a']['metrics'].keys():
            val_a = self.results['model_a']['metrics'][metric]
            val_b = self.results['model_b']['metrics'][metric]
            
            comparison[metric] = {
                'model_a': val_a,
                'model_b': val_b,
                'difference': val_b - val_a,
                'percent_change': ((val_b - val_a) / val_a * 100) if val_a != 0 else 0,
                'winner': self.model_b_name if val_b > val_a else self.model_a_name
            }
        
        self.results['comparison'] = comparison
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation based on test results.
        
        Returns:
            Recommendation dictionary
        """
        # Check statistical significance
        mcnemar_sig = self.results['statistical_tests']['mcnemar']['significant']
        bootstrap_sig = self.results['statistical_tests']['bootstrap']['significant']
        
        # Count metric wins
        wins_a = 0
        wins_b = 0
        
        for metric, comp in self.results['comparison'].items():
            if comp['winner'] == self.model_a_name:
                wins_a += 1
            else:
                wins_b += 1
        
        # Generate recommendation
        if not (mcnemar_sig or bootstrap_sig):
            recommendation = "No significant difference - keep current model"
            confidence = "low"
        elif wins_b > wins_a:
            recommendation = f"Deploy {self.model_b_name}"
            confidence = "high" if (mcnemar_sig and bootstrap_sig) else "medium"
        else:
            recommendation = f"Keep {self.model_a_name}"
            confidence = "high" if (mcnemar_sig and bootstrap_sig) else "medium"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'wins': {
                self.model_a_name: wins_a,
                self.model_b_name: wins_b
            },
            'statistical_significance': {
                'mcnemar': mcnemar_sig,
                'bootstrap': bootstrap_sig
            }
        }
    
    def generate_report(self, output_path: str = 'ab_test_report.json') -> None:
        """
        Generate detailed test report.
        
        Args:
            output_path: Path for output JSON file
        """
        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        report = {
            'test_info': {
                'model_a': self.model_a_name,
                'model_b': self.model_b_name,
                'timestamp': datetime.now().isoformat()
            },
            'results': convert_to_native(self.results),
            'recommendation': convert_to_native(self.get_recommendation())
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"A/B test report saved: {output_path}")
    
    def visualize_results(self, output_path: str = 'ab_test_visualization.png') -> None:
        """
        Create visualization of test results.
        
        Args:
            output_path: Path for output image
        """
        import matplotlib.pyplot as plt
        
        metrics = list(self.results['comparison'].keys())
        values_a = [self.results['model_a']['metrics'][m] for m in metrics]
        values_b = [self.results['model_b']['metrics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, values_a, width, label=self.model_a_name, alpha=0.8)
        bars2 = ax.bar(x + width/2, values_b, width, label=self.model_b_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('A/B Test Results Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {output_path}")


class MultiModelABTest:
    """
    Extended A/B testing for comparing multiple models simultaneously.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize multi-model A/B test.
        
        Args:
            models: Dictionary of {model_name: model_object}
        """
        self.models = models
        self.results = {}
        logger.info(f"Multi-model A/B test initialized with {len(models)} models")
    
    def run_tournament(self, X_test: np.ndarray, y_test: np.ndarray,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run tournament-style comparison of all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            metrics: List of metrics to evaluate
            
        Returns:
            Tournament results
        """
        logger.info("Running model tournament...")
        
        # Evaluate all models
        for name, model in self.models.items():
            pred = model.predict(X_test)
            
            try:
                proba = model.predict_proba(X_test)
            except:
                proba = None
            
            self.results[name] = {
                'predictions': pred,
                'probabilities': proba,
                'metrics': {}
            }
            
            # Calculate metrics
            default_metrics = ['accuracy', 'precision', 'recall', 'f1']
            metrics = metrics or default_metrics
            
            for metric in metrics:
                self.results[name]['metrics'][metric] = self._calculate_metric(
                    y_test, pred, proba, metric
                )
        
        # Rank models
        rankings = self._rank_models(metrics)
        
        # Pairwise comparisons
        pairwise = self._pairwise_comparisons(y_test)
        
        return {
            'individual_results': {
                name: res['metrics'] for name, res in self.results.items()
            },
            'rankings': rankings,
            'pairwise_comparisons': pairwise,
            'winner': rankings[0]['model']
        }
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray], metric: str) -> float:
        """Calculate metric (same as ABTestingFramework)."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return 0.0
    
    def _rank_models(self, metrics: List[str]) -> List[Dict[str, Any]]:
        """Rank models based on average metric scores."""
        rankings = []
        
        for name, result in self.results.items():
            avg_score = np.mean([result['metrics'][m] for m in metrics])
            rankings.append({
                'model': name,
                'average_score': float(avg_score),
                'metrics': result['metrics']
            })
        
        return sorted(rankings, key=lambda x: x['average_score'], reverse=True)
    
    def _pairwise_comparisons(self, y_true: np.ndarray) -> Dict[str, Dict[str, str]]:
        """Perform pairwise statistical comparisons."""
        comparisons = {}
        model_names = list(self.models.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                pred1 = self.results[name1]['predictions']
                pred2 = self.results[name2]['predictions']
                
                # Simple accuracy comparison
                acc1 = np.mean(pred1 == y_true)
                acc2 = np.mean(pred2 == y_true)
                
                key = f"{name1}_vs_{name2}"
                if acc1 > acc2:
                    comparisons[key] = f"{name1} better"
                elif acc2 > acc1:
                    comparisons[key] = f"{name2} better"
                else:
                    comparisons[key] = "tie"
        
        return comparisons
