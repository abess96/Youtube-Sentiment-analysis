"""
Model and data drift detection and monitoring system.
Task 4.3: Build model and data drift monitoring system
"""

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect statistical drift in data and model performance."""
    
    def __init__(self, reference_data=None, reference_predictions=None, 
                 threshold: float = 0.05):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.threshold = threshold
        self.drift_history = []
    
    def detect_data_drift_ks(self, reference_data, current_data) -> Dict[str, Any]:
        """Detect data drift using Kolmogorov-Smirnov test."""
        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError("Reference and current data must have same number of features")
        
        drift_detected = False
        feature_drifts = []
        
        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            feature_drift = {
                'feature_idx': i,
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': bool(p_value < self.threshold)
            }
            feature_drifts.append(feature_drift)
            
            if p_value < self.threshold:
                drift_detected = True
        
        # Calculate overall drift score
        drift_score = np.mean([fd['ks_statistic'] for fd in feature_drifts])
        
        return {
            'overall_drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'n_drifted_features': sum(fd['drift_detected'] for fd in feature_drifts),
            'total_features': len(feature_drifts),
            'feature_drifts': feature_drifts[:10]  # Top 10 for brevity
        }
    
    def detect_data_drift_psi(self, reference_data, current_data, 
                              n_bins: int = 10) -> Dict[str, Any]:
        """Detect data drift using Population Stability Index (PSI)."""
        psi_values = []
        
        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Create bins based on reference data
            bins = np.histogram_bin_edges(ref_feature, bins=n_bins)
            
            # Calculate distributions
            ref_dist, _ = np.histogram(ref_feature, bins=bins)
            curr_dist, _ = np.histogram(curr_feature, bins=bins)
            
            # Normalize
            ref_dist = ref_dist / len(ref_feature) + 1e-10
            curr_dist = curr_dist / len(curr_feature) + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
            psi_values.append(float(psi))
        
        avg_psi = np.mean(psi_values)
        
        # PSI interpretation: <0.1: no drift, 0.1-0.2: moderate, >0.2: significant
        drift_level = 'none' if avg_psi < 0.1 else ('moderate' if avg_psi < 0.2 else 'significant')
        
        return {
            'average_psi': float(avg_psi),
            'drift_level': drift_level,
            'drift_detected': bool(avg_psi > 0.1),
            'max_psi': float(np.max(psi_values)),
            'min_psi': float(np.min(psi_values))
        }
    
    def detect_prediction_drift(self, reference_predictions, current_predictions) -> Dict[str, Any]:
        """Detect drift in prediction distributions."""
        # Handle negative labels by shifting to non-negative
        ref_preds = np.array(reference_predictions)
        curr_preds = np.array(current_predictions)
        
        # Shift to non-negative if needed
        min_label = min(ref_preds.min(), curr_preds.min())
        if min_label < 0:
            ref_preds = ref_preds - min_label
            curr_preds = curr_preds - min_label
        
        # Chi-square test for categorical predictions
        ref_counts = np.bincount(ref_preds.astype(int))
        curr_counts = np.bincount(curr_preds.astype(int), minlength=len(ref_counts))
        
        # Ensure same length
        max_len = max(len(ref_counts), len(curr_counts))
        ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
        curr_counts = np.pad(curr_counts, (0, max_len - len(curr_counts)))
        
        # Avoid division by zero
        ref_counts = ref_counts + 1e-10
        curr_counts = curr_counts + 1e-10
        
        chi2_stat, p_value = stats.chisquare(curr_counts, ref_counts)
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'drift_detected': bool(p_value < self.threshold),
            'reference_distribution': (ref_counts / ref_counts.sum()).tolist(),
            'current_distribution': (curr_counts / curr_counts.sum()).tolist()
        }
    
    def detect_performance_drift(self, y_true_ref, y_pred_ref, 
                                 y_true_curr, y_pred_curr) -> Dict[str, Any]:
        """Detect drift in model performance."""
        ref_accuracy = accuracy_score(y_true_ref, y_pred_ref)
        curr_accuracy = accuracy_score(y_true_curr, y_pred_curr)
        
        accuracy_drop = ref_accuracy - curr_accuracy
        relative_drop = accuracy_drop / ref_accuracy if ref_accuracy > 0 else 0
        
        # Alert if performance drops by more than 5%
        performance_alert = bool(relative_drop > 0.05)
        
        return {
            'reference_accuracy': float(ref_accuracy),
            'current_accuracy': float(curr_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'relative_drop': float(relative_drop),
            'performance_alert': performance_alert,
            'severity': 'high' if relative_drop > 0.1 else ('medium' if relative_drop > 0.05 else 'low')
        }
    
    def monitor(self, current_data, current_predictions=None, 
                y_true_curr=None) -> Dict[str, Any]:
        """Comprehensive drift monitoring."""
        if self.reference_data is None:
            raise ValueError("Reference data not set. Initialize with reference data.")
        
        timestamp = datetime.now().isoformat()
        monitoring_report = {
            'timestamp': timestamp,
            'data_drift_ks': self.detect_data_drift_ks(self.reference_data, current_data),
            'data_drift_psi': self.detect_data_drift_psi(self.reference_data, current_data)
        }
        
        # Prediction drift
        if current_predictions is not None and self.reference_predictions is not None:
            monitoring_report['prediction_drift'] = self.detect_prediction_drift(
                self.reference_predictions, current_predictions
            )
        
        # Performance drift
        if y_true_curr is not None and current_predictions is not None:
            if hasattr(self, 'y_true_ref') and hasattr(self, 'y_pred_ref'):
                monitoring_report['performance_drift'] = self.detect_performance_drift(
                    self.y_true_ref, self.y_pred_ref, y_true_curr, current_predictions
                )
        
        # Overall drift status
        monitoring_report['overall_status'] = self._determine_overall_status(monitoring_report)
        
        # Store in history
        self.drift_history.append(monitoring_report)
        
        return monitoring_report
    
    def _determine_overall_status(self, report: Dict[str, Any]) -> str:
        """Determine overall drift status."""
        alerts = []
        
        if report['data_drift_ks']['overall_drift_detected']:
            alerts.append('data_drift')
        
        if report['data_drift_psi']['drift_level'] in ['moderate', 'significant']:
            alerts.append('psi_drift')
        
        if 'prediction_drift' in report and report['prediction_drift']['drift_detected']:
            alerts.append('prediction_drift')
        
        if 'performance_drift' in report and report['performance_drift']['performance_alert']:
            alerts.append('performance_degradation')
        
        if not alerts:
            return 'healthy'
        elif len(alerts) == 1:
            return 'warning'
        else:
            return 'critical'
    
    def generate_alert(self, monitoring_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alert based on monitoring report."""
        status = monitoring_report['overall_status']
        
        alert = {
            'timestamp': monitoring_report['timestamp'],
            'status': status,
            'requires_action': status in ['warning', 'critical'],
            'messages': []
        }
        
        if monitoring_report['data_drift_ks']['overall_drift_detected']:
            n_drifted = monitoring_report['data_drift_ks']['n_drifted_features']
            alert['messages'].append(f"Data drift detected in {n_drifted} features")
        
        if monitoring_report['data_drift_psi']['drift_level'] != 'none':
            level = monitoring_report['data_drift_psi']['drift_level']
            alert['messages'].append(f"PSI indicates {level} drift")
        
        if 'performance_drift' in monitoring_report:
            perf = monitoring_report['performance_drift']
            if perf['performance_alert']:
                alert['messages'].append(
                    f"Performance degradation: {perf['relative_drop']*100:.1f}% drop"
                )
        
        if not alert['messages']:
            alert['messages'].append("No drift detected. System healthy.")
        
        return alert
    
    def save_monitoring_history(self, output_path: str):
        """Save drift monitoring history."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        logger.info(f"Monitoring history saved to {output_path}")
