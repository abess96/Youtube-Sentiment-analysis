"""
MLflow integration for continuous learning system.
Task 8: MLflow Integration - Continuous learning metrics and retraining tracking
"""

import mlflow
import numpy as np
from typing import Dict, Any
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.mlflow_config import setup_mlflow, get_or_create_experiment

logger = logging.getLogger(__name__)


class ContinuousLearningMLflowTracker:
    """MLflow tracking for continuous learning operations."""
    
    def __init__(self, experiment_name: str = "continuous_learning"):
        setup_mlflow()
        get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        self.active_run = None
        
    def start_tracking(self, run_name: str = None):
        """Start MLflow tracking run."""
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run
    
    def end_tracking(self):
        """End MLflow tracking run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_incremental_update(self, update_count: int, batch_size: int, 
                               accuracy: float = None):
        """Log incremental learning update."""
        mlflow.log_metric("incremental_update_count", update_count)
        mlflow.log_metric("batch_size", batch_size)
        if accuracy:
            mlflow.log_metric("incremental_accuracy", accuracy, step=update_count)
        logger.info(f"Logged incremental update {update_count}")
    
    def log_active_learning_query(self, iteration: int, strategy: str, 
                                   n_samples: int, uncertainty_scores: np.ndarray = None):
        """Log active learning query."""
        mlflow.log_param(f"al_strategy_{iteration}", strategy)
        mlflow.log_metric("al_samples_queried", n_samples, step=iteration)
        
        if uncertainty_scores is not None:
            mlflow.log_metric("al_avg_uncertainty", float(uncertainty_scores.mean()), step=iteration)
            mlflow.log_metric("al_max_uncertainty", float(uncertainty_scores.max()), step=iteration)
        
        logger.info(f"Logged active learning query {iteration}")
    
    def log_drift_detection(self, drift_report: Dict[str, Any], step: int = 0):
        """Log drift detection results."""
        mlflow.log_metric("drift_ks_score", drift_report['data_drift_ks']['drift_score'], step=step)
        mlflow.log_metric("drift_psi", drift_report['data_drift_psi']['average_psi'], step=step)
        mlflow.log_param(f"drift_status_{step}", drift_report['overall_status'])
        
        if 'performance_drift' in drift_report:
            mlflow.log_metric("performance_drop", 
                            drift_report['performance_drift']['relative_drop'], step=step)
        
        logger.info(f"Logged drift detection at step {step}")
    
    def log_retraining(self, retrain_metrics: Dict[str, Any], reason: str):
        """Log model retraining event."""
        mlflow.log_param("retrain_reason", reason)
        mlflow.log_metric("retrain_train_accuracy", retrain_metrics['train_accuracy'])
        mlflow.log_metric("retrain_val_accuracy", retrain_metrics['val_accuracy'])
        mlflow.log_metric("retrain_val_f1", retrain_metrics['val_f1'])
        mlflow.log_metric("retrain_n_samples", retrain_metrics['n_train_samples'])
        
        logger.info(f"Logged retraining: {reason}")
    
    def log_model_checkpoint(self, model, checkpoint_name: str):
        """Log model checkpoint as artifact."""
        mlflow.sklearn.log_model(model, checkpoint_name)
        logger.info(f"Logged model checkpoint: {checkpoint_name}")
    
    def log_continuous_learning_summary(self, summary: Dict[str, Any]):
        """Log overall continuous learning summary."""
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"summary_{key}", value)
            else:
                mlflow.log_param(f"summary_{key}", str(value))
        
        logger.info("Logged continuous learning summary")
