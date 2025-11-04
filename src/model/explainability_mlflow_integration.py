"""
MLflow Integration for Explainability Features.
Logs SHAP values, explanations, and uncertainty metrics to MLflow.
"""

import numpy as np
import mlflow
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ExplainabilityMLflowLogger:
    """Logger for explainability metrics and artifacts in MLflow."""
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize MLflow logger for explainability.
        
        Args:
            run_id: Existing MLflow run ID (optional)
        """
        self.run_id = run_id
    
    def log_explanation_metrics(self, explanation_report: Dict[str, Any]) -> None:
        """
        Log explanation metrics to MLflow.
        
        Args:
            explanation_report: Report from ExplainabilityEngine
        """
        with mlflow.start_run(run_id=self.run_id):
            # Log number of explanations
            mlflow.log_metric("num_explanations", explanation_report['num_explanations'])
            
            # Log top features as parameters
            top_features = explanation_report['top_features'][:5]
            for i, feature in enumerate(top_features):
                mlflow.log_param(f"top_feature_{i+1}", feature)
            
            # Log global feature importance
            for feature, importance in list(explanation_report['global_feature_importance'].items())[:10]:
                mlflow.log_metric(f"importance_{feature}", importance)
            
            logger.info("Logged explanation metrics to MLflow")
    
    def log_uncertainty_metrics(self, uncertainty_stats: Dict[str, Any]) -> None:
        """
        Log uncertainty metrics to MLflow.
        
        Args:
            uncertainty_stats: Statistics from UncertaintyQuantifier
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metric("mean_uncertainty", uncertainty_stats['mean_uncertainty'])
            mlflow.log_metric("mean_confidence", uncertainty_stats['mean_confidence'])
            mlflow.log_metric("reliability_rate", uncertainty_stats['reliability_rate'])
            mlflow.log_metric("reliable_predictions", uncertainty_stats['reliable_predictions'])
            
            logger.info("Logged uncertainty metrics to MLflow")
    
    def log_debug_report(self, debug_report: Dict[str, Any]) -> None:
        """
        Log debugging report to MLflow.
        
        Args:
            debug_report: Report from ModelDebugger
        """
        with mlflow.start_run(run_id=self.run_id):
            # Log overall metrics
            for metric, value in debug_report['overall_metrics'].items():
                mlflow.log_metric(f"debug_{metric}", value)
            
            # Log misclassification rate
            mlflow.log_metric("misclassification_rate", 
                            debug_report['misclassification_analysis']['misclassification_rate'])
            
            # Log class-specific errors
            for class_name, errors in debug_report['misclassification_analysis']['class_specific_errors'].items():
                mlflow.log_metric(f"error_rate_{class_name}", errors['error_rate'])
            
            logger.info("Logged debug report to MLflow")
    
    def log_visualization(self, viz_path: str, artifact_name: str) -> None:
        """
        Log visualization to MLflow.
        
        Args:
            viz_path: Path to visualization file
            artifact_name: Name for the artifact
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(viz_path, artifact_path=f"explainability/{artifact_name}")
            logger.info(f"Logged visualization {artifact_name} to MLflow")
