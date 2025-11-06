"""
MLflow integration for experiment tracking and model management.
Provides utilities for logging models, parameters, metrics, and artifacts.
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base.base_model import BaseModel, ModelConfig
from src.config.mlflow_config import setup_mlflow, get_or_create_experiment

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.
    Handles logging of parameters, metrics, models, and artifacts.
    Uses existing mlflow_config for DagsHub integration.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional MLflow tracking server URI (overrides config)
        """
        self.experiment_name = experiment_name
        
        # Setup MLflow using existing configuration
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Using provided tracking URI: {tracking_uri}")
        else:
            # Use existing mlflow_config setup
            setup_mlflow()
        
        # Get or create experiment
        self.experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        logger.info(f"MLflow tracker initialized for experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Active MLflow run context
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)
            
            # MLflow has a limit on param value length
            for key, value in flat_params.items():
                str_value = str(value)
                if len(str_value) > 250:
                    str_value = str_value[:247] + "..."
                mlflow.log_param(key, str_value)
                
            logger.debug(f"Logged {len(flat_params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(key, float(value), step=step)
            
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(self, model: BaseModel, artifact_path: str = "model",
                  signature: Optional[Any] = None) -> None:
        """
        Log a model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within the run's artifact directory
            signature: Optional model signature
        """
        try:
            # Save model temporarily
            temp_path = f"temp_model_{artifact_path}.pkl"
            model.save(temp_path)
            
            # Log as artifact
            mlflow.log_artifact(temp_path, artifact_path)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            logger.info(f"Model logged to MLflow at {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_sklearn_model(self, model: Any, artifact_path: str = "model") -> None:
        """
        Log a scikit-learn model to MLflow.
        
        Args:
            model: Scikit-learn model
            artifact_path: Path within the run's artifact directory
        """
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Sklearn model logged to MLflow")
        except Exception as e:
            logger.error(f"Error logging sklearn model: {e}")
    
    def log_pytorch_model(self, model: Any, artifact_path: str = "model") -> None:
        """
        Log a PyTorch model to MLflow.
        
        Args:
            model: PyTorch model
            artifact_path: Path within the run's artifact directory
        """
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            logger.info(f"PyTorch model logged to MLflow")
        except Exception as e:
            logger.error(f"Error logging PyTorch model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file to MLflow.
        
        Args:
            local_path: Local file path
            artifact_path: Optional path within artifact directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Artifact logged: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str) -> None:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.debug(f"Dictionary logged as {filename}")
        except Exception as e:
            logger.error(f"Error logging dictionary: {e}")
    
    def log_figure(self, figure: Any, filename: str) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            filename: Name for the figure file
        """
        try:
            mlflow.log_figure(figure, filename)
            logger.debug(f"Figure logged as {filename}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Error setting tags: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', 
                     sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ModelTrainingLogger:
    """
    Specialized logger for model training with MLflow.
    Automatically logs training progress, metrics, and model artifacts.
    """
    
    def __init__(self, tracker: MLflowTracker):
        """
        Initialize training logger.
        
        Args:
            tracker: MLflowTracker instance
        """
        self.tracker = tracker
        self.current_run = None
    
    def log_training_run(self, model: BaseModel, X_train: np.ndarray, 
                        y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None,
                        run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a complete training run.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            run_name: Optional run name
            
        Returns:
            Dictionary with training results
        """
        with self.tracker.start_run(run_name=run_name) as run:
            self.current_run = run
            
            # Log model configuration
            self.tracker.log_params(model.config.hyperparameters)
            self.tracker.set_tags({
                'model_type': model.config.model_type,
                'model_name': model.config.model_name,
                'model_version': model.config.version
            })
            
            # Train model
            logger.info(f"Training {model.config.model_name}...")
            train_result = model.train(X_train, y_train)
            
            # Log training metrics
            if isinstance(train_result, dict):
                metrics = {k: v for k, v in train_result.items() 
                          if isinstance(v, (int, float, np.number))}
                self.tracker.log_metrics(metrics)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate_model(model, X_val, y_val, prefix='val_')
                self.tracker.log_metrics(val_metrics)
            
            # Log model
            self.tracker.log_model(model)
            
            # Log training info
            self.tracker.log_dict(train_result, "training_result.json")
            
            logger.info(f"Training run logged to MLflow: {run.info.run_id}")
            
            return {
                'run_id': run.info.run_id,
                'train_result': train_result
            }
    
    def log_ensemble_training(self, ensemble: BaseModel, base_models: List[BaseModel],
                            X_train: np.ndarray, y_train: np.ndarray,
                            run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Log ensemble model training.
        
        Args:
            ensemble: Ensemble model
            base_models: List of base models
            X_train: Training features
            y_train: Training labels
            run_name: Optional run name
            
        Returns:
            Dictionary with training results
        """
        with self.tracker.start_run(run_name=run_name) as run:
            # Log ensemble configuration
            self.tracker.set_tags({
                'model_type': 'ensemble',
                'ensemble_type': ensemble.config.model_type,
                'num_base_models': len(base_models)
            })
            
            # Log base model info
            base_model_info = [
                {'type': m.config.model_type, 'name': m.config.model_name}
                for m in base_models
            ]
            self.tracker.log_dict({'base_models': base_model_info}, 
                                 "base_models.json")
            
            # Train ensemble
            train_result = ensemble.train(X_train, y_train)
            
            # Log metrics
            if isinstance(train_result, dict):
                metrics = {k: v for k, v in train_result.items() 
                          if isinstance(v, (int, float, np.number))}
                self.tracker.log_metrics(metrics)
            
            # Log ensemble model
            self.tracker.log_model(ensemble, artifact_path="ensemble_model")
            
            logger.info(f"Ensemble training logged: {run.info.run_id}")
            
            return {
                'run_id': run.info.run_id,
                'train_result': train_result
            }
    
    def log_hyperparameter_optimization(self, optimizer: Any, 
                                       optimization_result: Dict[str, Any],
                                       run_name: Optional[str] = None) -> str:
        """
        Log hyperparameter optimization results.
        
        Args:
            optimizer: Hyperparameter optimizer
            optimization_result: Results from optimization
            run_name: Optional run name
            
        Returns:
            Run ID
        """
        with self.tracker.start_run(run_name=run_name) as run:
            # Log optimization configuration
            self.tracker.set_tags({
                'task': 'hyperparameter_optimization',
                'model_type': optimizer.model_type,
                'optimization_method': optimizer.__class__.__name__
            })
            
            # Log best parameters
            self.tracker.log_params(optimization_result['best_params'])
            
            # Log optimization metrics
            self.tracker.log_metrics({
                'best_score': optimization_result['best_score'],
                'n_iterations': optimization_result.get('n_iterations', 0),
                'elapsed_time': optimization_result.get('elapsed_time', 0)
            })
            
            # Log optimization history
            if 'optimization_history' in optimization_result:
                self.tracker.log_dict(
                    {'history': optimization_result['optimization_history']},
                    "optimization_history.json"
                )
            
            logger.info(f"Hyperparameter optimization logged: {run.info.run_id}")
            
            return run.info.run_id
    
    def _evaluate_model(self, model: BaseModel, X: np.ndarray, 
                       y: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """Evaluate model and return metrics."""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average='weighted'
        )
        
        return {
            f'{prefix}accuracy': accuracy,
            f'{prefix}precision': precision,
            f'{prefix}recall': recall,
            f'{prefix}f1': f1
        }


def create_mlflow_tracker(experiment_name: str = "sentiment_analysis",
                         tracking_uri: Optional[str] = None) -> MLflowTracker:
    """
    Convenience function to create an MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional tracking server URI
        
    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(experiment_name, tracking_uri)
