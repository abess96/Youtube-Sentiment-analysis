"""
Enhanced MLflow experiment tracking with advanced features.
Provides comprehensive logging, artifact management, and experiment comparison.
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.mlflow_config import setup_mlflow, get_or_create_experiment

logger = logging.getLogger(__name__)


class EnhancedMLflowTracker:
    """
    Enhanced MLflow tracker with advanced experiment tracking capabilities.
    Supports comprehensive logging, automated artifact management, and model registration.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize enhanced MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            setup_mlflow()
        
        self.experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        logger.info(f"Enhanced MLflow tracker initialized: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None,
                  nested: bool = False) -> mlflow.ActiveRun:
        """Start MLflow run with automatic metadata."""
        default_tags = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name
        }
        if tags:
            default_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=default_tags, nested=nested)
    
    def log_comprehensive_metrics(self, metrics: Dict[str, Any], 
                                  step: Optional[int] = None,
                                  prefix: str = '') -> None:
        """
        Log metrics with automatic type handling and nested structure support.
        
        Args:
            metrics: Dictionary of metrics (supports nested dicts)
            step: Optional step number
            prefix: Prefix for metric names
        """
        flat_metrics = self._flatten_dict(metrics, prefix)
        
        for key, value in flat_metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(key, float(value), step=step)
            elif isinstance(value, (list, np.ndarray)):
                # Log array statistics
                arr = np.array(value)
                mlflow.log_metric(f"{key}_mean", float(np.mean(arr)), step=step)
                mlflow.log_metric(f"{key}_std", float(np.std(arr)), step=step)
                mlflow.log_metric(f"{key}_min", float(np.min(arr)), step=step)
                mlflow.log_metric(f"{key}_max", float(np.max(arr)), step=step)
    
    def log_params_comprehensive(self, params: Dict[str, Any]) -> None:
        """Log parameters with automatic flattening and truncation."""
        flat_params = self._flatten_dict(params)
        
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 250:
                str_value = str_value[:247] + "..."
            mlflow.log_param(key, str_value)
    
    def log_model_with_signature(self, model: Any, artifact_path: str,
                                 input_example: Optional[np.ndarray] = None,
                                 model_type: str = 'sklearn') -> None:
        """
        Log model with automatic signature inference.
        
        Args:
            model: Model to log
            artifact_path: Path for model artifact
            input_example: Example input for signature inference
            model_type: Type of model ('sklearn', 'pytorch', 'custom')
        """
        try:
            if model_type == 'sklearn':
                if input_example is not None:
                    signature = mlflow.models.infer_signature(
                        input_example, 
                        model.predict(input_example)
                    )
                    mlflow.sklearn.log_model(model, artifact_path, signature=signature)
                else:
                    mlflow.sklearn.log_model(model, artifact_path)
            elif model_type == 'pytorch':
                mlflow.pytorch.log_model(model, artifact_path)
            else:
                # Custom model - save as pickle
                import pickle
                temp_path = f"temp_{artifact_path}.pkl"
                with open(temp_path, 'wb') as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(temp_path, artifact_path)
                Path(temp_path).unlink(missing_ok=True)
            
            logger.info(f"Model logged: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def register_model(self, model_uri: str, model_name: str,
                      tags: Optional[Dict[str, str]] = None,
                      description: Optional[str] = None) -> Any:
        """
        Register model to MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for registered model
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            ModelVersion object
        """
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Add tags and description
            client = mlflow.tracking.MlflowClient()
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(
                        model_name, 
                        model_version.version, 
                        key, 
                        value
                    )
            
            if description:
                client.update_model_version(
                    model_name,
                    model_version.version,
                    description=description
                )
            
            logger.info(f"Model registered: {model_name} v{model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def log_dataset_info(self, X: np.ndarray, y: np.ndarray, 
                        dataset_name: str = 'dataset') -> None:
        """Log comprehensive dataset information."""
        info = {
            f'{dataset_name}_samples': len(X),
            f'{dataset_name}_features': X.shape[1] if len(X.shape) > 1 else 1,
            f'{dataset_name}_classes': len(np.unique(y)),
        }
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            info[f'{dataset_name}_class_{cls}_count'] = int(count)
            info[f'{dataset_name}_class_{cls}_ratio'] = float(count / len(y))
        
        self.log_comprehensive_metrics(info)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           labels: Optional[List[str]] = None) -> None:
        """Log confusion matrix as artifact."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate default labels if not provided
        if labels is None:
            unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            labels = [str(label) for label in unique_labels]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
    
    def log_feature_importance(self, feature_names: List[str], 
                              importance_values: np.ndarray,
                              top_n: int = 20) -> None:
        """Log feature importance visualization."""
        import matplotlib.pyplot as plt
        
        # Sort by importance
        indices = np.argsort(importance_values)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance_values[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)
        
        # Log as JSON
        importance_dict = {
            feature_names[i]: float(importance_values[i]) 
            for i in indices
        }
        mlflow.log_dict(importance_dict, "feature_importance.json")
    
    def log_training_history(self, history: Dict[str, List[float]]) -> None:
        """Log training history with visualization."""
        import matplotlib.pyplot as plt
        
        # Log metrics over epochs
        for metric_name, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=epoch)
        
        # Create visualization
        fig, axes = plt.subplots(len(history), 1, figsize=(10, 4*len(history)))
        if len(history) == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, history.items()):
            ax.plot(values)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over epochs')
            ax.grid(True)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "training_history.png")
        plt.close(fig)
    
    def compare_runs(self, run_ids: List[str], 
                    metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of specific metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        client = mlflow.tracking.MlflowClient()
        
        comparison_data = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            
            row = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'status': run.info.status
            }
            
            # Add metrics
            if metrics:
                for metric in metrics:
                    row[metric] = run.data.metrics.get(metric, None)
            else:
                row.update(run.data.metrics)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Log comparison
        mlflow.log_dict(df.to_dict(), "run_comparison.json")
        
        return df
    
    def get_best_run(self, metric: str, mode: str = 'max') -> Tuple[str, float]:
        """
        Get best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (run_id, metric_value)
        """
        client = mlflow.tracking.MlflowClient()
        
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            return best_run.info.run_id, best_run.data.metrics.get(metric)
        
        return None, None
    
    def log_system_metrics(self) -> None:
        """Log system resource usage."""
        import psutil
        
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        self.log_comprehensive_metrics(metrics, prefix='system')
    
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


class ExperimentComparator:
    """
    Tool for comparing and analyzing multiple experiments.
    """
    
    def __init__(self, experiment_names: List[str]):
        """
        Initialize experiment comparator.
        
        Args:
            experiment_names: List of experiment names to compare
        """
        setup_mlflow()
        self.experiment_names = experiment_names
        self.client = mlflow.tracking.MlflowClient()
    
    def compare_experiments(self, metric: str, top_n: int = 10) -> pd.DataFrame:
        """
        Compare experiments based on a metric.
        
        Args:
            metric: Metric to compare
            top_n: Number of top runs to include
            
        Returns:
            DataFrame with comparison results
        """
        all_runs = []
        
        for exp_name in self.experiment_names:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment:
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=[f"metrics.{metric} DESC"],
                    max_results=top_n
                )
                
                for run in runs:
                    all_runs.append({
                        'experiment': exp_name,
                        'run_id': run.info.run_id,
                        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                        metric: run.data.metrics.get(metric),
                        'start_time': datetime.fromtimestamp(run.info.start_time / 1000)
                    })
        
        return pd.DataFrame(all_runs).sort_values(metric, ascending=False)
    
    def generate_comparison_report(self, metrics: List[str], 
                                  output_path: str = 'experiment_comparison.html') -> None:
        """
        Generate HTML comparison report.
        
        Args:
            metrics: List of metrics to include
            output_path: Path for output HTML file
        """
        import matplotlib.pyplot as plt
        
        report_html = "<html><head><title>Experiment Comparison</title></head><body>"
        report_html += "<h1>Experiment Comparison Report</h1>"
        
        for metric in metrics:
            df = self.compare_experiments(metric, top_n=10)
            
            report_html += f"<h2>Top Runs by {metric}</h2>"
            report_html += df.to_html(index=False)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            for exp_name in self.experiment_names:
                exp_data = df[df['experiment'] == exp_name]
                ax.scatter(exp_data['start_time'], exp_data[metric], 
                          label=exp_name, s=100, alpha=0.6)
            
            ax.set_xlabel('Time')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over Time')
            ax.legend()
            ax.grid(True)
            
            img_path = f"comparison_{metric}.png"
            plt.savefig(img_path)
            plt.close(fig)
            
            report_html += f'<img src="{img_path}" />'
        
        report_html += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Comparison report generated: {output_path}")
