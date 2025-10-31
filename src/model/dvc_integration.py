"""
DVC integration for data and model versioning.
Provides utilities for managing DVC pipelines, tracking data, and versioning models.
"""

import subprocess
import yaml
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DVCManager:
    """
    DVC pipeline and versioning manager.
    Handles DVC operations for data versioning and pipeline management.
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize DVC manager.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.dvc_file = self.repo_path / "dvc.yaml"
        self.params_file = self.repo_path / "params.yaml"
        
        logger.info(f"DVC manager initialized for repo: {repo_path}")
    
    def run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """
        Run a DVC command.
        
        Args:
            command: Command to run as list of strings
            
        Returns:
            CompletedProcess result
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"DVC command successful: {' '.join(command)}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e.stderr}")
            raise
    
    def add_data(self, data_path: str) -> None:
        """
        Add data file or directory to DVC tracking.
        
        Args:
            data_path: Path to data file or directory
        """
        logger.info(f"Adding {data_path} to DVC tracking...")
        self.run_command(["dvc", "add", data_path])
        logger.info(f"Successfully added {data_path} to DVC")
    
    def push(self, remote: Optional[str] = None) -> None:
        """
        Push data to DVC remote storage.
        
        Args:
            remote: Optional remote name
        """
        cmd = ["dvc", "push"]
        if remote:
            cmd.extend(["-r", remote])
        
        logger.info("Pushing data to DVC remote...")
        self.run_command(cmd)
        logger.info("Successfully pushed data to DVC remote")
    
    def pull(self, remote: Optional[str] = None) -> None:
        """
        Pull data from DVC remote storage.
        
        Args:
            remote: Optional remote name
        """
        cmd = ["dvc", "pull"]
        if remote:
            cmd.extend(["-r", remote])
        
        logger.info("Pulling data from DVC remote...")
        self.run_command(cmd)
        logger.info("Successfully pulled data from DVC remote")
    
    def repro(self, pipeline: Optional[str] = None, force: bool = False) -> None:
        """
        Reproduce DVC pipeline.
        
        Args:
            pipeline: Optional specific pipeline stage
            force: Force reproduction even if outputs exist
        """
        cmd = ["dvc", "repro"]
        if pipeline:
            cmd.append(pipeline)
        if force:
            cmd.append("--force")
        
        logger.info("Reproducing DVC pipeline...")
        self.run_command(cmd)
        logger.info("Pipeline reproduction complete")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Load parameters from params.yaml.
        
        Returns:
            Dictionary of parameters
        """
        if not self.params_file.exists():
            logger.warning(f"params.yaml not found at {self.params_file}")
            return {}
        
        with open(self.params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        return params or {}
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update parameters in params.yaml.
        
        Args:
            params: Dictionary of parameters to update
        """
        current_params = self.get_params()
        current_params.update(params)
        
        with open(self.params_file, 'w') as f:
            yaml.dump(current_params, f, default_flow_style=False)
        
        logger.info(f"Updated params.yaml with {len(params)} parameters")
    
    def get_metrics(self, metrics_file: str = "metrics.json") -> Dict[str, Any]:
        """
        Load metrics from a metrics file.
        
        Args:
            metrics_file: Path to metrics file
            
        Returns:
            Dictionary of metrics
        """
        metrics_path = self.repo_path / metrics_file
        
        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            return {}
        
        with open(metrics_path, 'r') as f:
            if metrics_file.endswith('.json'):
                metrics = json.load(f)
            elif metrics_file.endswith('.yaml') or metrics_file.endswith('.yml'):
                metrics = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported metrics file format: {metrics_file}")
                return {}
        
        return metrics or {}
    
    def save_metrics(self, metrics: Dict[str, Any], 
                    metrics_file: str = "metrics.json") -> None:
        """
        Save metrics to a file.
        
        Args:
            metrics: Dictionary of metrics
            metrics_file: Path to save metrics
        """
        metrics_path = self.repo_path / metrics_file
        
        with open(metrics_path, 'w') as f:
            if metrics_file.endswith('.json'):
                json.dump(metrics, f, indent=2)
            elif metrics_file.endswith('.yaml') or metrics_file.endswith('.yml'):
                yaml.dump(metrics, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported metrics file format: {metrics_file}")
                return
        
        logger.info(f"Saved metrics to {metrics_file}")


class DVCPipelineBuilder:
    """
    Builder for creating DVC pipeline stages.
    Helps construct dvc.yaml pipeline definitions.
    """
    
    def __init__(self, dvc_file: str = "dvc.yaml"):
        """
        Initialize pipeline builder.
        
        Args:
            dvc_file: Path to dvc.yaml file
        """
        self.dvc_file = Path(dvc_file)
        self.stages = {}
        
        # Load existing pipeline if it exists
        if self.dvc_file.exists():
            with open(self.dvc_file, 'r') as f:
                existing = yaml.safe_load(f)
                if existing and 'stages' in existing:
                    self.stages = existing['stages']
    
    def add_stage(self, name: str, cmd: str, deps: List[str] = None,
                 outs: List[str] = None, metrics: List[str] = None,
                 params: List[str] = None, plots: List[str] = None) -> 'DVCPipelineBuilder':
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            cmd: Command to execute
            deps: List of dependencies
            outs: List of outputs
            metrics: List of metrics files
            params: List of parameter references
            plots: List of plot files
            
        Returns:
            Self for chaining
        """
        stage = {'cmd': cmd}
        
        if deps:
            stage['deps'] = deps
        if outs:
            stage['outs'] = outs
        if metrics:
            stage['metrics'] = [{'path': m, 'cache': False} for m in metrics]
        if params:
            stage['params'] = params
        if plots:
            stage['plots'] = plots
        
        self.stages[name] = stage
        logger.info(f"Added stage '{name}' to pipeline")
        
        return self
    
    def add_model_training_stage(self, name: str, script: str,
                                 data_deps: List[str], model_output: str,
                                 metrics_output: str, params: List[str]) -> 'DVCPipelineBuilder':
        """
        Add a model training stage with common settings.
        
        Args:
            name: Stage name
            script: Training script path
            data_deps: Data dependencies
            model_output: Model output path
            metrics_output: Metrics output path
            params: Parameter references
            
        Returns:
            Self for chaining
        """
        return self.add_stage(
            name=name,
            cmd=f"python {script}",
            deps=data_deps + [script],
            outs=[model_output],
            metrics=[metrics_output],
            params=params
        )
    
    def save(self) -> None:
        """Save the pipeline to dvc.yaml."""
        pipeline = {'stages': self.stages}
        
        with open(self.dvc_file, 'w') as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved pipeline to {self.dvc_file}")
    
    def get_pipeline(self) -> Dict[str, Any]:
        """
        Get the current pipeline definition.
        
        Returns:
            Pipeline dictionary
        """
        return {'stages': self.stages}


class ModelVersionManager:
    """
    Manager for versioning models with DVC.
    Tracks model artifacts and their versions.
    """
    
    def __init__(self, models_dir: str = "models/trained_models"):
        """
        Initialize model version manager.
        
        Args:
            models_dir: Directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dvc_manager = DVCManager()
        
        logger.info(f"Model version manager initialized: {models_dir}")
    
    def save_and_version_model(self, model: Any, model_name: str,
                              version: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model and add it to DVC tracking.
        
        Args:
            model: Model to save
            model_name: Name of the model
            version: Version string
            metadata: Optional metadata to save
            
        Returns:
            Path to saved model
        """
        # Create version directory
        version_dir = self.models_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        model.save(str(model_path))
        
        # Save metadata
        if metadata:
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Add to DVC tracking
        self.dvc_manager.add_data(str(version_dir))
        
        logger.info(f"Model saved and versioned: {model_path}")
        
        return str(model_path)
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            return []
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        return sorted(versions)
    
    def get_model_path(self, model_name: str, version: str) -> Optional[str]:
        """
        Get the path to a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Path to model or None if not found
        """
        model_path = self.models_dir / model_name / version / "model.pkl"
        
        if model_path.exists():
            return str(model_path)
        
        return None


def create_model_training_pipeline(output_file: str = "dvc.yaml") -> DVCPipelineBuilder:
    """
    Create a standard model training pipeline.
    
    Args:
        output_file: Path to save pipeline
        
    Returns:
        DVCPipelineBuilder instance
    """
    builder = DVCPipelineBuilder(output_file)
    
    # Add preprocessing stage
    builder.add_stage(
        name="preprocess",
        cmd="python src/data/preprocess.py",
        deps=["data/raw", "src/data/preprocess.py"],
        outs=["data/processed"],
        params=["preprocess"]
    )
    
    # Add feature engineering stage
    builder.add_stage(
        name="feature_engineering",
        cmd="python src/features/build_features.py",
        deps=["data/processed", "src/features/build_features.py"],
        outs=["data/features"],
        params=["features"]
    )
    
    # Add model training stage
    builder.add_model_training_stage(
        name="train_model",
        script="src/model/train.py",
        data_deps=["data/features"],
        model_output="models/trained_models/model.pkl",
        metrics_output="metrics.json",
        params=["model"]
    )
    
    # Add evaluation stage
    builder.add_stage(
        name="evaluate",
        cmd="python src/model/evaluate.py",
        deps=["models/trained_models/model.pkl", "data/features"],
        metrics=["evaluation_metrics.json"],
        params=["evaluate"]
    )
    
    return builder
