"""
Automated model lifecycle management system.
Handles model versioning, promotion, rollback, and semantic versioning.
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.mlflow_config import setup_mlflow

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """
    Manages model lifecycle including versioning, promotion, and rollback.
    Implements semantic versioning and automated model management.
    """
    
    # Model stages
    STAGE_NONE = "None"
    STAGE_STAGING = "Staging"
    STAGE_PRODUCTION = "Production"
    STAGE_ARCHIVED = "Archived"
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize model lifecycle manager.
        
        Args:
            tracking_uri: Optional MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            setup_mlflow()
        
        self.client = MlflowClient()
        logger.info("Model lifecycle manager initialized")
    
    def register_model_version(self, run_id: str, model_name: str,
                              artifact_path: str = "model",
                              tags: Optional[Dict[str, str]] = None,
                              description: Optional[str] = None) -> Any:
        """
        Register a new model version.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact in run
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            ModelVersion object
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Add metadata
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        model_name,
                        model_version.version,
                        key,
                        value
                    )
            
            # Add timestamp tag
            self.client.set_model_version_tag(
                model_name,
                model_version.version,
                "registered_at",
                datetime.now().isoformat()
            )
            
            if description:
                self.client.update_model_version(
                    model_name,
                    model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} v{model_version.version}")
            return model_version
        
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def promote_model(self, model_name: str, version: str,
                     stage: str, archive_existing: bool = True) -> bool:
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage (Staging/Production)
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            Success status
        """
        try:
            # Archive existing models in target stage if requested
            if archive_existing:
                existing_versions = self.client.get_latest_versions(
                    model_name, 
                    stages=[stage]
                )
                for existing in existing_versions:
                    self.client.transition_model_version_stage(
                        model_name,
                        existing.version,
                        self.STAGE_ARCHIVED
                    )
                    logger.info(f"Archived {model_name} v{existing.version}")
            
            # Promote new version
            self.client.transition_model_version_stage(
                model_name,
                version,
                stage
            )
            
            # Add promotion metadata
            self.client.set_model_version_tag(
                model_name,
                version,
                f"promoted_to_{stage.lower()}_at",
                datetime.now().isoformat()
            )
            
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
        
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def rollback_model(self, model_name: str, stage: str,
                      target_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the registered model
            stage: Stage to rollback (Staging/Production)
            target_version: Specific version to rollback to (if None, uses previous)
            
        Returns:
            Success status
        """
        try:
            # Get current version in stage
            current_versions = self.client.get_latest_versions(
                model_name,
                stages=[stage]
            )
            
            if not current_versions:
                logger.warning(f"No model in {stage} stage to rollback from")
                return False
            
            current_version = current_versions[0]
            
            # Determine target version
            if target_version is None:
                # Get all versions sorted by version number
                all_versions = self.client.search_model_versions(
                    f"name='{model_name}'"
                )
                sorted_versions = sorted(
                    all_versions,
                    key=lambda x: int(x.version),
                    reverse=True
                )
                
                # Find previous version
                for i, v in enumerate(sorted_versions):
                    if v.version == current_version.version and i < len(sorted_versions) - 1:
                        target_version = sorted_versions[i + 1].version
                        break
                
                if target_version is None:
                    logger.warning("No previous version found for rollback")
                    return False
            
            # Archive current version
            self.client.transition_model_version_stage(
                model_name,
                current_version.version,
                self.STAGE_ARCHIVED
            )
            
            # Promote target version
            self.client.transition_model_version_stage(
                model_name,
                target_version,
                stage
            )
            
            # Add rollback metadata
            self.client.set_model_version_tag(
                model_name,
                target_version,
                "rolled_back_at",
                datetime.now().isoformat()
            )
            self.client.set_model_version_tag(
                model_name,
                target_version,
                "rolled_back_from",
                current_version.version
            )
            
            logger.info(f"Rolled back {model_name} from v{current_version.version} to v{target_version}")
            return True
        
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            return False
    
    def get_model_version_info(self, model_name: str, 
                              version: Optional[str] = None,
                              stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a model version.
        
        Args:
            model_name: Name of the registered model
            version: Specific version (if None, uses latest in stage)
            stage: Stage to get version from (if version is None)
            
        Returns:
            Dictionary with model version information
        """
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            elif stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not versions:
                    return {}
                model_version = versions[0]
            else:
                # Get latest version
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    return {}
                model_version = max(versions, key=lambda x: int(x.version))
            
            # Get run metrics
            run = self.client.get_run(model_version.run_id)
            
            return {
                'name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'creation_timestamp': datetime.fromtimestamp(
                    model_version.creation_timestamp / 1000
                ).isoformat(),
                'last_updated_timestamp': datetime.fromtimestamp(
                    model_version.last_updated_timestamp / 1000
                ).isoformat(),
                'description': model_version.description,
                'tags': model_version.tags,
                'metrics': run.data.metrics,
                'params': run.data.params
            }
        
        except Exception as e:
            logger.error(f"Error getting model version info: {e}")
            return {}
    
    def compare_model_versions(self, model_name: str, 
                              version1: str, version2: str,
                              metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            metrics: Optional list of specific metrics to compare
            
        Returns:
            Comparison results
        """
        info1 = self.get_model_version_info(model_name, version1)
        info2 = self.get_model_version_info(model_name, version2)
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics_comparison': {}
        }
        
        # Compare metrics
        if metrics:
            for metric in metrics:
                val1 = info1['metrics'].get(metric)
                val2 = info2['metrics'].get(metric)
                if val1 is not None and val2 is not None:
                    comparison['metrics_comparison'][metric] = {
                        'version1': val1,
                        'version2': val2,
                        'difference': val2 - val1,
                        'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else None
                    }
        else:
            # Compare all common metrics
            common_metrics = set(info1['metrics'].keys()) & set(info2['metrics'].keys())
            for metric in common_metrics:
                val1 = info1['metrics'][metric]
                val2 = info2['metrics'][metric]
                comparison['metrics_comparison'][metric] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': val2 - val1,
                    'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else None
                }
        
        return comparison
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Get the current production model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Loaded model or None
        """
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=[self.STAGE_PRODUCTION]
            )
            
            if not versions:
                logger.warning(f"No production model found for {model_name}")
                return None
            
            model_version = versions[0]
            model_uri = f"models:/{model_name}/{self.STAGE_PRODUCTION}"
            
            # Load model based on flavor
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except:
                try:
                    model = mlflow.pytorch.load_model(model_uri)
                except:
                    model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded production model: {model_name} v{model_version.version}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading production model: {e}")
            return None
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Version to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            return False
    
    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get complete lineage of a model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of version information sorted by version number
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            lineage = []
            for version in sorted(versions, key=lambda x: int(x.version)):
                lineage.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'created_at': datetime.fromtimestamp(
                        version.creation_timestamp / 1000
                    ).isoformat(),
                    'tags': version.tags,
                    'run_id': version.run_id
                })
            
            return lineage
        
        except Exception as e:
            logger.error(f"Error getting model lineage: {e}")
            return []
    
    def apply_semantic_version(self, model_name: str, version: str,
                              semantic_version: str) -> bool:
        """
        Apply semantic versioning tag to a model version.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            semantic_version: Semantic version (e.g., "1.2.3")
            
        Returns:
            Success status
        """
        try:
            self.client.set_model_version_tag(
                model_name,
                version,
                "semantic_version",
                semantic_version
            )
            logger.info(f"Applied semantic version {semantic_version} to {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Error applying semantic version: {e}")
            return False
