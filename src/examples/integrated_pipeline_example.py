"""
Integrated pipeline example showing Task 5 components with existing pipeline.
Demonstrates end-to-end workflow from training to deployment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import pickle
import mlflow

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mlops.experiment_tracking import EnhancedMLflowTracker
from src.mlops.lifecycle_manager import ModelLifecycleManager
from src.mlops.ab_testing import ABTestingFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedMLPipeline:
    """
    Integrated ML pipeline with enhanced experiment tracking and model management.
    Combines existing pipeline components with Task 5 enhancements.
    """
    
    def __init__(self, experiment_name: str = "sentiment_analysis_production"):
        """
        Initialize integrated pipeline.
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.tracker = EnhancedMLflowTracker(experiment_name=experiment_name)
        self.lifecycle_manager = ModelLifecycleManager()
        self.experiment_name = experiment_name
        
        logger.info(f"Initialized integrated pipeline: {experiment_name}")
    
    def train_and_register_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 model: any, model_name: str,
                                 model_config: dict,
                                 version_tag: str = "v1.0") -> tuple:
        """
        Train model with comprehensive tracking and register to model registry.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model: Model instance to train
            model_name: Name for registered model
            model_config: Model configuration dictionary
            version_tag: Version tag for the model
            
        Returns:
            Tuple of (trained_model, run_id, model_version)
        """
        logger.info(f"Training and registering model: {model_name}")
        
        with self.tracker.start_run(
            run_name=f"{model_name}_{version_tag}",
            tags={
                'model_name': model_name,
                'version': version_tag,
                'stage': 'training'
            }
        ) as run:
            # Log configuration
            self.tracker.log_params_comprehensive(model_config)
            
            # Log dataset information
            self.tracker.log_dataset_info(X_train, y_train, 'train')
            self.tracker.log_dataset_info(X_val, y_val, 'validation')
            
            # Train model
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            y_train_pred = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            
            # Log metrics
            self.tracker.log_comprehensive_metrics(train_metrics, prefix='train')
            self.tracker.log_comprehensive_metrics(val_metrics, prefix='val')
            
            # Log confusion matrix
            self.tracker.log_confusion_matrix(
                y_val, y_val_pred,
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                self.tracker.log_feature_importance(
                    feature_names,
                    model.feature_importances_
                )
            
            # Log model with signature
            self.tracker.log_model_with_signature(
                model, 'model',
                input_example=X_val[:5],
                model_type='sklearn'
            )
            
            # Log system metrics
            self.tracker.log_system_metrics()
            
            run_id = run.info.run_id
        
        # Register model to model registry
        model_version = self.lifecycle_manager.register_model_version(
            run_id=run_id,
            model_name=model_name,
            tags={
                'version': version_tag,
                'algorithm': model.__class__.__name__,
                'val_accuracy': str(val_metrics['accuracy'])
            },
            description=f"Model trained with {model.__class__.__name__}"
        )
        
        logger.info(f"Model registered: {model_name} v{model_version.version}")
        
        return model, run_id, model_version
    
    def compare_and_deploy(self, new_model: any, X_test: np.ndarray, y_test: np.ndarray,
                          model_name: str, new_version: str,
                          confidence_level: float = 0.95) -> dict:
        """
        Compare new model with production model and deploy if better.
        
        Args:
            new_model: New model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of registered model
            new_version: Version of new model
            confidence_level: Confidence level for A/B test
            
        Returns:
            Dictionary with deployment decision and results
        """
        logger.info("Comparing new model with production model...")
        
        # Get current production model
        prod_model = self.lifecycle_manager.get_production_model(model_name)
        
        if prod_model is None:
            logger.info("No production model found. Promoting new model to production.")
            
            # Promote to staging first
            self.lifecycle_manager.promote_model(
                model_name, new_version,
                ModelLifecycleManager.STAGE_STAGING
            )
            
            # Then to production
            self.lifecycle_manager.promote_model(
                model_name, new_version,
                ModelLifecycleManager.STAGE_PRODUCTION
            )
            
            return {
                'deployed': True,
                'reason': 'No existing production model',
                'new_version': new_version
            }
        
        # Run A/B test
        ab_test = ABTestingFramework(
            model_a=prod_model,
            model_b=new_model,
            model_a_name="Production Model",
            model_b_name=f"Candidate v{new_version}"
        )
        
        results = ab_test.run_ab_test(
            X_test, y_test,
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            confidence_level=confidence_level
        )
        
        # Get recommendation
        recommendation = ab_test.get_recommendation()
        
        # Save A/B test results
        ab_test.generate_report(f'ab_test_{model_name}_v{new_version}.json')
        ab_test.visualize_results(f'ab_test_{model_name}_v{new_version}.png')
        
        # Log A/B test results to MLflow
        with self.tracker.start_run(
            run_name=f"ab_test_{model_name}_v{new_version}",
            tags={'stage': 'ab_testing', 'model_name': model_name}
        ):
            self.tracker.log_comprehensive_metrics(results['comparison'])
            import json
            with open('ab_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact('ab_test_results.json')
            with open('recommendation.json', 'w') as f:
                json.dump(recommendation, f, indent=2, default=str)
            mlflow.log_artifact('recommendation.json')
        
        # Make deployment decision
        should_deploy = (
            recommendation['recommendation'].startswith('Deploy') and
            recommendation['confidence'] in ['high', 'medium']
        )
        
        if should_deploy:
            logger.info(f"Deploying new model v{new_version} to production")
            
            # Promote to staging first
            self.lifecycle_manager.promote_model(
                model_name, new_version,
                ModelLifecycleManager.STAGE_STAGING
            )
            
            # Then to production (archives existing)
            self.lifecycle_manager.promote_model(
                model_name, new_version,
                ModelLifecycleManager.STAGE_PRODUCTION,
                archive_existing=True
            )
            
            return {
                'deployed': True,
                'reason': recommendation['recommendation'],
                'confidence': recommendation['confidence'],
                'new_version': new_version,
                'ab_test_results': results
            }
        else:
            logger.info("Keeping current production model")
            
            return {
                'deployed': False,
                'reason': recommendation['recommendation'],
                'confidence': recommendation['confidence'],
                'ab_test_results': results
            }
    
    def rollback_production(self, model_name: str, target_version: str = None) -> bool:
        """
        Rollback production model to previous version.
        
        Args:
            model_name: Name of registered model
            target_version: Specific version to rollback to (optional)
            
        Returns:
            Success status
        """
        logger.info(f"Rolling back production model: {model_name}")
        
        success = self.lifecycle_manager.rollback_model(
            model_name,
            ModelLifecycleManager.STAGE_PRODUCTION,
            target_version
        )
        
        if success:
            logger.info("Rollback successful")
        else:
            logger.error("Rollback failed")
        
        return success
    
    def get_model_performance_history(self, model_name: str) -> pd.DataFrame:
        """
        Get performance history of all model versions.
        
        Args:
            model_name: Name of registered model
            
        Returns:
            DataFrame with version performance history
        """
        lineage = self.lifecycle_manager.get_model_lineage(model_name)
        
        history = []
        for version_info in lineage:
            version = version_info['version']
            info = self.lifecycle_manager.get_model_version_info(model_name, version)
            
            history.append({
                'version': version,
                'stage': version_info['stage'],
                'created_at': version_info['created_at'],
                'accuracy': info['metrics'].get('val_accuracy', 0),
                'f1_score': info['metrics'].get('val_f1', 0)
            })
        
        return pd.DataFrame(history)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def example_integrated_workflow():
    """Example of complete integrated workflow."""
    logger.info("=== Integrated Pipeline Example ===")
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    X, y = make_classification(
        n_samples=2000, n_features=20, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Initialize pipeline
    pipeline = IntegratedMLPipeline(experiment_name="sentiment_analysis_demo")
    
    # Train and register initial model
    logger.info("\n--- Training Initial Model ---")
    model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    trained_model_v1, run_id_v1, version_v1 = pipeline.train_and_register_model(
        X_train, y_train, X_val, y_val,
        model=model_v1,
        model_name="sentiment_classifier",
        model_config={'n_estimators': 100, 'algorithm': 'RandomForest'},
        version_tag="v1.0"
    )
    
    # Deploy initial model (no production model exists)
    logger.info("\n--- Deploying Initial Model ---")
    deployment_result = pipeline.compare_and_deploy(
        trained_model_v1, X_test, y_test,
        model_name="sentiment_classifier",
        new_version=version_v1.version
    )
    logger.info(f"Deployment result: {deployment_result}")
    
    # Train improved model
    logger.info("\n--- Training Improved Model ---")
    model_v2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    trained_model_v2, run_id_v2, version_v2 = pipeline.train_and_register_model(
        X_train, y_train, X_val, y_val,
        model=model_v2,
        model_name="sentiment_classifier",
        model_config={'n_estimators': 100, 'algorithm': 'GradientBoosting'},
        version_tag="v2.0"
    )
    
    # Compare and potentially deploy
    logger.info("\n--- Comparing Models ---")
    deployment_result = pipeline.compare_and_deploy(
        trained_model_v2, X_test, y_test,
        model_name="sentiment_classifier",
        new_version=version_v2.version
    )
    logger.info(f"Deployment result: {deployment_result}")
    
    # Get performance history
    logger.info("\n--- Model Performance History ---")
    history = pipeline.get_model_performance_history("sentiment_classifier")
    logger.info(f"\n{history}")
    
    # Demonstrate rollback (if needed)
    if deployment_result['deployed']:
        logger.info("\n--- Demonstrating Rollback ---")
        pipeline.rollback_production("sentiment_classifier")
    
    logger.info("\n=== Integrated Pipeline Example Completed ===")


if __name__ == "__main__":
    try:
        example_integrated_workflow()
    except Exception as e:
        logger.error(f"Error in integrated workflow: {e}", exc_info=True)
