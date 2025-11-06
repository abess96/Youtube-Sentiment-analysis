"""
Advanced model training script integrated with DVC and MLflow.
Trains multiple model types including traditional ML, deep learning, transformers, and ensembles.
"""

import yaml
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Set PYTHONPATH for pickle loading
import os
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + str(project_root / 'src')

from src.models.base.model_factory import ModelFactory
from src.models.ensemble.ensemble_models import VotingEnsemble, StackingEnsemble
from src.training.hyperparameter_tuning import BayesianOptimizer, AutoTuner
from src.mlops.mlflow_integration import MLflowTracker, ModelTrainingLogger
from src.mlops.dvc_integration import DVCManager, ModelVersionManager
from src.models.base.base_model import ModelConfig
from src.config.mlflow_config import setup_mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_params():
    """Load parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_features():
    """Load processed features."""
    logger.info("Loading features...")
    
    try:
        with open('data/features/selected_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
    except ModuleNotFoundError as e:
        logger.warning(f"Module import error when loading pickle: {e}")
        logger.info("Attempting to load with custom unpickler...")
        
        # Add src to sys.path for pickle loading
        import sys
        from pathlib import Path
        src_path = str(Path(__file__).parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Try loading again
        with open('data/features/selected_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
    
    # Debug: Check what keys are available
    logger.info(f"Available keys in features_data: {list(features_data.keys()) if isinstance(features_data, dict) else type(features_data)}")
    
    # Handle different possible data structures
    if isinstance(features_data, dict):
        if 'X_train' in features_data:
            X_train = features_data['X_train']
            X_test = features_data['X_test']
            y_train = features_data['y_train']
            y_test = features_data['y_test']
        elif 'train_features' in features_data:
            X_train = features_data['train_features']
            X_test = features_data['test_features']
            y_train = features_data['train_labels']
            y_test = features_data['test_labels']
        else:
            # Try to extract from nested structure
            keys = list(features_data.keys())
            logger.error(f"Unexpected data structure. Keys: {keys}")
            raise KeyError(f"Cannot find expected keys in features data. Available: {keys}")
    else:
        # Handle tuple or other structures
        if hasattr(features_data, '__len__') and len(features_data) == 4:
            X_train, X_test, y_train, y_test = features_data
        else:
            raise ValueError(f"Unexpected data type: {type(features_data)}")
    
    # Normalize labels to ensure they start from 0
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    logger.info(f"Original label range: {unique_labels}")
    
    if unique_labels.min() != 0:
        logger.info("Normalizing labels to start from 0...")
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        logger.info(f"Normalized labels: {np.unique(np.concatenate([y_train, y_test]))}")
    
    logger.info(f"Loaded features: Train={X_train.shape}, Test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_traditional_models(X_train, y_train, X_test, y_test, params, mlflow_logger):
    """Train traditional ML models."""
    logger.info("Training traditional ML models...")
    
    model_types = params['advanced_models']['model_types']
    trained_models = {}
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type}...")
        
        try:
            # Get model hyperparameters
            model_params = params['advanced_models'].get(model_type, {})
            
            # Create model
            model = ModelFactory.create_model(model_type, **model_params)
            
            # Train with MLflow logging
            result = mlflow_logger.log_training_run(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                run_name=f"{model_type}_training"
            )
            
            trained_models[model_type] = model
            results[model_type] = result
            
            logger.info(f"{model_type} training completed")
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            results[model_type] = {'status': 'failed', 'error': str(e)}
    
    return trained_models, results


def train_ensemble_model(base_models, X_train, y_train, X_test, y_test, params, mlflow_logger):
    """Train ensemble model."""
    logger.info("Training ensemble model...")
    
    if not params['advanced_models']['enable_ensemble']:
        logger.info("Ensemble training disabled")
        return None, None
    
    ensemble_config = params['advanced_models']['ensemble']
    ensemble_type = ensemble_config['type']
    
    try:
        if ensemble_type == 'voting':
            # Create voting ensemble
            config = ModelConfig(
                model_type='voting',
                model_name='voting_ensemble',
                hyperparameters={
                    'voting_type': ensemble_config['voting_type'],
                    'weights': ensemble_config['weights']
                }
            )
            
            ensemble = VotingEnsemble(config)
            
            # Add base models
            for model in base_models.values():
                ensemble.add_model(model)
            
        elif ensemble_type == 'stacking':
            # Create stacking ensemble
            config = ModelConfig(
                model_type='stacking',
                model_name='stacking_ensemble',
                hyperparameters={'use_probabilities': True}
            )
            
            ensemble = StackingEnsemble(config)
            
            # Add base models
            for model in base_models.values():
                ensemble.add_base_model(model)
            
            # Set meta-model
            meta_model = ModelFactory.create_model('logistic_regression', max_iter=1000)
            ensemble.set_meta_model(meta_model)
        
        else:
            logger.warning(f"Unknown ensemble type: {ensemble_type}")
            return None, None
        
        # Train ensemble
        result = mlflow_logger.log_ensemble_training(
            ensemble=ensemble,
            base_models=list(base_models.values()),
            X_train=X_train,
            y_train=y_train,
            run_name=f"{ensemble_type}_ensemble"
        )
        
        logger.info("Ensemble training completed")
        
        return ensemble, result
        
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        return None, {'status': 'failed', 'error': str(e)}


def perform_hyperparameter_tuning(model_type, X_train, y_train, params, mlflow_logger):
    """Perform hyperparameter tuning."""
    logger.info(f"Performing hyperparameter tuning for {model_type}...")
    
    if not params['advanced_models']['enable_hyperparameter_tuning']:
        logger.info("Hyperparameter tuning disabled")
        return None
    
    tuning_config = params['advanced_models']['hyperparameter_tuning']
    
    try:
        # Create auto-tuner
        tuner = AutoTuner(
            model_type=model_type,
            optimization_budget=tuning_config['optimization_budget']
        )
        
        # Run tuning
        result = tuner.tune(X_train, y_train)
        
        # Log to MLflow
        mlflow_logger.log_hyperparameter_optimization(
            optimizer=tuner.best_optimizer,
            optimization_result=result,
            run_name=f"{model_type}_hyperparameter_tuning"
        )
        
        logger.info(f"Hyperparameter tuning completed for {model_type}")
        logger.info(f"Best parameters: {result['best_params']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        return None


def save_models(trained_models, ensemble_model, version_manager):
    """Save and version models."""
    logger.info("Saving models...")
    
    # Create output directory
    output_dir = Path('models/trained_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_models = {}
    
    # Save traditional models
    for model_name, model in trained_models.items():
        try:
            model_path = output_dir / f"{model_name}_model.pkl"
            model.save(str(model_path))
            saved_models[model_name] = str(model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving {model_name}: {e}")
    
    # Save ensemble model
    if ensemble_model is not None:
        try:
            ensemble_path = output_dir / "ensemble_model.pkl"
            ensemble_model.save(str(ensemble_path))
            saved_models['ensemble'] = str(ensemble_path)
            logger.info(f"Saved ensemble to {ensemble_path}")
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
    
    return saved_models


def save_metrics(results, ensemble_result):
    """Save training metrics."""
    logger.info("Saving metrics...")
    
    # Create metrics directory
    metrics_dir = Path('models/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile metrics
    metrics = {
        'traditional_models': {},
        'ensemble': {}
    }
    
    # Extract metrics from results
    for model_name, result in results.items():
        if isinstance(result, dict) and 'train_result' in result:
            train_result = result['train_result']
            metrics['traditional_models'][model_name] = {
                'status': train_result.get('status', 'unknown'),
                'run_id': result.get('run_id', 'unknown')
            }
    
    # Add ensemble metrics
    if ensemble_result is not None:
        metrics['ensemble'] = {
            'status': ensemble_result.get('status', 'unknown'),
            'run_id': ensemble_result.get('run_id', 'unknown')
        }
    
    # Save metrics
    metrics_file = metrics_dir / 'training_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_file}")


def main():
    """Main training pipeline."""
    logger.info("Starting advanced model training pipeline...")
    
    # Load parameters
    params = load_params()
    
    # Setup MLflow using existing configuration (DagsHub or local)
    setup_mlflow()
    
    # Initialize MLflow tracker with structured experiment name
    mlflow_params = params.get('mlflow', {})
    experiment_name = mlflow_params.get('experiments', {}).get('model_training', '03_Model_Training')
    mlflow_tracker = MLflowTracker(
        experiment_name=experiment_name
        # tracking_uri is handled by setup_mlflow()
    )
    mlflow_logger = ModelTrainingLogger(mlflow_tracker)
    
    # Initialize DVC
    dvc_manager = DVCManager()
    version_manager = ModelVersionManager()
    
    # Load features
    X_train, X_test, y_train, y_test = load_features()
    
    # Train traditional models
    trained_models, results = train_traditional_models(
        X_train, y_train, X_test, y_test, params, mlflow_logger
    )
    
    # Perform hyperparameter tuning if enabled
    if params['advanced_models']['enable_hyperparameter_tuning']:
        for model_type in params['advanced_models']['model_types']:
            perform_hyperparameter_tuning(
                model_type, X_train, y_train, params, mlflow_logger
            )
    
    # Train ensemble model
    ensemble_model, ensemble_result = train_ensemble_model(
        trained_models, X_train, y_train, X_test, y_test, params, mlflow_logger
    )
    
    # Save models
    saved_models = save_models(trained_models, ensemble_model, version_manager)
    
    # Save metrics
    save_metrics(results, ensemble_result)
    
    logger.info("Advanced model training pipeline completed!")
    logger.info(f"Trained {len(trained_models)} models")
    logger.info(f"Saved models: {list(saved_models.keys())}")


if __name__ == "__main__":
    main()
