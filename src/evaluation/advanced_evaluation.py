"""
Advanced model evaluation script for all model types.
Evaluates traditional ML, deep learning, transformer, and ensemble models.
"""

import yaml
import pickle
import json
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.models.base.model_factory import ModelFactory
from src.mlops.mlflow_integration import MLflowTracker
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
    
    with open('data/features/selected_features.pkl', 'rb') as f:
        features_data = pickle.load(f)
    
    X_test = features_data['X_test']
    y_test = features_data['y_test']
    
    logger.info(f"Loaded test features: {X_test.shape}")
    
    return X_test, y_test


def load_models():
    """Load all trained models."""
    logger.info("Loading trained models...")
    
    models_dir = Path('models/trained_models')
    loaded_models = {}
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return loaded_models
    
    # Load all .pkl files
    for model_file in models_dir.glob('*.pkl'):
        model_name = model_file.stem.replace('_model', '')
        
        try:
            # Try to load as BaseModel
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            loaded_models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    return loaded_models


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model."""
    logger.info(f"Evaluating {model_name}...")
    
    try:
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def compare_models(all_metrics):
    """Compare all models and find the best one."""
    logger.info("Comparing models...")
    
    comparison = {}
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in all_metrics.items():
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            f1 = metrics['f1_score']
            
            comparison[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
    
    logger.info(f"Best model: {best_model} with accuracy {best_accuracy:.4f}")
    
    return comparison, best_model


def save_evaluation_results(all_metrics, comparison, best_model):
    """Save evaluation results."""
    logger.info("Saving evaluation results...")
    
    # Create metrics directory
    metrics_dir = Path('models/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile results
    results = {
        'model_metrics': all_metrics,
        'comparison': comparison,
        'best_model': best_model
    }
    
    # Save to file
    output_file = metrics_dir / 'evaluation_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")
    
    return results


def log_to_mlflow(all_metrics, comparison, best_model, params):
    """Log evaluation results to MLflow."""
    logger.info("Logging evaluation results to MLflow...")
    
    # Setup MLflow using existing configuration
    setup_mlflow()
    
    mlflow_params = params.get('mlflow', {})
    experiment_name = mlflow_params.get('experiments', {}).get('model_evaluation', '04_Model_Evaluation')
    tracker = MLflowTracker(
        experiment_name=experiment_name
    )
    
    with tracker.start_run(run_name="model_evaluation") as run:
        # Log comparison metrics
        for model_name, metrics in comparison.items():
            for metric_name, value in metrics.items():
                tracker.log_metrics({f"{model_name}_{metric_name}": value})
        
        # Log best model
        tracker.set_tags({
            'best_model': best_model,
            'evaluation_stage': 'complete'
        })
        
        # Log detailed metrics as artifact
        tracker.log_dict(all_metrics, "detailed_metrics.json")
        
        logger.info(f"Evaluation logged to MLflow: {run.info.run_id}")


def main():
    """Main evaluation pipeline."""
    logger.info("Starting advanced model evaluation pipeline...")
    
    # Load parameters
    params = load_params()
    
    # Load test data
    X_test, y_test = load_features()
    
    # Load models
    models = load_models()
    
    if not models:
        logger.error("No models found to evaluate!")
        return
    
    # Evaluate all models
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[model_name] = metrics
    
    # Compare models
    comparison, best_model = compare_models(all_metrics)
    
    # Save results
    results = save_evaluation_results(all_metrics, comparison, best_model)
    
    # Log to MLflow
    log_to_mlflow(all_metrics, comparison, best_model, params)
    
    logger.info("Advanced model evaluation pipeline completed!")
    logger.info(f"Evaluated {len(models)} models")
    logger.info(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
