"""
DVC pipeline stage for training with enhanced MLflow tracking.
Integrates Task 5 components into the DVC pipeline.
"""

import sys
from pathlib import Path
import pickle
import yaml
import json
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.enhanced_mlflow_tracker import EnhancedMLflowTracker
from src.model.model_lifecycle_manager import ModelLifecycleManager
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features():
    """Load features from DVC pipeline."""
    with open('data/features/selected_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict):
        if 'X_train' in data:
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        elif 'train_features' in data:
            return data['train_features'], data['test_features'], data['train_labels'], data['test_labels']
    
    # If data is tuple/list
    if isinstance(data, (tuple, list)) and len(data) == 4:
        return data[0], data[1], data[2], data[3]
    
    raise ValueError(f"Unexpected data format in selected_features.pkl. Keys: {data.keys() if isinstance(data, dict) else type(data)}")


def load_params():
    """Load parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params.get('enhanced_tracking', {})


def train_and_track():
    """Train model with enhanced tracking."""
    logger.info("Starting enhanced tracking pipeline...")
    
    # Load data
    try:
        X_train, X_test, y_train, y_test = load_features()
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        logger.info("Using extracted_features.pkl instead...")
        with open('data/features/extracted_features.pkl', 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            X_train = data.get('X_train', data.get('train_features'))
            X_test = data.get('X_test', data.get('test_features'))
            y_train = data.get('y_train', data.get('train_labels'))
            y_test = data.get('y_test', data.get('test_labels'))
        else:
            X_train, X_test, y_train, y_test = data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Load params
    params = load_params()
    model_type = params.get('model_type', 'random_forest')
    n_estimators = params.get('n_estimators', 100)
    
    # Initialize tracker
    tracker = EnhancedMLflowTracker(experiment_name="sentiment_analysis_dvc")
    lifecycle = ModelLifecycleManager()
    
    # Train model
    with tracker.start_run(
        run_name=f"{model_type}_dvc",
        tags={'pipeline': 'dvc', 'model_type': model_type}
    ) as run:
        
        # Create model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        
        # Log params
        tracker.log_params_comprehensive({
            'model_type': model_type,
            'n_estimators': n_estimators,
            'random_state': 42
        })
        
        # Log dataset info
        tracker.log_dataset_info(X_train_split, y_train_split, 'train')
        tracker.log_dataset_info(X_val, y_val, 'validation')
        tracker.log_dataset_info(X_test, y_test, 'test')
        
        # Train
        logger.info(f"Training {model_type}...")
        model.fit(X_train_split, y_train_split)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        train_pred = model.predict(X_train_split)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train_split, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            y_val, val_pred, average='weighted'
        )
        
        # Log metrics
        tracker.log_comprehensive_metrics({
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1
        })
        
        # Log confusion matrix
        tracker.log_confusion_matrix(y_val, val_pred)
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(X_train_split.shape[1])]
            tracker.log_feature_importance(feature_names, model.feature_importances_)
        
        # Log model
        tracker.log_model_with_signature(model, 'model', X_val[:5], 'sklearn')
        
        run_id = run.info.run_id
    
    # Register model
    model_version = lifecycle.register_model_version(
        run_id=run_id,
        model_name="sentiment_classifier_dvc",
        tags={'pipeline': 'dvc', 'model_type': model_type},
        description=f"Model trained via DVC pipeline with {model_type}"
    )
    
    # Save metrics for DVC
    metrics = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'val_f1': float(val_f1),
        'model_version': model_version.version,
        'run_id': run_id
    }
    
    Path('models/metrics').mkdir(parents=True, exist_ok=True)
    with open('models/metrics/enhanced_tracking_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    Path('models/enhanced_tracked').mkdir(parents=True, exist_ok=True)
    with open('models/enhanced_tracked/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model registered: v{model_version.version}")
    logger.info(f"Metrics saved to models/metrics/enhanced_tracking_metrics.json")
    
    return metrics


if __name__ == "__main__":
    train_and_track()
