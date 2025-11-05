"""
Production-ready continuous learning pipeline.
Task 8: Implement continuous learning and model updates
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import pickle
from typing import Dict, Any, Optional
from datetime import datetime

from incremental_learner import IncrementalLearner
from active_learner import ActiveLearner
from auto_retrainer import AutoRetrainer
from drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """Production continuous learning pipeline."""
    
    def __init__(self, config_path: str = "params.yaml"):
        self.config = self._load_config(config_path)
        self.incremental_learner = None
        self.active_learner = None
        self.auto_retrainer = None
        self.drift_detector = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from params.yaml."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize(self, X_train, y_train, model=None):
        """Initialize all components."""
        config = self.config['continuous_learning']
        
        # Initialize incremental learner
        if config['enable_incremental_learning']:
            batch_size = config['incremental_learning']['batch_size']
            self.incremental_learner = IncrementalLearner(
                base_model=model,
                batch_size=batch_size
            )
            self.incremental_learner.fit_streaming(X_train, y_train, 
                                                   classes=np.unique(y_train))
            logger.info("Incremental learner initialized")
        
        # Initialize active learner
        if config['enable_active_learning']:
            strategy = config['active_learning']['strategy']
            model_for_active = self.incremental_learner.base_model if self.incremental_learner else model
            self.active_learner = ActiveLearner(
                model=model_for_active,
                strategy=strategy
            )
            logger.info(f"Active learner initialized with {strategy} strategy")
        
        # Initialize drift detector
        y_pred_train = self.incremental_learner.predict(X_train) if self.incremental_learner else model.predict(X_train)
        self.drift_detector = DriftDetector(
            reference_data=X_train,
            reference_predictions=y_pred_train,
            threshold=self.config['evaluation']['drift_threshold']
        )
        self.drift_detector.y_true_ref = y_train
        self.drift_detector.y_pred_ref = y_pred_train
        logger.info("Drift detector initialized")
        
        # Initialize auto retrainer
        if config['enable_auto_retraining']:
            retrain_config = config['auto_retraining']
            model_for_retrain = self.incremental_learner.base_model if self.incremental_learner else model
            self.auto_retrainer = AutoRetrainer(
                model=model_for_retrain,
                drift_detector=self.drift_detector,
                retrain_threshold=retrain_config['retrain_threshold'],
                min_samples=retrain_config['min_samples_for_retrain']
            )
            logger.info("Auto retrainer initialized")
    
    def process_new_data(self, X_new, y_new=None) -> Dict[str, Any]:
        """Process new incoming data."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_new)
        }
        
        # Make predictions
        if self.incremental_learner:
            y_pred = self.incremental_learner.predict(X_new)
            results['predictions'] = y_pred.tolist()
        
        # Monitor drift
        if self.drift_detector:
            drift_report = self.drift_detector.monitor(X_new, y_pred, y_new)
            results['drift_status'] = drift_report['overall_status']
            results['drift_detected'] = drift_report['data_drift_ks']['overall_drift_detected']
            
            # Generate alert if needed
            if drift_report['overall_status'] in ['warning', 'critical']:
                alert = self.drift_detector.generate_alert(drift_report)
                results['alert'] = alert
                logger.warning(f"Drift alert: {alert['messages']}")
        
        # Check if retraining needed
        if self.auto_retrainer and y_new is not None:
            should_retrain, reason = self.auto_retrainer.should_retrain(X_new, y_new, y_pred)
            results['should_retrain'] = should_retrain
            results['retrain_reason'] = reason
            
            if should_retrain:
                logger.info(f"Retraining triggered: {reason}")
                retrain_result = self.auto_retrainer.auto_retrain_pipeline(X_new, y_new)
                results['retrain_result'] = retrain_result
        
        return results
    
    def incremental_update(self, X_new, y_new):
        """Incrementally update model with new labeled data."""
        if not self.incremental_learner:
            logger.warning("Incremental learner not initialized")
            return
        
        self.incremental_learner.fit_streaming(X_new, y_new)
        logger.info(f"Model incrementally updated with {len(X_new)} samples")
    
    def query_for_labeling(self, X_pool, n_samples: int = 10) -> np.ndarray:
        """Query most informative samples for labeling."""
        if not self.active_learner:
            logger.warning("Active learner not initialized")
            return np.random.choice(len(X_pool), n_samples, replace=False)
        
        indices = self.active_learner.query(X_pool, n_samples)
        logger.info(f"Queried {n_samples} samples for labeling")
        return indices
    
    def update_with_feedback(self, X_labeled, y_labeled):
        """Update model with human-labeled feedback."""
        if self.active_learner:
            self.active_learner.teach(X_labeled, y_labeled)
        elif self.incremental_learner:
            self.incremental_learner.partial_fit(X_labeled, y_labeled)
        
        logger.info(f"Model updated with {len(X_labeled)} feedback samples")
    
    def save_state(self, output_dir: str = "models/continuous_learning"):
        """Save pipeline state."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.incremental_learner:
            self.incremental_learner.save_checkpoint(
                str(output_path / "incremental_model.pkl")
            )
        
        if self.auto_retrainer:
            self.auto_retrainer.save_retrain_history(
                str(output_path / "retrain_history.json")
            )
            self.auto_retrainer.save_model(
                str(output_path / "retrained_model.pkl")
            )
        
        if self.drift_detector:
            self.drift_detector.save_monitoring_history(
                str(output_path / "drift_history.json")
            )
        
        logger.info(f"Pipeline state saved to {output_dir}")
    
    def load_state(self, input_dir: str = "models/continuous_learning"):
        """Load pipeline state."""
        input_path = Path(input_dir)
        
        if self.incremental_learner and (input_path / "incremental_model.pkl").exists():
            self.incremental_learner.load_checkpoint(
                str(input_path / "incremental_model.pkl")
            )
        
        logger.info(f"Pipeline state loaded from {input_dir}")


def run_continuous_learning_pipeline():
    """Run the continuous learning pipeline."""
    logger.info("=== Starting Continuous Learning Pipeline ===\n")
    
    # Load data
    logger.info("Loading training data...")
    try:
        with open("data/features/extracted_features.pkl", "rb") as f:
            features_data = pickle.load(f)
        
        X_train = features_data['X_train']
        y_train = features_data['y_train']
        X_test = features_data['X_test']
        y_test = features_data['y_test']
        
        logger.info(f"Loaded {len(X_train)} training samples")
    except FileNotFoundError:
        logger.error("Feature data not found. Run feature extraction first.")
        return
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline()
    pipeline.initialize(X_train, y_train)
    
    # Simulate new data stream
    logger.info("\nSimulating new data stream...")
    n_new = min(200, len(X_test))
    X_new = X_test[:n_new]
    y_new = y_test[:n_new]
    
    # Process new data
    results = pipeline.process_new_data(X_new, y_new)
    logger.info(f"Processing results: {results['drift_status']}")
    
    # Incremental update
    if results.get('should_retrain'):
        logger.info("Performing incremental update...")
        pipeline.incremental_update(X_new, y_new)
    
    # Active learning simulation
    if pipeline.active_learner:
        logger.info("\nSimulating active learning...")
        X_pool = X_test[n_new:n_new+100]
        query_indices = pipeline.query_for_labeling(X_pool, n_samples=10)
        
        # Simulate labeling
        X_to_label = X_pool[query_indices]
        y_labeled = y_test[n_new:n_new+100][query_indices]
        
        pipeline.update_with_feedback(X_to_label, y_labeled)
    
    # Save state
    pipeline.save_state()
    
    logger.info("\n=== Continuous Learning Pipeline Complete ===")


if __name__ == "__main__":
    run_continuous_learning_pipeline()
