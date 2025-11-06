"""
Example demonstrating continuous learning pipeline integration.
Task 8: Implement continuous learning and model updates
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import logging
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.incremental_learner import IncrementalLearner
from training.active_learner import ActiveLearner
from training.auto_retrainer import AutoRetrainer
from evaluation.drift_detection import DriftDetector
from config.mlflow_config import setup_mlflow, get_or_create_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup MLflow with structured experiment name
try:
    setup_mlflow()
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    experiment_name = params.get('mlflow', {}).get('experiments', {}).get('continuous_learning', '06_Continuous_Learning')
    get_or_create_experiment(experiment_name)
except Exception as e:
    logger.warning(f"MLflow setup failed: {e}")


def continuous_learning_pipeline_example():
    """Demonstrate complete continuous learning pipeline."""
    
    logger.info("=== Continuous Learning Pipeline Example ===\n")
    
    # Simulate initial training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    X_initial = np.random.randn(n_samples, n_features)
    y_initial = np.random.randint(0, 3, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_initial, y_initial, test_size=0.2, random_state=42
    )
    
    # 1. Initialize Incremental Learner
    logger.info("1. Initializing Incremental Learner")
    incremental_learner = IncrementalLearner(batch_size=32)
    
    # Initial training
    incremental_learner.fit_streaming(X_train, y_train, classes=[0, 1, 2])
    initial_accuracy = (incremental_learner.predict(X_test) == y_test).mean()
    logger.info(f"Initial model accuracy: {initial_accuracy:.4f}\n")
    
    # 2. Simulate streaming data updates
    logger.info("2. Processing streaming data updates")
    X_stream = np.random.randn(200, n_features)
    y_stream = np.random.randint(0, 3, 200)
    
    incremental_learner.fit_streaming(X_stream, y_stream)
    updated_accuracy = (incremental_learner.predict(X_test) == y_test).mean()
    logger.info(f"Updated model accuracy: {updated_accuracy:.4f}\n")
    
    # 3. Active Learning for sample selection
    logger.info("3. Active Learning for intelligent sample selection")
    
    # Create unlabeled pool
    X_pool = np.random.randn(500, n_features)
    
    active_learner = ActiveLearner(
        model=incremental_learner.base_model,
        strategy='uncertainty'
    )
    
    # Query most informative samples
    query_indices = active_learner.query(X_pool, n_samples=20)
    logger.info(f"Selected {len(query_indices)} samples for labeling")
    
    # Simulate human labeling
    X_to_label = X_pool[query_indices]
    y_labeled = np.random.randint(0, 3, len(query_indices))
    
    # Update model with labeled samples
    active_learner.teach(X_to_label, y_labeled)
    
    stats = active_learner.get_query_statistics()
    logger.info(f"Active learning stats: {stats}\n")
    
    # 4. Drift Detection
    logger.info("4. Drift Detection and Monitoring")
    
    drift_detector = DriftDetector(
        reference_data=X_train,
        reference_predictions=incremental_learner.predict(X_train),
        threshold=0.05
    )
    
    # Simulate data with drift
    X_drifted = np.random.randn(300, n_features) + 0.5  # Shifted distribution
    y_drifted_pred = incremental_learner.predict(X_drifted)
    
    drift_report = drift_detector.monitor(X_drifted, y_drifted_pred)
    logger.info(f"Drift status: {drift_report['overall_status']}")
    logger.info(f"Data drift detected: {drift_report['data_drift_ks']['overall_drift_detected']}")
    logger.info(f"PSI drift level: {drift_report['data_drift_psi']['drift_level']}\n")
    
    # 5. Automated Retraining
    logger.info("5. Automated Retraining Pipeline")
    
    # Use fitted model from incremental learner
    auto_retrainer = AutoRetrainer(
        model=incremental_learner.base_model,
        drift_detector=drift_detector,
        retrain_threshold=0.05
    )
    
    # Set baseline performance
    auto_retrainer.performance_baseline = (incremental_learner.predict(X_test) == y_test).mean()
    
    # Check if retraining needed
    X_new = np.random.randn(500, n_features)
    y_new = np.random.randint(0, 3, 500)
    
    should_retrain, reason = auto_retrainer.should_retrain(X_new, y_new)
    logger.info(f"Should retrain: {should_retrain}")
    logger.info(f"Reason: {reason}")
    
    if should_retrain:
        result = auto_retrainer.auto_retrain_pipeline(X_new, y_new)
        if result['retrained']:
            logger.info(f"Retraining completed successfully")
            logger.info(f"Metrics: {result['metrics']}")
    
    # 6. Drift Adaptation
    logger.info("\n6. Adapting to Drift")
    
    if drift_report['overall_status'] in ['warning', 'critical']:
        severity = 'severe' if drift_report['overall_status'] == 'critical' else 'moderate'
        incremental_learner.adapt_to_drift(X_drifted[:100], 
                                           np.random.randint(0, 3, 100),
                                           drift_severity=severity)
        logger.info(f"Model adapted to {severity} drift")
    
    # 7. Save checkpoints
    logger.info("\n7. Saving Model Checkpoints")
    
    output_dir = Path("models/continuous_learning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    incremental_learner.save_checkpoint(str(output_dir / "incremental_model.pkl"))
    auto_retrainer.save_retrain_history(str(output_dir / "retrain_history.json"))
    drift_detector.save_monitoring_history(str(output_dir / "drift_history.json"))
    
    logger.info("Checkpoints saved successfully")
    
    logger.info("\n=== Continuous Learning Pipeline Complete ===")


def text_based_continuous_learning_example():
    """Example with actual text data."""
    
    logger.info("\n=== Text-Based Continuous Learning Example ===\n")
    
    # Sample text data
    texts_initial = [
        "This is great!", "Excellent work", "Love it",
        "Terrible experience", "Very bad", "Hate this",
        "It's okay", "Not bad", "Average quality"
    ] * 50
    
    labels_initial = [1, 1, 1, 0, 0, 0, 2, 2, 2] * 50
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=100)
    X_initial = vectorizer.fit_transform(texts_initial).toarray()
    y_initial = np.array(labels_initial)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_initial, y_initial, test_size=0.2, random_state=42
    )
    
    # Initialize incremental learner
    learner = IncrementalLearner(batch_size=16)
    learner.fit_streaming(X_train, y_train, classes=[0, 1, 2])
    
    accuracy = (learner.predict(X_test) == y_test).mean()
    logger.info(f"Initial accuracy on text data: {accuracy:.4f}")
    
    # Simulate new comments
    texts_new = [
        "Amazing product!", "Worst ever", "It's fine"
    ] * 20
    labels_new = [1, 0, 2] * 20
    
    X_new = vectorizer.transform(texts_new).toarray()
    y_new = np.array(labels_new)
    
    # Update model
    learner.fit_streaming(X_new, y_new)
    
    updated_accuracy = (learner.predict(X_test) == y_test).mean()
    logger.info(f"Updated accuracy: {updated_accuracy:.4f}")
    
    logger.info("\n=== Text-Based Example Complete ===")


if __name__ == "__main__":
    continuous_learning_pipeline_example()
    text_based_continuous_learning_example()
