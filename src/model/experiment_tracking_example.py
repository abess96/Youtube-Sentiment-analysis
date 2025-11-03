"""
Example usage of enhanced experiment tracking and model management.
Demonstrates Task 5 components: enhanced MLflow tracking, lifecycle management, and A/B testing.
"""

import numpy as np
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.enhanced_mlflow_tracker import EnhancedMLflowTracker, ExperimentComparator
from src.model.model_lifecycle_manager import ModelLifecycleManager
from src.model.ab_testing_framework import ABTestingFramework, MultiModelABTest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_enhanced_tracking():
    """Example: Enhanced MLflow experiment tracking."""
    logger.info("=== Enhanced Experiment Tracking Example ===")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize enhanced tracker
    tracker = EnhancedMLflowTracker(experiment_name="enhanced_tracking_demo")
    
    # Train and log model with comprehensive tracking
    with tracker.start_run(run_name="random_forest_enhanced", 
                          tags={'model_type': 'ensemble', 'version': 'v1.0'}):
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log parameters
        tracker.log_params_comprehensive({
            'model': 'RandomForest',
            'n_estimators': 100,
            'random_state': 42,
            'hyperparameters': {
                'max_depth': None,
                'min_samples_split': 2
            }
        })
        
        # Log dataset info
        tracker.log_dataset_info(X_train, y_train, 'train')
        tracker.log_dataset_info(X_test, y_test, 'test')
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Log comprehensive metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        tracker.log_comprehensive_metrics({
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Log confusion matrix
        tracker.log_confusion_matrix(y_test, y_pred, labels=['Class 0', 'Class 1', 'Class 2'])
        
        # Log feature importance
        tracker.log_feature_importance(
            [f'feature_{i}' for i in range(X.shape[1])],
            model.feature_importances_
        )
        
        # Log model with signature
        tracker.log_model_with_signature(
            model, 
            'model',
            input_example=X_test[:5],
            model_type='sklearn'
        )
        
        # Log system metrics
        tracker.log_system_metrics()
        
        logger.info("Enhanced tracking completed successfully")


def example_lifecycle_management():
    """Example: Model lifecycle management."""
    logger.info("=== Model Lifecycle Management Example ===")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize components
    tracker = EnhancedMLflowTracker(experiment_name="lifecycle_demo")
    lifecycle_manager = ModelLifecycleManager()
    
    # Train and register model v1
    with tracker.start_run(run_name="model_v1") as run:
        model_v1 = LogisticRegression(random_state=42)
        model_v1.fit(X_train, y_train)
        
        acc_v1 = model_v1.score(X_test, y_test)
        tracker.log_comprehensive_metrics({'accuracy': acc_v1})
        tracker.log_model_with_signature(model_v1, 'model', X_test[:5], 'sklearn')
        
        run_id_v1 = run.info.run_id
    
    # Register model v1
    model_version_v1 = lifecycle_manager.register_model_version(
        run_id=run_id_v1,
        model_name="sentiment_classifier",
        tags={'version': 'v1.0', 'algorithm': 'logistic_regression'},
        description="Initial logistic regression model"
    )
    
    # Promote to staging
    lifecycle_manager.promote_model(
        "sentiment_classifier",
        model_version_v1.version,
        ModelLifecycleManager.STAGE_STAGING
    )
    
    # Train improved model v2
    with tracker.start_run(run_name="model_v2") as run:
        model_v2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_v2.fit(X_train, y_train)
        
        acc_v2 = model_v2.score(X_test, y_test)
        tracker.log_comprehensive_metrics({'accuracy': acc_v2})
        tracker.log_model_with_signature(model_v2, 'model', X_test[:5], 'sklearn')
        
        run_id_v2 = run.info.run_id
    
    # Register model v2
    model_version_v2 = lifecycle_manager.register_model_version(
        run_id=run_id_v2,
        model_name="sentiment_classifier",
        tags={'version': 'v2.0', 'algorithm': 'gradient_boosting'},
        description="Improved gradient boosting model"
    )
    
    # Compare versions
    comparison = lifecycle_manager.compare_model_versions(
        "sentiment_classifier",
        model_version_v1.version,
        model_version_v2.version,
        metrics=['accuracy']
    )
    
    logger.info(f"Model comparison: {comparison}")
    
    # Promote v2 to production if better
    if acc_v2 > acc_v1:
        lifecycle_manager.promote_model(
            "sentiment_classifier",
            model_version_v2.version,
            ModelLifecycleManager.STAGE_PRODUCTION,
            archive_existing=True
        )
        logger.info("Model v2 promoted to production")
    
    # Apply semantic versioning
    lifecycle_manager.apply_semantic_version(
        "sentiment_classifier",
        model_version_v2.version,
        "2.0.0"
    )
    
    # Get model lineage
    lineage = lifecycle_manager.get_model_lineage("sentiment_classifier")
    logger.info(f"Model lineage: {lineage}")
    
    # Get production model
    prod_model = lifecycle_manager.get_production_model("sentiment_classifier")
    if prod_model:
        logger.info("Successfully loaded production model")


def example_ab_testing():
    """Example: A/B testing framework."""
    logger.info("=== A/B Testing Example ===")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train two models
    model_a = LogisticRegression(random_state=42)
    model_a.fit(X_train, y_train)
    
    model_b = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b.fit(X_train, y_train)
    
    # Initialize A/B test
    ab_test = ABTestingFramework(
        model_a=model_a,
        model_b=model_b,
        model_a_name="Logistic Regression",
        model_b_name="Random Forest"
    )
    
    # Run A/B test
    results = ab_test.run_ab_test(
        X_test, y_test,
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        confidence_level=0.95
    )
    
    # Display results
    logger.info("\n=== A/B Test Results ===")
    logger.info(f"Model A metrics: {results['model_a']['metrics']}")
    logger.info(f"Model B metrics: {results['model_b']['metrics']}")
    logger.info(f"Statistical tests: {results['statistical_tests']}")
    
    # Get recommendation
    recommendation = ab_test.get_recommendation()
    logger.info(f"\nRecommendation: {recommendation}")
    
    # Generate report
    ab_test.generate_report('ab_test_results.json')
    ab_test.visualize_results('ab_test_comparison.png')
    
    logger.info("A/B test completed successfully")


def example_multi_model_tournament():
    """Example: Multi-model tournament."""
    logger.info("=== Multi-Model Tournament Example ===")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Run tournament
    tournament = MultiModelABTest(models)
    results = tournament.run_tournament(
        X_test, y_test,
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    # Display results
    logger.info("\n=== Tournament Results ===")
    logger.info(f"Rankings: {results['rankings']}")
    logger.info(f"Winner: {results['winner']}")
    logger.info(f"Pairwise comparisons: {results['pairwise_comparisons']}")


def example_experiment_comparison():
    """Example: Compare multiple experiments."""
    logger.info("=== Experiment Comparison Example ===")
    
    # Create multiple experiments with different configurations
    experiment_names = []
    
    for i, n_estimators in enumerate([50, 100, 200]):
        exp_name = f"rf_experiment_{n_estimators}"
        experiment_names.append(exp_name)
        
        tracker = EnhancedMLflowTracker(experiment_name=exp_name)
        
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                                   n_informative=10, n_redundant=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with tracker.start_run(run_name=f"rf_{n_estimators}_trees"):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            acc = model.score(X_test, y_test)
            tracker.log_comprehensive_metrics({
                'accuracy': acc,
                'n_estimators': n_estimators
            })
    
    # Compare experiments
    comparator = ExperimentComparator(experiment_names)
    comparison_df = comparator.compare_experiments('accuracy', top_n=5)
    
    logger.info("\n=== Experiment Comparison ===")
    logger.info(f"\n{comparison_df}")
    
    # Generate comparison report
    comparator.generate_comparison_report(
        metrics=['accuracy'],
        output_path='experiment_comparison_report.html'
    )


if __name__ == "__main__":
    # Run all examples
    try:
        example_enhanced_tracking()
        print("\n" + "="*80 + "\n")
        
        example_lifecycle_management()
        print("\n" + "="*80 + "\n")
        
        example_ab_testing()
        print("\n" + "="*80 + "\n")
        
        example_multi_model_tournament()
        print("\n" + "="*80 + "\n")
        
        example_experiment_comparison()
        
        logger.info("\n=== All examples completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
