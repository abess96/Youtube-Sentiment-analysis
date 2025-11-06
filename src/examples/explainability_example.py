"""
Example script demonstrating explainability features.
Shows how to use LIME, SHAP, uncertainty quantification, and model debugging.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.explainability.explainability_engine import ExplainabilityEngine, TextExplainer
from src.explainability.uncertainty_quantification import UncertaintyQuantifier, EnsembleUncertainty
from src.explainability.model_debugger import ModelDebugger
from src.models.base.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load sample data for demonstration."""
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'interim'
    
    try:
        train_df = pd.read_csv(data_path / 'train_processed.csv')
        test_df = pd.read_csv(data_path / 'test_processed.csv')
        
        # Sample for faster demonstration
        train_sample = train_df.sample(n=min(1000, len(train_df)), random_state=42)
        test_sample = test_df.sample(n=min(200, len(test_df)), random_state=42)
        
        return train_sample, test_sample
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Generate synthetic data for demonstration
        logger.info("Generating synthetic data for demonstration")
        
        np.random.seed(42)
        n_samples = 500
        n_features = 100
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 3, n_samples)
        
        X_test = np.random.randn(100, n_features)
        y_test = np.random.randint(0, 3, 100)
        
        return (X_train, y_train), (X_test, y_test)


def demonstrate_lime_shap(model, X_test, y_test, feature_names):
    """Demonstrate LIME and SHAP explanations."""
    logger.info("\n=== LIME and SHAP Explanations ===")
    
    # Initialize explainability engine
    explainer = ExplainabilityEngine(
        model=model,
        feature_names=feature_names,
        class_names=['negative', 'neutral', 'positive']
    )
    
    # LIME explanation for a single instance
    logger.info("\n1. Generating LIME explanation...")
    lime_exp = explainer.explain_with_lime(X_test, instance_idx=0, num_features=10)
    
    logger.info(f"Prediction: {lime_exp.prediction} (Confidence: {lime_exp.confidence:.2%})")
    logger.info("Top features:")
    for feature, importance in list(lime_exp.feature_importance.items())[:5]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Visualize LIME explanation
    output_dir = Path(__file__).parent.parent.parent.parent / 'models' / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    explainer.visualize_explanation(lime_exp, save_path=str(output_dir / 'lime_explanation.png'))
    
    # SHAP explanation
    logger.info("\n2. Generating SHAP explanation...")
    try:
        shap_exp = explainer.explain_with_shap(X_test, background_samples=50, instance_idx=0)
        
        logger.info(f"Prediction: {shap_exp.prediction} (Confidence: {shap_exp.confidence:.2%})")
        logger.info("Top SHAP features:")
        for feature, importance in list(shap_exp.feature_importance.items())[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        # Visualize SHAP explanation
        explainer.visualize_explanation(shap_exp, save_path=str(output_dir / 'shap_explanation.png'))
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
    
    # Global feature importance
    logger.info("\n3. Calculating global feature importance...")
    global_importance = explainer.get_global_feature_importance(X_test, method='permutation', top_k=10)
    
    logger.info("Top 10 most important features globally:")
    for feature, importance in global_importance.items():
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Generate explanation report
    logger.info("\n4. Generating explanation report...")
    report = explainer.generate_explanation_report(X_test, y_test, sample_size=10)
    
    logger.info(f"Generated {report['num_explanations']} explanations")
    logger.info(f"Top global features: {report['top_features'][:5]}")


def demonstrate_uncertainty(model, X_test):
    """Demonstrate uncertainty quantification."""
    logger.info("\n=== Uncertainty Quantification ===")
    
    # Initialize uncertainty quantifier
    quantifier = UncertaintyQuantifier(
        model=model,
        confidence_level=0.95,
        reliability_threshold=0.7
    )
    
    # Bootstrap uncertainty estimation
    logger.info("\n1. Bootstrap uncertainty estimation...")
    uncertainty_results = quantifier.predict_with_uncertainty(
        X_test[:50],  # Use subset for faster computation
        method='bootstrap',
        n_iterations=30,
        sample_ratio=0.8
    )
    
    # Display results for first few predictions
    logger.info("\nUncertainty results for first 5 predictions:")
    for i, result in enumerate(uncertainty_results[:5]):
        logger.info(f"\nPrediction {i}:")
        logger.info(f"  Class: {result.prediction}")
        logger.info(f"  Confidence: {result.mean_confidence:.2%} ± {result.std_confidence:.2%}")
        logger.info(f"  95% CI: [{result.confidence_interval[0]:.2%}, {result.confidence_interval[1]:.2%}]")
        logger.info(f"  Uncertainty score: {result.uncertainty_score:.4f}")
        logger.info(f"  Reliable: {result.is_reliable}")
    
    # Get uncertainty statistics
    logger.info("\n2. Uncertainty statistics...")
    stats = quantifier.get_uncertainty_statistics(uncertainty_results)
    
    logger.info(f"Mean uncertainty: {stats['mean_uncertainty']:.4f}")
    logger.info(f"Mean confidence: {stats['mean_confidence']:.2%}")
    logger.info(f"Reliable predictions: {stats['reliable_predictions']}/{stats['total_predictions']} ({stats['reliability_rate']:.2%})")
    
    # Filter by uncertainty
    logger.info("\n3. Filtering by uncertainty...")
    filtered_X, filtered_results = quantifier.filter_by_uncertainty(
        X_test[:50],
        uncertainty_results,
        max_uncertainty=0.3
    )
    
    logger.info(f"Kept {len(filtered_results)}/{len(uncertainty_results)} predictions after filtering")


def demonstrate_debugging(model, X_test, y_test, feature_names):
    """Demonstrate model debugging tools."""
    logger.info("\n=== Model Debugging ===")
    
    # Initialize debugger
    debugger = ModelDebugger(
        model=model,
        class_names=['negative', 'neutral', 'positive']
    )
    
    # Analyze misclassifications
    logger.info("\n1. Analyzing misclassifications...")
    misclass_analysis = debugger.analyze_misclassifications(
        X_test, y_test, feature_names=feature_names, top_k=5
    )
    
    logger.info(f"Total misclassified: {misclass_analysis.total_misclassified}")
    logger.info(f"Misclassification rate: {misclass_analysis.misclassification_rate:.2%}")
    
    logger.info("\nConfusion patterns:")
    for pattern, count in list(misclass_analysis.confusion_patterns.items())[:5]:
        logger.info(f"  {pattern}: {count}")
    
    logger.info("\nClass-specific errors:")
    for class_name, errors in misclass_analysis.class_specific_errors.items():
        logger.info(f"  {class_name}: {errors['misclassified']}/{errors['total_samples']} ({errors['error_rate']:.2%})")
    
    # Visualize confusion patterns
    output_dir = Path(__file__).parent.parent.parent.parent / 'models' / 'evaluation'
    debugger.visualize_confusion_patterns(
        misclass_analysis,
        save_path=str(output_dir / 'confusion_patterns.png')
    )
    
    # Identify failure modes
    logger.info("\n2. Identifying failure modes...")
    failure_modes = debugger.identify_failure_modes(X_test, y_test, feature_names)
    
    logger.info(f"Low-confidence errors: {len(failure_modes['low_confidence_errors'])}")
    logger.info(f"High-confidence errors: {len(failure_modes['high_confidence_errors'])}")
    logger.info(f"Boundary cases: {len(failure_modes['boundary_cases'])}")
    
    # Generate debug report
    logger.info("\n3. Generating debug report...")
    debug_report = debugger.generate_debug_report(X_test, y_test, feature_names)
    
    logger.info("\nOverall metrics:")
    for metric, value in debug_report['overall_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nRecommendations:")
    for i, rec in enumerate(debug_report['recommendations'], 1):
        logger.info(f"  {i}. {rec}")
    
    # Save debug report
    import json
    report_path = output_dir / 'debug_report.json'
    with open(report_path, 'w') as f:
        json.dump(debug_report, f, indent=2)
    logger.info(f"\nSaved debug report to {report_path}")


def main():
    """Main demonstration function."""
    logger.info("Starting Explainability Features Demonstration")
    
    # Load data
    logger.info("\n=== Loading Data ===")
    data = load_data()
    
    if isinstance(data[0], tuple):
        # Synthetic data
        X_train, y_train = data[0]
        X_test, y_test = data[1]
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    else:
        # Real data
        train_df, test_df = data
        
        # For demonstration, use simple features
        feature_cols = [col for col in train_df.columns if col not in ['sentiment', 'text', 'Unnamed: 0']]
        
        if len(feature_cols) == 0:
            logger.error("No feature columns found. Using synthetic data.")
            return main()  # Retry with synthetic data
        
        X_train = train_df[feature_cols].values
        y_train = train_df['sentiment'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['sentiment'].values
        feature_names = feature_cols
    
    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Test data: {X_test.shape}")
    
    # Train a simple model
    logger.info("\n=== Training Model ===")
    model_factory = ModelFactory()
    model = model_factory.create_model('logistic_regression', max_iter=1000, random_state=42)
    
    logger.info("Training logistic regression model...")
    model.train(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Demonstrate explainability features
    demonstrate_lime_shap(model.model, X_test, y_test, feature_names)
    demonstrate_uncertainty(model.model, X_test)
    demonstrate_debugging(model.model, X_test, y_test, feature_names)
    
    logger.info("\n=== Demonstration Complete ===")
    logger.info("Check the 'models/evaluation' directory for visualizations and reports")


if __name__ == "__main__":
    main()
