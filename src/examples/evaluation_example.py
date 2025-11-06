"""
Example usage of the comprehensive evaluation and monitoring system.
Demonstrates Tasks 4.1, 4.2, and 4.3
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.evaluation.evaluator import ComprehensiveEvaluator
from src.evaluation.bias_detection import BiasDetector
from src.evaluation.drift_detection import DriftDetector


def example_comprehensive_evaluation():
    """Example: Comprehensive model evaluation with cross-validation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Comprehensive Evaluation")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                               n_informative=15, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:800], y[:800])
    
    # Evaluate
    evaluator = ComprehensiveEvaluator(n_splits=5)
    
    # Basic evaluation
    metrics = evaluator.evaluate(model, X[800:], y[800:])
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
    print(f"F1-Score (weighted): {metrics['weighted']['f1_score']:.4f}")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = evaluator.cross_validate(model, X[800:], y[800:])
    print(f"CV Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    print(f"95% CI: [{cv_results['accuracy']['ci_95'][0]:.4f}, {cv_results['accuracy']['ci_95'][1]:.4f}]")
    
    # Generate report
    report = evaluator.generate_report(metrics)
    print("\n" + report)


def example_bias_detection():
    """Example: Bias detection and fairness evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Bias Detection")
    print("="*60)
    
    # Simulate sentiment data
    np.random.seed(42)
    n_samples = 500
    
    # Create biased predictions (short texts get lower accuracy)
    texts = []
    y_true = []
    y_pred = []
    
    for i in range(n_samples):
        # Generate text of varying lengths
        length = np.random.choice([5, 15, 30])
        text = " ".join(["word"] * length)
        texts.append(text)
        
        # True label
        true_label = np.random.randint(0, 3)
        y_true.append(true_label)
        
        # Biased prediction (shorter texts more likely to be misclassified)
        if length < 10 and np.random.random() < 0.3:
            pred_label = (true_label + 1) % 3  # Wrong prediction
        else:
            pred_label = true_label
        y_pred.append(pred_label)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Detect bias
    bias_detector = BiasDetector()
    fairness_metrics = bias_detector.evaluate_fairness(y_true, y_pred, texts)
    
    print(f"\nFairness Score: {fairness_metrics['overall_fairness_score']:.4f}")
    print(f"Length Bias Disparity: {fairness_metrics['sentiment_bias']['length_bias']['disparity']:.4f}")
    print(f"Short Text Accuracy: {fairness_metrics['sentiment_bias']['length_bias']['short_accuracy']:.4f}")
    print(f"Long Text Accuracy: {fairness_metrics['sentiment_bias']['length_bias']['long_accuracy']:.4f}")
    
    # Get recommendations
    recommendations = bias_detector.generate_mitigation_recommendations(fairness_metrics)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")


def example_drift_detection():
    """Example: Data and model drift detection."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Drift Detection")
    print("="*60)
    
    # Generate reference data
    np.random.seed(42)
    X_ref = np.random.randn(500, 10)
    y_ref = np.random.randint(0, 3, 500)
    y_pred_ref = y_ref.copy()
    
    # Generate current data with drift
    X_curr = np.random.randn(500, 10) + 0.5  # Shifted distribution
    y_curr = np.random.randint(0, 3, 500)
    y_pred_curr = y_curr.copy()
    # Introduce some prediction errors
    error_idx = np.random.choice(500, 50, replace=False)
    y_pred_curr[error_idx] = (y_pred_curr[error_idx] + 1) % 3
    
    # Initialize drift detector
    drift_detector = DriftDetector(
        reference_data=X_ref,
        reference_predictions=y_pred_ref,
        threshold=0.05
    )
    drift_detector.y_true_ref = y_ref
    drift_detector.y_pred_ref = y_pred_ref
    
    # Monitor current data
    monitoring_report = drift_detector.monitor(X_curr, y_pred_curr, y_curr)
    
    print(f"\nOverall Status: {monitoring_report['overall_status']}")
    print(f"Data Drift Detected: {monitoring_report['data_drift_ks']['overall_drift_detected']}")
    print(f"Drift Score: {monitoring_report['data_drift_ks']['drift_score']:.4f}")
    print(f"PSI Score: {monitoring_report['data_drift_psi']['average_psi']:.4f}")
    print(f"PSI Level: {monitoring_report['data_drift_psi']['drift_level']}")
    
    if 'performance_drift' in monitoring_report:
        perf = monitoring_report['performance_drift']
        print(f"\nPerformance Drift:")
        print(f"  Reference Accuracy: {perf['reference_accuracy']:.4f}")
        print(f"  Current Accuracy: {perf['current_accuracy']:.4f}")
        print(f"  Relative Drop: {perf['relative_drop']*100:.1f}%")
        print(f"  Severity: {perf['severity']}")
    
    # Generate alert
    alert = drift_detector.generate_alert(monitoring_report)
    print(f"\nAlert Status: {alert['status']}")
    print(f"Requires Action: {alert['requires_action']}")
    print("Messages:")
    for msg in alert['messages']:
        print(f"  - {msg}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION & MONITORING EXAMPLES")
    print("Task 4: Evaluation and Monitoring System")
    print("="*60)
    
    example_comprehensive_evaluation()
    example_bias_detection()
    example_drift_detection()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
