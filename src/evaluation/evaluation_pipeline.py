"""
Comprehensive evaluation and monitoring pipeline.
Integrates Tasks 4.1, 4.2, and 4.3
"""

import yaml
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any

from src.evaluation.evaluator import ComprehensiveEvaluator
from src.evaluation.bias_detection import BiasDetector
from src.evaluation.drift_detection import DriftDetector
from src.mlops.mlflow_integration import MLflowTracker
from src.config.mlflow_config import setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Comprehensive evaluation pipeline with monitoring and bias detection."""
    
    def __init__(self, params_path: str = 'params.yaml'):
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        self.evaluator = ComprehensiveEvaluator(n_splits=5)
        self.bias_detector = BiasDetector()
        self.drift_detector = None
        self.mlflow_tracker = None
    
    def load_data(self) -> tuple:
        """Load test data and features."""
        logger.info("Loading test data...")
        
        # Load processed test data
        test_data = pd.read_csv('data/interim/test_processed.csv')
        test_data.fillna('', inplace=True)
        
        # Load features
        with open('data/features/selected_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
        
        X_test = features_data['test_features']
        y_test = features_data['test_labels']
        texts = test_data['clean_comment'].values
        
        logger.info(f"Loaded {len(y_test)} test samples")
        return X_test, y_test, texts
    
    def load_model(self, model_path: str):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}")
        import joblib
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        return model
    
    def run_comprehensive_evaluation(self, model, X_test, y_test) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Running comprehensive evaluation...")
        
        # Basic evaluation
        metrics = self.evaluator.evaluate(model, X_test, y_test)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self.evaluator.cross_validate(model, X_test, y_test)
        
        # Generate report
        report_text = self.evaluator.generate_report(metrics)
        
        return {
            'metrics': metrics,
            'cross_validation': cv_results,
            'report': report_text
        }
    
    def run_bias_detection(self, y_test, y_pred, texts) -> Dict[str, Any]:
        """Run bias detection and fairness evaluation."""
        logger.info("Running bias detection...")
        
        fairness_metrics = self.bias_detector.evaluate_fairness(y_test, y_pred, texts)
        recommendations = self.bias_detector.generate_mitigation_recommendations(fairness_metrics)
        
        return {
            'fairness_metrics': fairness_metrics,
            'recommendations': recommendations
        }
    
    def run_drift_monitoring(self, X_test, y_test, y_pred) -> Dict[str, Any]:
        """Run drift monitoring."""
        logger.info("Running drift monitoring...")
        
        # Split data for reference and current
        split_idx = len(X_test) // 2
        X_ref, X_curr = X_test[:split_idx], X_test[split_idx:]
        y_ref, y_curr = y_test[:split_idx], y_test[split_idx:]
        y_pred_ref, y_pred_curr = y_pred[:split_idx], y_pred[split_idx:]
        
        # Initialize drift detector with reference data
        self.drift_detector = DriftDetector(
            reference_data=X_ref,
            reference_predictions=y_pred_ref,
            threshold=0.05
        )
        self.drift_detector.y_true_ref = y_ref
        self.drift_detector.y_pred_ref = y_pred_ref
        
        # Monitor current data
        monitoring_report = self.drift_detector.monitor(
            X_curr, y_pred_curr, y_curr
        )
        
        # Generate alert
        alert = self.drift_detector.generate_alert(monitoring_report)
        
        return {
            'monitoring_report': monitoring_report,
            'alert': alert
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'models/evaluation'):
        """Save evaluation results."""
        logger.info("Saving evaluation results...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive metrics
        with open(output_path / 'comprehensive_metrics.json', 'w') as f:
            json.dump(results['evaluation'], f, indent=2)
        
        # Save bias report
        with open(output_path / 'bias_report.json', 'w') as f:
            json.dump(results['bias'], f, indent=2)
        
        # Save drift report
        with open(output_path / 'drift_report.json', 'w') as f:
            json.dump(results['drift'], f, indent=2)
        
        # Save drift monitoring history
        with open(output_path / 'drift_monitoring_history.json', 'w') as f:
            json.dump({'history': [results['drift']['monitoring_report']]}, f, indent=2)
        
        # Save text report
        with open(output_path / 'evaluation_report.txt', 'w') as f:
            f.write(results['evaluation']['report'])
            f.write("\n\n" + "="*60 + "\n")
            f.write("BIAS DETECTION RECOMMENDATIONS\n")
            f.write("="*60 + "\n")
            for rec in results['bias']['recommendations']:
                f.write(f"- {rec}\n")
        
        logger.info(f"Results saved to {output_path}")
    
    def log_to_mlflow(self, results: Dict[str, Any]):
        """Log results to MLflow."""
        logger.info("Logging to MLflow...")
        
        setup_mlflow()
        mlflow_params = self.params.get('mlflow', {})
        experiment_name = mlflow_params.get('experiments', {}).get('model_evaluation', '04_Model_Evaluation')
        self.mlflow_tracker = MLflowTracker(
            experiment_name=experiment_name
        )
        
        with self.mlflow_tracker.start_run(run_name="comprehensive_evaluation") as run:
            # Log evaluation metrics
            eval_metrics = results['evaluation']['metrics']
            self.mlflow_tracker.log_metrics({
                'accuracy': eval_metrics['accuracy'],
                'balanced_accuracy': eval_metrics['balanced_accuracy'],
                'matthews_corrcoef': eval_metrics['matthews_corrcoef'],
                'cohen_kappa': eval_metrics['cohen_kappa'],
                'weighted_f1': eval_metrics['weighted']['f1_score'],
                'weighted_precision': eval_metrics['weighted']['precision'],
                'weighted_recall': eval_metrics['weighted']['recall']
            })
            
            # Log CV results
            cv_results = results['evaluation']['cross_validation']
            for metric, values in cv_results.items():
                self.mlflow_tracker.log_metrics({
                    f'cv_{metric}_mean': values['mean'],
                    f'cv_{metric}_std': values['std']
                })
            
            # Log bias metrics
            fairness = results['bias']['fairness_metrics']
            self.mlflow_tracker.log_metrics({
                'fairness_score': fairness['overall_fairness_score'],
                'length_bias_disparity': fairness['sentiment_bias']['length_bias']['disparity']
            })
            
            # Log drift metrics
            drift = results['drift']['monitoring_report']
            self.mlflow_tracker.log_metrics({
                'drift_score': drift['data_drift_ks']['drift_score'],
                'psi_score': drift['data_drift_psi']['average_psi']
            })
            
            # Log tags
            self.mlflow_tracker.set_tags({
                'evaluation_type': 'comprehensive',
                'drift_status': drift['overall_status'],
                'bias_detected': str(fairness['overall_fairness_score'] < 0.9)
            })
            
            # Log artifacts
            self.mlflow_tracker.log_dict(results['evaluation']['metrics'], 'metrics.json')
            self.mlflow_tracker.log_dict(results['bias'], 'bias_report.json')
            self.mlflow_tracker.log_dict(results['drift'], 'drift_report.json')
            
            logger.info(f"Results logged to MLflow run: {run.info.run_id}")
    
    def run(self, model_path: str = None):
        """Run complete evaluation pipeline."""
        logger.info("Starting comprehensive evaluation pipeline...")
        
        # Load data
        X_test, y_test, texts = self.load_data()
        
        # Load model
        if model_path is None:
            from pathlib import Path
            if Path('lgbm_model.pkl').exists():
                model_path = 'lgbm_model.pkl'
            else:
                model_path = 'models/trained_models/lightgbm_model.pkl'
        model = self.load_model(model_path)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Run evaluations
        results = {
            'evaluation': self.run_comprehensive_evaluation(model, X_test, y_test),
            'bias': self.run_bias_detection(y_test, y_pred, texts),
            'drift': self.run_drift_monitoring(X_test, y_test, y_pred)
        }
        
        # Save results
        self.save_results(results)
        
        # Log to MLflow
        self.log_to_mlflow(results)
        
        logger.info("Evaluation pipeline completed successfully!")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        metrics = results['evaluation']['metrics']
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"F1-Score (weighted): {metrics['weighted']['f1_score']:.4f}")
        
        fairness = results['bias']['fairness_metrics']
        print(f"\nFairness Score: {fairness['overall_fairness_score']:.4f}")
        
        drift = results['drift']['monitoring_report']
        print(f"\nDrift Status: {drift['overall_status']}")
        
        print("\nRecommendations:")
        for rec in results['bias']['recommendations']:
            print(f"  - {rec}")
        
        print("="*60 + "\n")


def main():
    """Main execution function."""
    pipeline = EvaluationPipeline()
    results = pipeline.run()
    return results


if __name__ == "__main__":
    main()
