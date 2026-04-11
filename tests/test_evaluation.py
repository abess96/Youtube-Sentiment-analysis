"""
Tests for src/evaluation module:
- src/evaluation/__init__.py
- src/evaluation/bias_detection.py
- src/evaluation/drift_detection.py
- src/evaluation/evaluator.py
- src/evaluation/advanced_evaluation.py (log_to_mlflow experiment name logic)
- src/evaluation/evaluation_pipeline.py (log_to_mlflow experiment name logic)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEvaluationInit(unittest.TestCase):
    """Tests for src/evaluation/__init__.py."""

    def test_module_importable(self):
        """src.evaluation should be importable."""
        import src.evaluation
        self.assertIsNotNone(src.evaluation)

    def test_module_docstring(self):
        """Module should have a docstring describing it."""
        import src.evaluation
        self.assertIn('evaluation', src.evaluation.__doc__.lower())


class TestBiasDetectorImport(unittest.TestCase):
    """Tests that BiasDetector is importable from the new path."""

    def test_bias_detector_importable_from_new_path(self):
        """BiasDetector should be importable from src.evaluation.bias_detection."""
        from src.evaluation.bias_detection import BiasDetector
        self.assertTrue(callable(BiasDetector))

    def test_bias_detector_instantiable(self):
        from src.evaluation.bias_detection import BiasDetector
        detector = BiasDetector()
        self.assertIsNotNone(detector)
        self.assertEqual(detector.protected_attributes, [])
        self.assertEqual(detector.bias_metrics, {})


class TestBiasDetectorDetectSentimentBias(unittest.TestCase):
    """Tests for BiasDetector.detect_sentiment_bias()."""

    def setUp(self):
        from src.evaluation.bias_detection import BiasDetector
        self.detector = BiasDetector()

    def _make_data(self, n=20):
        np.random.seed(42)
        y_true = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))
        y_pred = y_true.copy()
        # Introduce a few errors
        y_pred[0] = 1
        y_pred[3] = 0
        texts = [f"word{'_extra' * (i % 5 + 1)}" for i in range(n)]
        return y_true, y_pred, texts

    def test_returns_dict_with_length_bias(self):
        y_true, y_pred, texts = self._make_data()
        result = self.detector.detect_sentiment_bias(y_true, y_pred, texts)
        self.assertIn('length_bias', result)

    def test_length_bias_keys(self):
        y_true, y_pred, texts = self._make_data()
        result = self.detector.detect_sentiment_bias(y_true, y_pred, texts)
        lb = result['length_bias']
        self.assertIn('short_accuracy', lb)
        self.assertIn('long_accuracy', lb)
        self.assertIn('disparity', lb)

    def test_length_bias_accuracy_in_range(self):
        y_true, y_pred, texts = self._make_data()
        result = self.detector.detect_sentiment_bias(y_true, y_pred, texts)
        lb = result['length_bias']
        self.assertGreaterEqual(lb['short_accuracy'], 0.0)
        self.assertLessEqual(lb['short_accuracy'], 1.0)
        self.assertGreaterEqual(lb['long_accuracy'], 0.0)
        self.assertLessEqual(lb['long_accuracy'], 1.0)

    def test_disparity_non_negative(self):
        y_true, y_pred, texts = self._make_data()
        result = self.detector.detect_sentiment_bias(y_true, y_pred, texts)
        self.assertGreaterEqual(result['length_bias']['disparity'], 0.0)

    def test_class_accuracy_keys_present(self):
        y_true, y_pred, texts = self._make_data()
        result = self.detector.detect_sentiment_bias(y_true, y_pred, texts)
        # Should have a key for each unique class in y_true
        for cls in np.unique(y_true):
            self.assertIn(f'class_{cls}_accuracy', result)

    def test_perfect_predictions_zero_disparity(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        texts = ['short text', 'a very very long text with many words', 'hi',
                 'another very long text with lots of tokens in it', 'ok', 'last one here more']
        result = self.detector.detect_sentiment_bias(y, y, texts)
        # With perfect predictions disparity should be 0
        self.assertAlmostEqual(result['length_bias']['disparity'], 0.0)


class TestBiasDetectorDemographicParity(unittest.TestCase):
    """Tests for BiasDetector.calculate_demographic_parity()."""

    def setUp(self):
        from src.evaluation.bias_detection import BiasDetector
        self.detector = BiasDetector()

    def test_returns_float(self):
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        protected = np.array([0, 0, 0, 1, 1, 1])
        result = self.detector.calculate_demographic_parity(y_pred, protected)
        self.assertIsInstance(result, float)

    def test_single_group_returns_zero(self):
        y_pred = np.array([1, 0, 1, 0])
        protected = np.array([0, 0, 0, 0])
        result = self.detector.calculate_demographic_parity(y_pred, protected)
        self.assertEqual(result, 0.0)

    def test_equal_positive_rates_returns_zero(self):
        y_pred = np.array([1, 0, 1, 0])
        protected = np.array([0, 0, 1, 1])
        result = self.detector.calculate_demographic_parity(y_pred, protected)
        self.assertAlmostEqual(result, 0.0)

    def test_maximum_disparity(self):
        # Group 0: all predicted 1, Group 1: all predicted 0
        y_pred = np.array([1, 1, 0, 0])
        protected = np.array([0, 0, 1, 1])
        result = self.detector.calculate_demographic_parity(y_pred, protected)
        self.assertAlmostEqual(result, 1.0)

    def test_partial_disparity(self):
        # Group 0: 100% positive, Group 1: 50% positive -> disparity = 0.5
        y_pred = np.array([1, 1, 1, 0])
        protected = np.array([0, 0, 1, 1])
        result = self.detector.calculate_demographic_parity(y_pred, protected)
        self.assertAlmostEqual(result, 0.5)


class TestDriftDetectorImport(unittest.TestCase):
    """Tests that DriftDetector is importable from the new path."""

    def test_drift_detector_importable_from_new_path(self):
        from src.evaluation.drift_detection import DriftDetector
        self.assertTrue(callable(DriftDetector))

    def test_drift_detector_instantiable(self):
        from src.evaluation.drift_detection import DriftDetector
        detector = DriftDetector()
        self.assertIsNotNone(detector)
        self.assertEqual(detector.threshold, 0.05)
        self.assertEqual(detector.drift_history, [])


class TestDriftDetectorKS(unittest.TestCase):
    """Tests for DriftDetector.detect_data_drift_ks()."""

    def setUp(self):
        from src.evaluation.drift_detection import DriftDetector
        self.detector = DriftDetector(threshold=0.05)

    def _make_identical_data(self, n=100, features=5):
        np.random.seed(0)
        data = np.random.randn(n, features)
        return data, data.copy()

    def _make_drifted_data(self, n=100, features=5):
        np.random.seed(0)
        ref = np.random.randn(n, features)
        # Significant shift in distribution
        curr = np.random.randn(n, features) + 10.0
        return ref, curr

    def test_returns_expected_keys(self):
        ref, curr = self._make_identical_data()
        result = self.detector.detect_data_drift_ks(ref, curr)
        for key in ['overall_drift_detected', 'drift_score', 'n_drifted_features',
                    'total_features', 'feature_drifts']:
            self.assertIn(key, result)

    def test_no_drift_identical_data(self):
        """KS test should not detect drift in identical data."""
        ref, _ = self._make_identical_data()
        result = self.detector.detect_data_drift_ks(ref, ref)
        self.assertFalse(result['overall_drift_detected'])
        self.assertEqual(result['n_drifted_features'], 0)

    def test_drift_detected_with_shifted_data(self):
        """KS test should detect drift when distribution is strongly shifted."""
        ref, curr = self._make_drifted_data()
        result = self.detector.detect_data_drift_ks(ref, curr)
        self.assertTrue(result['overall_drift_detected'])
        self.assertGreater(result['n_drifted_features'], 0)

    def test_total_features_matches_input(self):
        n_features = 7
        ref = np.random.randn(50, n_features)
        curr = np.random.randn(50, n_features)
        result = self.detector.detect_data_drift_ks(ref, curr)
        self.assertEqual(result['total_features'], n_features)

    def test_raises_on_mismatched_features(self):
        ref = np.random.randn(50, 5)
        curr = np.random.randn(50, 6)
        with self.assertRaises(ValueError):
            self.detector.detect_data_drift_ks(ref, curr)

    def test_drift_score_is_float(self):
        ref, curr = self._make_identical_data()
        result = self.detector.detect_data_drift_ks(ref, curr)
        self.assertIsInstance(result['drift_score'], float)

    def test_feature_drifts_list_max_10(self):
        """feature_drifts should be capped at 10 for brevity."""
        ref = np.random.randn(100, 15)
        curr = np.random.randn(100, 15)
        result = self.detector.detect_data_drift_ks(ref, curr)
        self.assertLessEqual(len(result['feature_drifts']), 10)

    def test_custom_threshold_respected(self):
        """A very strict threshold (0) should flag all features as drifted."""
        from src.evaluation.drift_detection import DriftDetector
        strict_detector = DriftDetector(threshold=0.0)
        ref = np.random.randn(100, 3)
        curr = np.random.randn(100, 3)
        result = strict_detector.detect_data_drift_ks(ref, curr)
        self.assertTrue(result['overall_drift_detected'])


class TestEvaluatorImport(unittest.TestCase):
    """Tests that ComprehensiveEvaluator is importable from the new path."""

    def test_evaluator_importable_from_new_path(self):
        from src.evaluation.evaluator import ComprehensiveEvaluator
        self.assertTrue(callable(ComprehensiveEvaluator))

    def test_evaluator_instantiable(self):
        from src.evaluation.evaluator import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        self.assertIsNotNone(evaluator)


class TestAdvancedEvaluationExperimentName(unittest.TestCase):
    """Tests for the structured experiment name logic in advanced_evaluation.log_to_mlflow()."""

    @patch('src.evaluation.advanced_evaluation.MLflowTracker')
    @patch('src.evaluation.advanced_evaluation.setup_mlflow')
    def test_uses_structured_experiment_name_from_params(self, mock_setup, mock_tracker_cls):
        """log_to_mlflow should use params['mlflow']['experiments']['model_evaluation']."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        from src.evaluation.advanced_evaluation import log_to_mlflow

        params = {
            'mlflow': {
                'experiments': {
                    'model_evaluation': 'custom_04_eval'
                }
            }
        }
        log_to_mlflow({}, {}, 'model_a', params)

        mock_tracker_cls.assert_called_once_with(experiment_name='custom_04_eval')

    @patch('src.evaluation.advanced_evaluation.MLflowTracker')
    @patch('src.evaluation.advanced_evaluation.setup_mlflow')
    def test_falls_back_to_default_experiment_name(self, mock_setup, mock_tracker_cls):
        """log_to_mlflow should use '04_Model_Evaluation' when experiments key missing."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        from src.evaluation.advanced_evaluation import log_to_mlflow

        # No 'experiments' key under 'mlflow'
        log_to_mlflow({}, {}, 'model_a', {'mlflow': {}})
        mock_tracker_cls.assert_called_once_with(experiment_name='04_Model_Evaluation')

    @patch('src.evaluation.advanced_evaluation.MLflowTracker')
    @patch('src.evaluation.advanced_evaluation.setup_mlflow')
    def test_falls_back_when_no_mlflow_key(self, mock_setup, mock_tracker_cls):
        """log_to_mlflow should use '04_Model_Evaluation' when mlflow key is absent."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        from src.evaluation.advanced_evaluation import log_to_mlflow

        log_to_mlflow({}, {}, 'model_a', {})
        mock_tracker_cls.assert_called_once_with(experiment_name='04_Model_Evaluation')

    @patch('src.evaluation.advanced_evaluation.MLflowTracker')
    @patch('src.evaluation.advanced_evaluation.setup_mlflow')
    def test_setup_mlflow_called(self, mock_setup, mock_tracker_cls):
        """log_to_mlflow should call setup_mlflow() for MLflow configuration."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        from src.evaluation.advanced_evaluation import log_to_mlflow
        log_to_mlflow({}, {}, 'best_model', {})
        mock_setup.assert_called_once()


class TestEvaluationPipelineExperimentName(unittest.TestCase):
    """Tests for the structured experiment name logic in EvaluationPipeline.log_to_mlflow()."""

    def _make_pipeline(self, params):
        """Create an EvaluationPipeline with mocked dependencies."""
        from src.evaluation.evaluation_pipeline import EvaluationPipeline
        with patch.object(EvaluationPipeline, '__init__', lambda self, params_path=None: None):
            pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
            pipeline.params = params
            pipeline.mlflow_tracker = None
            return pipeline

    @patch('src.evaluation.evaluation_pipeline.MLflowTracker')
    @patch('src.evaluation.evaluation_pipeline.setup_mlflow')
    def test_uses_structured_experiment_name(self, mock_setup, mock_tracker_cls):
        """EvaluationPipeline.log_to_mlflow should use structured experiment name."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        params = {
            'mlflow': {
                'experiments': {
                    'model_evaluation': '04_Model_Evaluation'
                }
            }
        }
        pipeline = self._make_pipeline(params)

        # Provide minimal results dict to avoid KeyError
        results = {
            'evaluation': {
                'metrics': {
                    'accuracy': 0.9,
                    'balanced_accuracy': 0.88,
                    'matthews_corrcoef': 0.85,
                    'cohen_kappa': 0.87,
                    'weighted': {'f1_score': 0.9, 'precision': 0.9, 'recall': 0.9}
                },
                'cross_validation': {}
            }
        }

        pipeline.log_to_mlflow(results)
        mock_tracker_cls.assert_called_once_with(experiment_name='04_Model_Evaluation')

    @patch('src.evaluation.evaluation_pipeline.MLflowTracker')
    @patch('src.evaluation.evaluation_pipeline.setup_mlflow')
    def test_falls_back_to_default(self, mock_setup, mock_tracker_cls):
        """EvaluationPipeline.log_to_mlflow falls back to '04_Model_Evaluation'."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_tracker.start_run.return_value = mock_run

        pipeline = self._make_pipeline({'mlflow': {}})

        results = {
            'evaluation': {
                'metrics': {
                    'accuracy': 0.9,
                    'balanced_accuracy': 0.88,
                    'matthews_corrcoef': 0.85,
                    'cohen_kappa': 0.87,
                    'weighted': {'f1_score': 0.9, 'precision': 0.9, 'recall': 0.9}
                },
                'cross_validation': {}
            }
        }

        pipeline.log_to_mlflow(results)
        mock_tracker_cls.assert_called_once_with(experiment_name='04_Model_Evaluation')


if __name__ == '__main__':
    unittest.main()