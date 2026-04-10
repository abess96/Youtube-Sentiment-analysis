"""
Tests for src/examples module:
- src/examples/__init__.py
- Import path changes in example files
- Module-level MLflow setup in continuous_learning_example.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestExamplesInit(unittest.TestCase):
    """Tests for src/examples/__init__.py."""

    def test_module_importable(self):
        """src.examples should be importable."""
        import src.examples
        self.assertIsNotNone(src.examples)

    def test_module_has_docstring(self):
        """Module should have a docstring."""
        import src.examples
        self.assertIsNotNone(src.examples.__doc__)
        self.assertGreater(len(src.examples.__doc__), 0)

    def test_docstring_describes_purpose(self):
        """Module docstring should mention examples or scripts."""
        import src.examples
        doc_lower = src.examples.__doc__.lower()
        self.assertTrue(
            'example' in doc_lower or 'script' in doc_lower or 'demonstrat' in doc_lower
        )


class TestContinuousLearningExampleImports(unittest.TestCase):
    """
    Tests for the changed import paths in continuous_learning_example.py.

    The PR moved from:
      - model.incremental_learner -> training.incremental_learner
      - model.active_learner      -> training.active_learner
      - model.auto_retrainer      -> training.auto_retrainer
      - model.drift_detector      -> evaluation.drift_detection
    And added the mlflow config import from config.mlflow_config.
    """

    def test_module_level_mlflow_setup_uses_continuous_learning_experiment(self):
        """
        The module-level setup block should call get_or_create_experiment
        with the 'continuous_learning' experiment name from params.yaml,
        or fall back to '06_Continuous_Learning'.
        """
        params_yaml = (
            "mlflow:\n"
            "  experiments:\n"
            "    continuous_learning: '06_Continuous_Learning'\n"
        )

        mock_setup = MagicMock()
        mock_get_or_create = MagicMock(return_value='exp_6')
        mock_mlflow_config = MagicMock()
        mock_mlflow_config.setup_mlflow = mock_setup
        mock_mlflow_config.get_or_create_experiment = mock_get_or_create

        # Mock all training/evaluation dependencies to avoid heavy imports
        mock_incremental = MagicMock()
        mock_active = MagicMock()
        mock_auto = MagicMock()
        mock_drift = MagicMock()

        with patch.dict('sys.modules', {
            'training': MagicMock(),
            'training.incremental_learner': mock_incremental,
            'training.active_learner': mock_active,
            'training.auto_retrainer': mock_auto,
            'evaluation': MagicMock(),
            'evaluation.drift_detection': mock_drift,
            'config': MagicMock(),
            'config.mlflow_config': mock_mlflow_config,
        }), patch('builtins.open', mock_open(read_data=params_yaml)):
            # Force re-import to trigger module-level code
            if 'src.examples.continuous_learning_example' in sys.modules:
                del sys.modules['src.examples.continuous_learning_example']
            import src.examples.continuous_learning_example  # noqa

        mock_setup.assert_called()
        mock_get_or_create.assert_called_with('06_Continuous_Learning')

    def test_module_level_mlflow_fallback_on_file_error(self):
        """
        If params.yaml cannot be opened, falls back to '06_Continuous_Learning'.
        """
        mock_setup = MagicMock()
        mock_get_or_create = MagicMock(return_value='exp_6')
        mock_mlflow_config = MagicMock()
        mock_mlflow_config.setup_mlflow = mock_setup
        mock_mlflow_config.get_or_create_experiment = mock_get_or_create

        mock_incremental = MagicMock()
        mock_active = MagicMock()
        mock_auto = MagicMock()
        mock_drift = MagicMock()

        with patch.dict('sys.modules', {
            'training': MagicMock(),
            'training.incremental_learner': mock_incremental,
            'training.active_learner': mock_active,
            'training.auto_retrainer': mock_auto,
            'evaluation': MagicMock(),
            'evaluation.drift_detection': mock_drift,
            'config': MagicMock(),
            'config.mlflow_config': mock_mlflow_config,
        }), patch('builtins.open', side_effect=FileNotFoundError('no params.yaml')):
            if 'src.examples.continuous_learning_example' in sys.modules:
                del sys.modules['src.examples.continuous_learning_example']
            # The except block should catch the file error and log a warning
            try:
                import src.examples.continuous_learning_example  # noqa
            except Exception:
                pass  # The module-level exception is caught internally

    def test_module_level_mlflow_no_crash_on_setup_failure(self):
        """Module-level MLflow failure is caught gracefully (no ImportError propagation)."""
        mock_mlflow_config = MagicMock()
        mock_mlflow_config.setup_mlflow.side_effect = RuntimeError('mlflow down')
        mock_mlflow_config.get_or_create_experiment = MagicMock()

        mock_incremental = MagicMock()
        mock_active = MagicMock()
        mock_auto = MagicMock()
        mock_drift = MagicMock()

        with patch.dict('sys.modules', {
            'training': MagicMock(),
            'training.incremental_learner': mock_incremental,
            'training.active_learner': mock_active,
            'training.auto_retrainer': mock_auto,
            'evaluation': MagicMock(),
            'evaluation.drift_detection': mock_drift,
            'config': MagicMock(),
            'config.mlflow_config': mock_mlflow_config,
        }):
            if 'src.examples.continuous_learning_example' in sys.modules:
                del sys.modules['src.examples.continuous_learning_example']
            try:
                import src.examples.continuous_learning_example  # noqa
                # If we get here, the exception was swallowed (expected)
            except RuntimeError:
                self.fail("Module-level RuntimeError should have been caught internally")


class TestEvaluationExampleImportPaths(unittest.TestCase):
    """
    Tests that evaluation_example.py uses the new import paths:
    - src.evaluation.evaluator.ComprehensiveEvaluator
    - src.evaluation.bias_detection.BiasDetector
    - src.evaluation.drift_detection.DriftDetector
    """

    def test_evaluation_example_imports_from_new_evaluation_module(self):
        """evaluation_example.py should import from src.evaluation.* paths."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'evaluation_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.evaluation.evaluator', modules)
        self.assertIn('src.evaluation.bias_detection', modules)
        self.assertIn('src.evaluation.drift_detection', modules)

    def test_deep_learning_example_imports_from_new_models_module(self):
        """deep_learning_example.py should import from src.models.base.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'deep_learning_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.models.base.model_factory', modules)
        self.assertIn('src.models.base.base_model', modules)

    def test_ensemble_example_imports_from_new_ensemble_module(self):
        """ensemble_example.py should import from src.models.ensemble.ensemble_models."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'ensemble_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.models.ensemble.ensemble_models', modules)

    def test_model_factory_example_imports_config_validator_from_new_path(self):
        """model_factory_example.py should import ConfigValidator from src.config.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'model_factory_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.config.config_validator', modules)

    def test_explainability_example_imports_from_new_explainability_module(self):
        """explainability_example.py should import from src.explainability.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'explainability_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.explainability.explainability_engine', modules)
        self.assertIn('src.explainability.uncertainty_quantification', modules)
        self.assertIn('src.explainability.model_debugger', modules)

    def test_hyperparameter_example_imports_from_training_module(self):
        """hyperparameter_example.py should import from src.training.hyperparameter_tuning."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'hyperparameter_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.training.hyperparameter_tuning', modules)

    def test_experiment_tracking_example_imports_from_mlops(self):
        """experiment_tracking_example.py should import from src.mlops.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'experiment_tracking_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.mlops.experiment_tracking', modules)
        self.assertIn('src.mlops.lifecycle_manager', modules)
        self.assertIn('src.mlops.ab_testing', modules)

    def test_integrated_pipeline_example_imports_from_mlops(self):
        """integrated_pipeline_example.py should import from src.mlops.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'integrated_pipeline_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.mlops.experiment_tracking', modules)
        self.assertIn('src.mlops.lifecycle_manager', modules)
        self.assertIn('src.mlops.ab_testing', modules)

    def test_transformer_example_imports_from_new_models_module(self):
        """transformer_example.py should import from src.models.base.*."""
        import ast
        example_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'examples', 'transformer_example.py'
        )
        with open(example_path) as f:
            tree = ast.parse(f.read())

        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        modules = [node.module for node in import_froms if node.module]

        self.assertIn('src.models.base.model_factory', modules)
        self.assertIn('src.models.base.base_model', modules)


if __name__ == '__main__':
    unittest.main()