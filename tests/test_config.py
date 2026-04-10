"""
Tests for src/config module:
- src/config/__init__.py
- src/config/mlflow_config.py
- src/config/config_validator.py
"""

import os
import unittest
from unittest.mock import patch, MagicMock, call
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfigInit(unittest.TestCase):
    """Tests for src/config/__init__.py exports."""

    def test_setup_mlflow_exported(self):
        """setup_mlflow should be importable from src.config."""
        from src.config import setup_mlflow
        self.assertTrue(callable(setup_mlflow))

    def test_get_or_create_experiment_exported(self):
        """get_or_create_experiment should be importable from src.config."""
        from src.config import get_or_create_experiment
        self.assertTrue(callable(get_or_create_experiment))

    def test_all_list(self):
        """__all__ should contain the two exported names."""
        import src.config as config_module
        self.assertIn('setup_mlflow', config_module.__all__)
        self.assertIn('get_or_create_experiment', config_module.__all__)

    def test_functions_are_same_as_mlflow_config(self):
        """Exported functions should be the same objects as in mlflow_config."""
        from src.config import setup_mlflow, get_or_create_experiment
        from src.config.mlflow_config import (
            setup_mlflow as sm,
            get_or_create_experiment as goce,
        )
        self.assertIs(setup_mlflow, sm)
        self.assertIs(get_or_create_experiment, goce)


class TestSetupMlflow(unittest.TestCase):
    """Tests for src/config/mlflow_config.setup_mlflow()."""

    def setUp(self):
        # Clear relevant env vars before each test
        for key in [
            'MLFLOW_TRACKING_URI',
            'DAGSHUB_PAT',
            'MLFLOW_TRACKING_USERNAME',
            'MLFLOW_TRACKING_PASSWORD',
        ]:
            os.environ.pop(key, None)

    def tearDown(self):
        for key in [
            'MLFLOW_TRACKING_URI',
            'DAGSHUB_PAT',
            'MLFLOW_TRACKING_USERNAME',
            'MLFLOW_TRACKING_PASSWORD',
        ]:
            os.environ.pop(key, None)

    @patch('src.config.mlflow_config.mlflow')
    def test_sets_tracking_uri_when_env_var_present(self, mock_mlflow):
        """setup_mlflow() calls mlflow.set_tracking_uri when MLFLOW_TRACKING_URI is set."""
        os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/user/repo.mlflow'
        from src.config.mlflow_config import setup_mlflow
        setup_mlflow()
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            'https://dagshub.com/user/repo.mlflow'
        )

    @patch('src.config.mlflow_config.mlflow')
    def test_no_tracking_uri_call_when_env_var_absent(self, mock_mlflow):
        """setup_mlflow() does not call set_tracking_uri when env var is absent."""
        from src.config.mlflow_config import setup_mlflow
        setup_mlflow()
        mock_mlflow.set_tracking_uri.assert_not_called()

    @patch('src.config.mlflow_config.mlflow')
    def test_dagshub_pat_sets_credentials(self, mock_mlflow):
        """DAGSHUB_PAT sets both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD."""
        os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/user/repo.mlflow'
        os.environ['DAGSHUB_PAT'] = 'my_secret_pat'
        from src.config.mlflow_config import setup_mlflow
        setup_mlflow()
        self.assertEqual(os.environ.get('MLFLOW_TRACKING_USERNAME'), 'my_secret_pat')
        self.assertEqual(os.environ.get('MLFLOW_TRACKING_PASSWORD'), 'my_secret_pat')

    @patch('src.config.mlflow_config.mlflow')
    def test_username_password_env_vars_preserved(self, mock_mlflow):
        """Explicit username/password env vars are set when tracking URI is present."""
        os.environ['MLFLOW_TRACKING_URI'] = 'https://example.com/mlflow'
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'user'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'pass'
        from src.config.mlflow_config import setup_mlflow
        setup_mlflow()
        self.assertEqual(os.environ.get('MLFLOW_TRACKING_USERNAME'), 'user')
        self.assertEqual(os.environ.get('MLFLOW_TRACKING_PASSWORD'), 'pass')

    @patch('src.config.mlflow_config.mlflow')
    def test_no_error_without_any_env_vars(self, mock_mlflow):
        """setup_mlflow() completes without error when no env vars are set."""
        from src.config.mlflow_config import setup_mlflow
        # Should not raise
        setup_mlflow()


class TestGetOrCreateExperiment(unittest.TestCase):
    """Tests for src/config/mlflow_config.get_or_create_experiment()."""

    @patch('src.config.mlflow_config.mlflow')
    def test_returns_existing_experiment_id(self, mock_mlflow):
        """Returns the experiment_id when experiment already exists."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = 'existing_id_42'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        from src.config.mlflow_config import get_or_create_experiment
        result = get_or_create_experiment('my_experiment')

        mock_mlflow.get_experiment_by_name.assert_called_once_with('my_experiment')
        mock_mlflow.create_experiment.assert_not_called()
        self.assertEqual(result, 'existing_id_42')

    @patch('src.config.mlflow_config.mlflow')
    def test_creates_new_experiment_when_not_found(self, mock_mlflow):
        """Creates a new experiment when it doesn't exist and returns its id."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = 'new_id_99'

        from src.config.mlflow_config import get_or_create_experiment
        result = get_or_create_experiment('brand_new_experiment')

        mock_mlflow.create_experiment.assert_called_once_with('brand_new_experiment')
        self.assertEqual(result, 'new_id_99')

    @patch('src.config.mlflow_config.mlflow')
    def test_returns_none_on_exception(self, mock_mlflow):
        """Returns None when mlflow raises an exception."""
        mock_mlflow.get_experiment_by_name.side_effect = RuntimeError('connection error')

        from src.config.mlflow_config import get_or_create_experiment
        result = get_or_create_experiment('failing_experiment')

        self.assertIsNone(result)

    @patch('src.config.mlflow_config.mlflow')
    def test_structured_experiment_name_01(self, mock_mlflow):
        """Handles structured experiment name '01_Data_Preprocessing'."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = '1'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        from src.config.mlflow_config import get_or_create_experiment
        result = get_or_create_experiment('01_Data_Preprocessing')
        self.assertEqual(result, '1')

    @patch('src.config.mlflow_config.mlflow')
    def test_structured_experiment_name_04(self, mock_mlflow):
        """Handles structured experiment name '04_Model_Evaluation'."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = '4'

        from src.config.mlflow_config import get_or_create_experiment
        result = get_or_create_experiment('04_Model_Evaluation')
        mock_mlflow.create_experiment.assert_called_once_with('04_Model_Evaluation')
        self.assertEqual(result, '4')


class TestConfigValidator(unittest.TestCase):
    """Tests for src/config/config_validator.ConfigValidator."""

    def setUp(self):
        from src.config.config_validator import ConfigValidator
        self.validator = ConfigValidator

    # --- validate_config: valid cases ---

    def test_valid_logistic_regression_config(self):
        valid_params = {'max_iter': 1000, 'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
        is_valid, errors = self.validator.validate_config('logistic_regression', valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_random_forest_config(self):
        valid_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
        is_valid, errors = self.validator.validate_config('random_forest', valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_xgboost_config(self):
        valid_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
        is_valid, errors = self.validator.validate_config('xgboost', valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_lightgbm_config(self):
        valid_params = {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'num_leaves': 31}
        is_valid, errors = self.validator.validate_config('lightgbm', valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_svm_config(self):
        valid_params = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'scale'}
        is_valid, errors = self.validator.validate_config('svm', valid_params)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    # --- validate_config: invalid cases ---

    def test_invalid_logistic_regression_c_too_low(self):
        is_valid, errors = self.validator.validate_config(
            'logistic_regression', {'C': 0.00001}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('C' in e for e in errors))

    def test_invalid_logistic_regression_c_too_high(self):
        is_valid, errors = self.validator.validate_config(
            'logistic_regression', {'C': 99999}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('C' in e for e in errors))

    def test_invalid_logistic_regression_penalty(self):
        is_valid, errors = self.validator.validate_config(
            'logistic_regression', {'penalty': 'invalid_penalty'}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('penalty' in e for e in errors))

    def test_invalid_logistic_regression_solver(self):
        is_valid, errors = self.validator.validate_config(
            'logistic_regression', {'solver': 'not_a_solver'}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('solver' in e for e in errors))

    def test_invalid_random_forest_n_estimators_zero(self):
        is_valid, errors = self.validator.validate_config(
            'random_forest', {'n_estimators': 0}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('n_estimators' in e for e in errors))

    def test_invalid_random_forest_n_estimators_too_high(self):
        is_valid, errors = self.validator.validate_config(
            'random_forest', {'n_estimators': 1001}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('n_estimators' in e for e in errors))

    def test_invalid_xgboost_learning_rate_too_low(self):
        is_valid, errors = self.validator.validate_config(
            'xgboost', {'learning_rate': 0.0001}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('learning_rate' in e for e in errors))

    def test_invalid_xgboost_learning_rate_too_high(self):
        is_valid, errors = self.validator.validate_config(
            'xgboost', {'learning_rate': 1.5}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('learning_rate' in e for e in errors))

    def test_invalid_svm_kernel(self):
        is_valid, errors = self.validator.validate_config(
            'svm', {'kernel': 'invalid_kernel'}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('kernel' in e for e in errors))

    def test_invalid_lightgbm_num_leaves_too_low(self):
        is_valid, errors = self.validator.validate_config(
            'lightgbm', {'num_leaves': 1}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('num_leaves' in e for e in errors))

    def test_invalid_type_string_for_int_param(self):
        is_valid, errors = self.validator.validate_config(
            'random_forest', {'n_estimators': 'many'}
        )
        self.assertFalse(is_valid)
        self.assertTrue(any('n_estimators' in e for e in errors))

    # --- validate_config: unknown model type ---

    def test_unknown_model_type_returns_valid(self):
        """Unknown model types pass validation (no rules defined)."""
        is_valid, errors = self.validator.validate_config(
            'neural_network', {'hidden_layers': 3}
        )
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    # --- validate_config: unknown parameters are skipped ---

    def test_unknown_param_is_skipped(self):
        """Parameters not in validation rules are silently skipped."""
        is_valid, errors = self.validator.validate_config(
            'random_forest', {'random_state': 42, 'unknown_param': 'value'}
        )
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    # --- validate_config: empty hyperparameters ---

    def test_empty_hyperparameters_passes(self):
        is_valid, errors = self.validator.validate_config('logistic_regression', {})
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    # --- validate_config: multiple errors collected ---

    def test_multiple_errors_collected(self):
        """All validation errors are collected, not short-circuited."""
        params = {'C': 0.00001, 'penalty': 'invalid', 'solver': 'bad_solver'}
        is_valid, errors = self.validator.validate_config('logistic_regression', params)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 2)

    # --- get_default_hyperparameters ---

    def test_default_hyperparameters_logistic_regression(self):
        defaults = self.validator.get_default_hyperparameters('logistic_regression')
        self.assertIn('max_iter', defaults)
        self.assertIn('C', defaults)
        self.assertIn('penalty', defaults)
        self.assertIn('solver', defaults)
        self.assertEqual(defaults['C'], 1.0)
        self.assertEqual(defaults['penalty'], 'l2')

    def test_default_hyperparameters_random_forest(self):
        defaults = self.validator.get_default_hyperparameters('random_forest')
        self.assertIn('n_estimators', defaults)
        self.assertEqual(defaults['n_estimators'], 100)
        self.assertIn('random_state', defaults)

    def test_default_hyperparameters_xgboost(self):
        defaults = self.validator.get_default_hyperparameters('xgboost')
        self.assertIn('learning_rate', defaults)
        self.assertEqual(defaults['learning_rate'], 0.1)
        self.assertEqual(defaults['max_depth'], 6)

    def test_default_hyperparameters_lightgbm(self):
        defaults = self.validator.get_default_hyperparameters('lightgbm')
        self.assertIn('num_leaves', defaults)
        self.assertEqual(defaults['num_leaves'], 31)
        self.assertEqual(defaults['max_depth'], -1)

    def test_default_hyperparameters_svm(self):
        defaults = self.validator.get_default_hyperparameters('svm')
        self.assertIn('C', defaults)
        self.assertEqual(defaults['C'], 1.0)
        self.assertEqual(defaults['kernel'], 'rbf')

    def test_default_hyperparameters_unknown_model(self):
        """Unknown model type returns empty dict."""
        defaults = self.validator.get_default_hyperparameters('unknown_model')
        self.assertEqual(defaults, {})

    # --- default hyperparameters pass their own validation ---

    def test_defaults_pass_validation_logistic_regression(self):
        defaults = self.validator.get_default_hyperparameters('logistic_regression')
        is_valid, errors = self.validator.validate_config('logistic_regression', defaults)
        self.assertTrue(is_valid, f"Default params failed validation: {errors}")

    def test_defaults_pass_validation_random_forest(self):
        defaults = self.validator.get_default_hyperparameters('random_forest')
        is_valid, errors = self.validator.validate_config('random_forest', defaults)
        self.assertTrue(is_valid, f"Default params failed validation: {errors}")

    def test_defaults_pass_validation_xgboost(self):
        defaults = self.validator.get_default_hyperparameters('xgboost')
        is_valid, errors = self.validator.validate_config('xgboost', defaults)
        self.assertTrue(is_valid, f"Default params failed validation: {errors}")

    def test_defaults_pass_validation_lightgbm(self):
        defaults = self.validator.get_default_hyperparameters('lightgbm')
        is_valid, errors = self.validator.validate_config('lightgbm', defaults)
        self.assertTrue(is_valid, f"Default params failed validation: {errors}")

    def test_defaults_pass_validation_svm(self):
        defaults = self.validator.get_default_hyperparameters('svm')
        is_valid, errors = self.validator.validate_config('svm', defaults)
        self.assertTrue(is_valid, f"Default params failed validation: {errors}")

    # --- _validate_type static method ---

    def test_validate_type_none_expected_type(self):
        """None expected_type always returns True."""
        from src.config.config_validator import ConfigValidator
        self.assertTrue(ConfigValidator._validate_type(42, None))
        self.assertTrue(ConfigValidator._validate_type('str', None))

    def test_validate_type_single_type_match(self):
        from src.config.config_validator import ConfigValidator
        self.assertTrue(ConfigValidator._validate_type(42, int))
        self.assertFalse(ConfigValidator._validate_type('str', int))

    def test_validate_type_tuple_of_types(self):
        from src.config.config_validator import ConfigValidator
        self.assertTrue(ConfigValidator._validate_type(3.14, (int, float)))
        self.assertTrue(ConfigValidator._validate_type(3, (int, float)))
        self.assertFalse(ConfigValidator._validate_type('str', (int, float)))

    def test_validate_type_none_value_with_optional(self):
        from src.config.config_validator import ConfigValidator
        self.assertTrue(ConfigValidator._validate_type(None, (int, type(None))))

    # --- boundary value tests ---

    def test_xgboost_max_depth_boundary_min(self):
        is_valid, errors = self.validator.validate_config('xgboost', {'max_depth': 1})
        self.assertTrue(is_valid)

    def test_xgboost_max_depth_boundary_max(self):
        is_valid, errors = self.validator.validate_config('xgboost', {'max_depth': 20})
        self.assertTrue(is_valid)

    def test_xgboost_max_depth_exceeds_max(self):
        is_valid, errors = self.validator.validate_config('xgboost', {'max_depth': 21})
        self.assertFalse(is_valid)

    def test_lightgbm_max_depth_negative_one_valid(self):
        """lightgbm allows max_depth=-1 (no limit)."""
        is_valid, errors = self.validator.validate_config('lightgbm', {'max_depth': -1})
        self.assertTrue(is_valid)

    def test_logistic_regression_max_iter_boundary_min(self):
        is_valid, _ = self.validator.validate_config('logistic_regression', {'max_iter': 1})
        self.assertTrue(is_valid)

    def test_logistic_regression_max_iter_boundary_max(self):
        is_valid, _ = self.validator.validate_config('logistic_regression', {'max_iter': 10000})
        self.assertTrue(is_valid)

    def test_logistic_regression_max_iter_exceeds_max(self):
        is_valid, errors = self.validator.validate_config('logistic_regression', {'max_iter': 10001})
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()