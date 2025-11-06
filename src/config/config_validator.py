"""
Configuration validation for model parameters.
Ensures model configurations are valid before instantiation.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates model configurations and hyperparameters."""
    
    # Define valid hyperparameter ranges for different model types
    _validation_rules = {
        'logistic_regression': {
            'max_iter': {'type': int, 'min': 1, 'max': 10000},
            'C': {'type': (int, float), 'min': 0.0001, 'max': 1000},
            'penalty': {'type': str, 'values': ['l1', 'l2', 'elasticnet', 'none']},
            'solver': {'type': str, 'values': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        },
        'random_forest': {
            'n_estimators': {'type': int, 'min': 1, 'max': 1000},
            'max_depth': {'type': (int, type(None)), 'min': 1, 'max': 100},
            'min_samples_split': {'type': (int, float), 'min': 2},
            'min_samples_leaf': {'type': (int, float), 'min': 1}
        },
        'xgboost': {
            'n_estimators': {'type': int, 'min': 1, 'max': 1000},
            'max_depth': {'type': int, 'min': 1, 'max': 20},
            'learning_rate': {'type': float, 'min': 0.001, 'max': 1.0},
            'subsample': {'type': float, 'min': 0.1, 'max': 1.0}
        },
        'lightgbm': {
            'n_estimators': {'type': int, 'min': 1, 'max': 1000},
            'max_depth': {'type': int, 'min': -1, 'max': 100},
            'learning_rate': {'type': float, 'min': 0.001, 'max': 1.0},
            'num_leaves': {'type': int, 'min': 2, 'max': 1000}
        },
        'svm': {
            'C': {'type': (int, float), 'min': 0.0001, 'max': 1000},
            'kernel': {'type': str, 'values': ['linear', 'poly', 'rbf', 'sigmoid']},
            'degree': {'type': int, 'min': 1, 'max': 10},
            'gamma': {'type': (str, float), 'values': ['scale', 'auto']}
        }
    }
    
    @classmethod
    def validate_config(cls, model_type: str, hyperparameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate model configuration.
        
        Args:
            model_type: Type of model
            hyperparameters: Dictionary of hyperparameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if model_type not in cls._validation_rules:
            logger.warning(f"No validation rules defined for model type: {model_type}")
            return True, []
        
        rules = cls._validation_rules[model_type]
        
        for param_name, param_value in hyperparameters.items():
            if param_name not in rules:
                continue  # Skip parameters without validation rules
            
            rule = rules[param_name]
            
            # Type validation
            if not cls._validate_type(param_value, rule.get('type')):
                errors.append(
                    f"Parameter '{param_name}' has invalid type. "
                    f"Expected {rule.get('type')}, got {type(param_value)}"
                )
                continue
            
            # Value range validation
            if 'min' in rule and param_value is not None:
                if isinstance(param_value, (int, float)) and param_value < rule['min']:
                    errors.append(
                        f"Parameter '{param_name}' value {param_value} is below minimum {rule['min']}"
                    )
            
            if 'max' in rule and param_value is not None:
                if isinstance(param_value, (int, float)) and param_value > rule['max']:
                    errors.append(
                        f"Parameter '{param_name}' value {param_value} exceeds maximum {rule['max']}"
                    )
            
            # Categorical value validation
            if 'values' in rule and param_value not in rule['values']:
                errors.append(
                    f"Parameter '{param_name}' has invalid value '{param_value}'. "
                    f"Must be one of: {rule['values']}"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"Configuration validation failed for {model_type}: {errors}")
        
        return is_valid, errors
    
    @staticmethod
    def _validate_type(value: Any, expected_type: Any) -> bool:
        """
        Validate if value matches expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type or tuple of types
            
        Returns:
            True if type matches, False otherwise
        """
        if expected_type is None:
            return True
        
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        
        return isinstance(value, expected_type)
    
    @classmethod
    def get_default_hyperparameters(cls, model_type: str) -> Dict[str, Any]:
        """
        Get recommended default hyperparameters for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default hyperparameters
        """
        defaults = {
            'logistic_regression': {
                'max_iter': 1000,
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        }
        
        return defaults.get(model_type, {})
