"""
Example usage of the ModelFactory and base model interfaces.
Demonstrates how to create, train, and use different model types.
"""

import numpy as np
from src.models.base.model_factory import ModelFactory
from src.models.base.base_model import ModelConfig
from src.config.config_validator import ConfigValidator


def example_basic_usage():
    """Basic example of creating and using models."""
    
    # Create sample data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 10)
    
    # Example 1: Create a model with default configuration
    print("Example 1: Creating model with defaults")
    model = ModelFactory.create_model('logistic_regression')
    print(f"Created model: {model.config.model_name}")
    
    # Train the model
    train_result = model.train(X_train, y_train)
    print(f"Training result: {train_result}")
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Example 2: Create a model with custom hyperparameters
    print("\nExample 2: Creating model with custom hyperparameters")
    custom_config = ModelConfig(
        model_type='random_forest',
        model_name='custom_rf_model',
        hyperparameters={
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42
        }
    )
    
    rf_model = ModelFactory.create_model('random_forest', config=custom_config)
    rf_model.train(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print(f"Random Forest predictions: {rf_predictions[:5]}")
    
    # Example 3: Using kwargs to override defaults
    print("\nExample 3: Creating model with kwargs")
    xgb_model = ModelFactory.create_model(
        'xgboost',
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05
    )
    xgb_model.train(X_train, y_train)
    
    # Example 4: List available models
    print("\nExample 4: Available models")
    available_models = ModelFactory.list_available_models()
    print(f"Available model types: {available_models}")
    
    # Example 5: Get model information
    print("\nExample 5: Model information")
    model_info = ModelFactory.get_model_info('lightgbm')
    print(f"LightGBM info: {model_info}")


def example_config_validation():
    """Example of configuration validation."""
    
    print("\nConfiguration Validation Examples")
    print("=" * 50)
    
    # Valid configuration
    valid_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    is_valid, errors = ConfigValidator.validate_config('xgboost', valid_params)
    print(f"Valid config: {is_valid}, Errors: {errors}")
    
    # Invalid configuration (learning_rate too high)
    invalid_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 2.0  # Invalid: > 1.0
    }
    is_valid, errors = ConfigValidator.validate_config('xgboost', invalid_params)
    print(f"Invalid config: {is_valid}, Errors: {errors}")
    
    # Get default hyperparameters
    defaults = ConfigValidator.get_default_hyperparameters('random_forest')
    print(f"Default Random Forest params: {defaults}")


def example_model_persistence():
    """Example of saving and loading models."""
    
    print("\nModel Persistence Example")
    print("=" * 50)
    
    # Create and train a model
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    model = ModelFactory.create_model('logistic_regression')
    model.train(X_train, y_train)
    
    # Save the model
    model_path = 'models/trained_models/example_model.pkl'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Load the model
    loaded_model = ModelFactory.create_model('logistic_regression')
    loaded_model.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Verify predictions match
    X_test = np.random.randn(10, 10)
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    print(f"Predictions match: {np.array_equal(original_pred, loaded_pred)}")


if __name__ == "__main__":
    print("ModelFactory Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_config_validation()
        # example_model_persistence()  # Uncomment to test persistence
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()
