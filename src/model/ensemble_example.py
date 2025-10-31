"""
Example usage of ensemble models for sentiment analysis.
Demonstrates voting, stacking, and blending ensemble methods.
"""

import numpy as np
from src.model.model_factory import ModelFactory
from src.model.ensemble_models import VotingEnsemble, StackingEnsemble, BlendingEnsemble
from src.model.base_model import ModelConfig


def create_sample_data(num_samples=1000, num_features=20, num_classes=3):
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


def example_voting_ensemble():
    """Example of voting ensemble with multiple models."""
    print("\n" + "="*60)
    print("Voting Ensemble Example")
    print("="*60)
    
    # Create sample data
    X_train, y_train = create_sample_data(num_samples=800)
    X_test, y_test = create_sample_data(num_samples=200)
    
    # Create voting ensemble
    config = ModelConfig(
        model_type='voting',
        model_name='voting_ensemble',
        hyperparameters={
            'voting_type': 'soft',  # or 'hard'
            'weights': None  # Equal weights
        }
    )
    
    ensemble = VotingEnsemble(config)
    
    # Add base models
    print("Creating base models...")
    
    # Model 1: Logistic Regression
    lr_model = ModelFactory.create_model('logistic_regression', max_iter=500)
    ensemble.add_model(lr_model)
    
    # Model 2: Random Forest
    rf_model = ModelFactory.create_model('random_forest', n_estimators=50)
    ensemble.add_model(rf_model)
    
    # Model 3: LightGBM
    lgbm_model = ModelFactory.create_model('lightgbm', n_estimators=50)
    ensemble.add_model(lgbm_model)
    
    print(f"Added {len(ensemble.base_models)} base models")
    
    # Train ensemble
    print("\nTraining voting ensemble...")
    train_result = ensemble.train(X_train, y_train)
    print(f"Training completed: {train_result['status']}")
    print(f"Voting type: {train_result['voting_type']}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Ensemble accuracy: {accuracy:.4f}")
    
    print(f"\nSample predictions: {predictions[:10]}")
    print(f"Sample probabilities:\n{probabilities[:3]}")


def example_weighted_voting():
    """Example of weighted voting ensemble."""
    print("\n" + "="*60)
    print("Weighted Voting Ensemble Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=800)
    X_test, y_test = create_sample_data(num_samples=200)
    
    # Create weighted voting ensemble
    config = ModelConfig(
        model_type='voting',
        model_name='weighted_voting',
        hyperparameters={
            'voting_type': 'soft',
            'weights': [2.0, 1.5, 1.0]  # Higher weight for first model
        }
    )
    
    ensemble = VotingEnsemble(config)
    
    # Add models
    ensemble.add_model(ModelFactory.create_model('lightgbm', n_estimators=100))
    ensemble.add_model(ModelFactory.create_model('random_forest', n_estimators=50))
    ensemble.add_model(ModelFactory.create_model('logistic_regression'))
    
    print(f"Weights: {config.hyperparameters['weights']}")
    
    # Train
    print("\nTraining weighted ensemble...")
    ensemble.train(X_train, y_train)
    
    # Predict
    predictions = ensemble.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Weighted ensemble accuracy: {accuracy:.4f}")


def example_stacking_ensemble():
    """Example of stacking ensemble with meta-model."""
    print("\n" + "="*60)
    print("Stacking Ensemble Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=800)
    X_test, y_test = create_sample_data(num_samples=200)
    
    # Create stacking ensemble
    config = ModelConfig(
        model_type='stacking',
        model_name='stacking_ensemble',
        hyperparameters={
            'use_probabilities': True
        }
    )
    
    ensemble = StackingEnsemble(config)
    
    # Add base models (diverse set)
    print("Creating base models...")
    ensemble.add_base_model(ModelFactory.create_model('random_forest', n_estimators=50))
    ensemble.add_base_model(ModelFactory.create_model('lightgbm', n_estimators=50))
    ensemble.add_base_model(ModelFactory.create_model('logistic_regression'))
    
    # Set meta-model (typically a simple model)
    print("Setting meta-model (Logistic Regression)...")
    meta_model = ModelFactory.create_model('logistic_regression', max_iter=500)
    ensemble.set_meta_model(meta_model)
    
    # Train ensemble
    print("\nTraining stacking ensemble...")
    train_result = ensemble.train(X_train, y_train)
    print(f"Training completed: {train_result['status']}")
    print(f"Number of base models: {train_result['num_base_models']}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"Stacking ensemble accuracy: {accuracy:.4f}")
    
    print(f"\nSample predictions: {predictions[:10]}")


def example_blending_ensemble():
    """Example of blending ensemble."""
    print("\n" + "="*60)
    print("Blending Ensemble Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=800)
    X_test, y_test = create_sample_data(num_samples=200)
    
    # Create blending ensemble
    config = ModelConfig(
        model_type='blending',
        model_name='blending_ensemble',
        hyperparameters={
            'blend_ratio': 0.2  # 20% for blending, 80% for training
        }
    )
    
    ensemble = BlendingEnsemble(config)
    
    # Add base models
    print("Creating base models...")
    ensemble.add_base_model(ModelFactory.create_model('random_forest', n_estimators=50))
    ensemble.add_base_model(ModelFactory.create_model('xgboost', n_estimators=50))
    ensemble.add_base_model(ModelFactory.create_model('lightgbm', n_estimators=50))
    
    # Set meta-model
    print("Setting meta-model...")
    meta_model = ModelFactory.create_model('logistic_regression')
    ensemble.set_meta_model(meta_model)
    
    # Train ensemble
    print("\nTraining blending ensemble...")
    train_result = ensemble.train(X_train, y_train)
    print(f"Training completed: {train_result['status']}")
    print(f"Train size: {train_result['train_size']}, Blend size: {train_result['blend_size']}")
    
    # Make predictions
    predictions = ensemble.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Blending ensemble accuracy: {accuracy:.4f}")


def example_ensemble_comparison():
    """Compare different ensemble methods."""
    print("\n" + "="*60)
    print("Ensemble Methods Comparison")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=800)
    X_test, y_test = create_sample_data(num_samples=200)
    
    results = {}
    
    # 1. Voting Ensemble
    print("\n1. Training Voting Ensemble...")
    voting_config = ModelConfig(
        model_type='voting',
        model_name='voting',
        hyperparameters={'voting_type': 'soft'}
    )
    voting = VotingEnsemble(voting_config)
    voting.add_model(ModelFactory.create_model('random_forest', n_estimators=50))
    voting.add_model(ModelFactory.create_model('lightgbm', n_estimators=50))
    voting.add_model(ModelFactory.create_model('logistic_regression'))
    voting.train(X_train, y_train)
    
    voting_pred = voting.predict(X_test)
    results['Voting'] = np.mean(voting_pred == y_test)
    print(f"Voting accuracy: {results['Voting']:.4f}")
    
    # 2. Stacking Ensemble
    print("\n2. Training Stacking Ensemble...")
    stacking_config = ModelConfig(
        model_type='stacking',
        model_name='stacking',
        hyperparameters={'use_probabilities': True}
    )
    stacking = StackingEnsemble(stacking_config)
    stacking.add_base_model(ModelFactory.create_model('random_forest', n_estimators=50))
    stacking.add_base_model(ModelFactory.create_model('lightgbm', n_estimators=50))
    stacking.add_base_model(ModelFactory.create_model('logistic_regression'))
    stacking.set_meta_model(ModelFactory.create_model('logistic_regression'))
    stacking.train(X_train, y_train)
    
    stacking_pred = stacking.predict(X_test)
    results['Stacking'] = np.mean(stacking_pred == y_test)
    print(f"Stacking accuracy: {results['Stacking']:.4f}")
    
    # 3. Blending Ensemble
    print("\n3. Training Blending Ensemble...")
    blending_config = ModelConfig(
        model_type='blending',
        model_name='blending',
        hyperparameters={'blend_ratio': 0.2}
    )
    blending = BlendingEnsemble(blending_config)
    blending.add_base_model(ModelFactory.create_model('random_forest', n_estimators=50))
    blending.add_base_model(ModelFactory.create_model('lightgbm', n_estimators=50))
    blending.add_base_model(ModelFactory.create_model('logistic_regression'))
    blending.set_meta_model(ModelFactory.create_model('logistic_regression'))
    blending.train(X_train, y_train)
    
    blending_pred = blending.predict(X_test)
    results['Blending'] = np.mean(blending_pred == y_test)
    print(f"Blending accuracy: {results['Blending']:.4f}")
    
    # 4. Single model baseline
    print("\n4. Training Single Model Baseline (LightGBM)...")
    baseline = ModelFactory.create_model('lightgbm', n_estimators=100)
    baseline.train(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    results['Baseline'] = np.mean(baseline_pred == y_test)
    print(f"Baseline accuracy: {results['Baseline']:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    for method, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:15s}: {accuracy:.4f}")


if __name__ == "__main__":
    print("Ensemble Models Examples")
    print("="*60)
    
    try:
        # Run examples
        example_voting_ensemble()
        example_weighted_voting()
        example_stacking_ensemble()
        example_blending_ensemble()
        example_ensemble_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
