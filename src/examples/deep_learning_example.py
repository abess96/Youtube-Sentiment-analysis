"""
Example usage of deep learning models for sentiment analysis.
Demonstrates LSTM, GRU, CNN, and BiLSTM models.
"""

import numpy as np
from src.models.base.model_factory import ModelFactory
from src.models.base.base_model import ModelConfig


def create_sample_data(num_samples=1000, seq_length=50, vocab_size=10000, num_classes=3):
    """Create sample sequential data for testing."""
    X = np.random.randint(1, vocab_size, size=(num_samples, seq_length))
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y


def example_lstm_model():
    """Example of training and using LSTM model."""
    print("\n" + "="*60)
    print("LSTM Model Example")
    print("="*60)
    
    # Create sample data
    X_train, y_train = create_sample_data(num_samples=500, seq_length=50)
    X_test, y_test = create_sample_data(num_samples=100, seq_length=50)
    
    # Create LSTM model with custom configuration
    config = ModelConfig(
        model_type='lstm',
        model_name='sentiment_lstm',
        hyperparameters={
            'vocab_size': 10000,
            'num_classes': 3,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'n_layers': 2,
            'dropout': 0.3,
            'use_attention': True,
            'batch_size': 32,
            'epochs': 3,
            'learning_rate': 0.001
        }
    )
    
    model = ModelFactory.create_model('lstm', config=config)
    print(f"Created model: {model.config.model_name}")
    print(f"Device: {model.device}")
    
    # Train the model
    print("\nTraining LSTM model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test[:10])
    probabilities = model.predict_proba(X_test[:10])
    
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities shape: {probabilities.shape}")
    print(f"First prediction probabilities: {probabilities[0]}")


def example_gru_model():
    """Example of training and using GRU model."""
    print("\n" + "="*60)
    print("GRU Model Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=500, seq_length=50)
    X_test, y_test = create_sample_data(num_samples=100, seq_length=50)
    
    # Create GRU model using kwargs
    model = ModelFactory.create_model(
        'gru',
        vocab_size=10000,
        num_classes=3,
        embedding_dim=64,
        hidden_dim=128,
        epochs=3
    )
    
    print(f"Created GRU model")
    
    # Train
    print("\nTraining GRU model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Predict
    predictions = model.predict(X_test[:5])
    print(f"Sample predictions: {predictions}")


def example_cnn_model():
    """Example of training and using CNN model."""
    print("\n" + "="*60)
    print("CNN Model Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=500, seq_length=50)
    X_test, y_test = create_sample_data(num_samples=100, seq_length=50)
    
    # Create CNN model
    model = ModelFactory.create_model(
        'cnn',
        vocab_size=10000,
        num_classes=3,
        embedding_dim=64,
        n_filters=100,
        filter_sizes=[3, 4, 5],
        epochs=3
    )
    
    print(f"Created CNN model")
    
    # Train
    print("\nTraining CNN model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Predict
    predictions = model.predict(X_test[:5])
    probabilities = model.predict_proba(X_test[:5])
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities:\n{probabilities}")


def example_bilstm_model():
    """Example of training and using Bidirectional LSTM model."""
    print("\n" + "="*60)
    print("Bidirectional LSTM Model Example")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=500, seq_length=50)
    X_test, y_test = create_sample_data(num_samples=100, seq_length=50)
    
    # Create BiLSTM model
    model = ModelFactory.create_model(
        'bilstm',
        vocab_size=10000,
        num_classes=3,
        embedding_dim=64,
        hidden_dim=128,
        epochs=3
    )
    
    print(f"Created BiLSTM model")
    print(f"Bidirectional: {model.config.hyperparameters['bidirectional']}")
    
    # Train
    print("\nTraining BiLSTM model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Predict
    predictions = model.predict(X_test[:5])
    print(f"Sample predictions: {predictions}")


def example_model_comparison():
    """Compare different model architectures."""
    print("\n" + "="*60)
    print("Model Architecture Comparison")
    print("="*60)
    
    X_train, y_train = create_sample_data(num_samples=300, seq_length=50)
    X_test, y_test = create_sample_data(num_samples=50, seq_length=50)
    
    model_types = ['lstm', 'gru', 'cnn', 'bilstm']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        
        model = ModelFactory.create_model(
            model_type,
            vocab_size=10000,
            num_classes=3,
            embedding_dim=32,
            hidden_dim=64,
            epochs=2,
            batch_size=32
        )
        
        train_result = model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate simple accuracy
        accuracy = np.mean(predictions == y_test)
        results[model_type] = {
            'final_loss': train_result.get('final_loss', 'N/A'),
            'accuracy': accuracy
        }
        
        print(f"{model_type.upper()} - Loss: {results[model_type]['final_loss']:.4f}, "
              f"Accuracy: {results[model_type]['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    for model_type, metrics in results.items():
        print(f"{model_type.upper():10s} - Loss: {metrics['final_loss']:.4f}, "
              f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    print("Deep Learning Models Examples")
    print("="*60)
    
    try:
        # Run individual examples
        example_lstm_model()
        example_gru_model()
        example_cnn_model()
        example_bilstm_model()
        
        # Compare models
        example_model_comparison()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure PyTorch is installed: pip install torch")
