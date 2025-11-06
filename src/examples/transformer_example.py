"""
Example usage of transformer models for sentiment analysis.
Demonstrates BERT, RoBERTa, and DistilBERT fine-tuning.
"""

import numpy as np
from src.models.base.model_factory import ModelFactory
from src.models.base.base_model import ModelConfig


def create_sample_text_data(num_samples=100):
    """Create sample text data for testing."""
    # Sample sentiment texts
    positive_texts = [
        "This is amazing! I love it!",
        "Great product, highly recommend!",
        "Excellent quality and fast shipping.",
        "Best purchase I've made this year!",
        "Absolutely fantastic experience!"
    ]
    
    negative_texts = [
        "Terrible product, waste of money.",
        "Very disappointed with the quality.",
        "Would not recommend to anyone.",
        "Worst experience ever.",
        "Complete waste of time and money."
    ]
    
    neutral_texts = [
        "It's okay, nothing special.",
        "Average product, does the job.",
        "Neither good nor bad.",
        "It works as expected.",
        "Standard quality, no complaints."
    ]
    
    # Generate dataset
    texts = []
    labels = []
    
    for _ in range(num_samples // 3):
        texts.append(np.random.choice(positive_texts))
        labels.append(2)  # Positive
        
        texts.append(np.random.choice(negative_texts))
        labels.append(0)  # Negative
        
        texts.append(np.random.choice(neutral_texts))
        labels.append(1)  # Neutral
    
    return np.array(texts), np.array(labels)


def example_bert_model():
    """Example of fine-tuning BERT for sentiment analysis."""
    print("\n" + "="*60)
    print("BERT Model Example")
    print("="*60)
    
    # Create sample data
    X_train, y_train = create_sample_text_data(num_samples=60)
    X_test, y_test = create_sample_text_data(num_samples=15)
    
    # Create BERT model
    config = ModelConfig(
        model_type='bert',
        model_name='bert_sentiment',
        hyperparameters={
            'num_classes': 3,
            'model_name': 'bert-base-uncased',
            'max_length': 128,
            'dropout': 0.1,
            'batch_size': 8,
            'epochs': 2,
            'learning_rate': 2e-5
        }
    )
    
    print("Creating BERT model...")
    model = ModelFactory.create_model('bert', config=config)
    print(f"Model created: {model.config.model_name}")
    print(f"Device: {model.device}")
    
    # Train the model
    print("\nFine-tuning BERT model...")
    print("Note: This may take a few minutes...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Make predictions
    print("\nMaking predictions on test data...")
    test_samples = X_test[:5]
    predictions = model.predict(test_samples)
    probabilities = model.predict_proba(test_samples)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    for i, text in enumerate(test_samples):
        pred_label = sentiment_map[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        print(f"\nText: {text}")
        print(f"Prediction: {pred_label} (confidence: {confidence:.3f})")
        print(f"Probabilities: Neg={probabilities[i][0]:.3f}, "
              f"Neu={probabilities[i][1]:.3f}, Pos={probabilities[i][2]:.3f}")


def example_roberta_model():
    """Example of fine-tuning RoBERTa for sentiment analysis."""
    print("\n" + "="*60)
    print("RoBERTa Model Example")
    print("="*60)
    
    X_train, y_train = create_sample_text_data(num_samples=60)
    X_test, y_test = create_sample_text_data(num_samples=15)
    
    # Create RoBERTa model using kwargs
    print("Creating RoBERTa model...")
    model = ModelFactory.create_model(
        'roberta',
        num_classes=3,
        model_name='roberta-base',
        max_length=128,
        batch_size=8,
        epochs=2
    )
    
    print(f"Model created with {model.model_name}")
    
    # Train
    print("\nFine-tuning RoBERTa model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Predict
    test_samples = X_test[:3]
    predictions = model.predict(test_samples)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    print("\nPredictions:")
    for i, text in enumerate(test_samples):
        print(f"Text: {text} -> {sentiment_map[predictions[i]]}")


def example_distilbert_model():
    """Example of fine-tuning DistilBERT for sentiment analysis."""
    print("\n" + "="*60)
    print("DistilBERT Model Example (Faster & Lighter)")
    print("="*60)
    
    X_train, y_train = create_sample_text_data(num_samples=60)
    X_test, y_test = create_sample_text_data(num_samples=15)
    
    # Create DistilBERT model
    print("Creating DistilBERT model...")
    model = ModelFactory.create_model(
        'distilbert',
        num_classes=3,
        max_length=128,
        batch_size=8,
        epochs=2
    )
    
    print(f"Model created: {model.model_name}")
    print("DistilBERT is 40% smaller and 60% faster than BERT!")
    
    # Train
    print("\nFine-tuning DistilBERT model...")
    train_result = model.train(X_train, y_train)
    print(f"Training completed: {train_result}")
    
    # Predict with probabilities
    test_samples = X_test[:3]
    predictions = model.predict(test_samples)
    probabilities = model.predict_proba(test_samples)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    print("\nDetailed predictions:")
    for i, text in enumerate(test_samples):
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment_map[predictions[i]]}")
        print(f"Confidence scores: {probabilities[i]}")


def example_model_comparison():
    """Compare different transformer architectures."""
    print("\n" + "="*60)
    print("Transformer Model Comparison")
    print("="*60)
    
    X_train, y_train = create_sample_text_data(num_samples=60)
    X_test, y_test = create_sample_text_data(num_samples=30)
    
    model_types = ['bert', 'roberta', 'distilbert']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        try:
            model = ModelFactory.create_model(
                model_type,
                num_classes=3,
                max_length=128,
                batch_size=8,
                epochs=1  # Quick training for comparison
            )
            
            train_result = model.train(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == y_test)
            results[model_type] = {
                'final_loss': train_result.get('final_loss', 'N/A'),
                'accuracy': accuracy
            }
            
            print(f"{model_type.upper()} - Loss: {results[model_type]['final_loss']:.4f}, "
                  f"Accuracy: {results[model_type]['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    for model_type, metrics in results.items():
        if 'error' in metrics:
            print(f"{model_type.upper():12s} - Error: {metrics['error']}")
        else:
            print(f"{model_type.upper():12s} - Loss: {metrics['final_loss']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}")


def example_custom_texts():
    """Test models with custom text inputs."""
    print("\n" + "="*60)
    print("Custom Text Prediction Example")
    print("="*60)
    
    # Train a quick model
    X_train, y_train = create_sample_text_data(num_samples=60)
    
    print("Training DistilBERT for quick inference...")
    model = ModelFactory.create_model(
        'distilbert',
        num_classes=3,
        max_length=128,
        batch_size=8,
        epochs=1
    )
    
    model.train(X_train, y_train)
    
    # Test with custom texts
    custom_texts = np.array([
        "I absolutely love this product! It's the best!",
        "This is terrible, I want my money back.",
        "It's okay, nothing special about it.",
        "Wow! This exceeded all my expectations!",
        "Disappointed with the quality and service."
    ])
    
    predictions = model.predict(custom_texts)
    probabilities = model.predict_proba(custom_texts)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    print("\nPredictions on custom texts:")
    print("="*60)
    for i, text in enumerate(custom_texts):
        pred_label = sentiment_map[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        print(f"\nText: {text}")
        print(f"Sentiment: {pred_label} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    print("Transformer Models Examples")
    print("="*60)
    print("Note: These examples require 'transformers' and 'torch' libraries")
    print("Install with: pip install transformers torch")
    print("="*60)
    
    try:
        # Run individual examples
        # Uncomment the examples you want to run
        
        # example_bert_model()
        # example_roberta_model()
        example_distilbert_model()  # Fastest option
        # example_model_comparison()
        # example_custom_texts()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed:")
        print("  pip install transformers torch")
