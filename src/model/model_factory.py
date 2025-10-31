"""
Model Factory for creating different types of models.
Provides a unified interface for instantiating various model architectures.
"""

from typing import Dict, Any, Optional
from src.model.base_model import (
    BaseModel, ModelConfig, TraditionalMLModel, 
    DeepLearningModel, TransformerModel
)
import logging

logger = logging.getLogger(__name__)

# Lazy import for deep learning models to avoid dependency issues
def _import_deep_learning_models():
    """Lazy import of deep learning models."""
    try:
        from src.model.deep_learning_models import (
            LSTMModel, GRUModel, CNNModel, BiLSTMModel
        )
        return {
            'lstm': LSTMModel,
            'gru': GRUModel,
            'cnn': CNNModel,
            'bilstm': BiLSTMModel
        }
    except ImportError as e:
        logger.warning(f"Deep learning models not available: {e}")
        return {}


def _import_transformer_models():
    """Lazy import of transformer models."""
    try:
        from src.model.transformer_models import (
            BERTModel, RoBERTaModel, DistilBERTModel
        )
        return {
            'bert': BERTModel,
            'roberta': RoBERTaModel,
            'distilbert': DistilBERTModel
        }
    except ImportError as e:
        logger.warning(f"Transformer models not available: {e}")
        return {}


def _import_ensemble_models():
    """Lazy import of ensemble models."""
    try:
        from src.model.ensemble_models import (
            VotingEnsemble, StackingEnsemble, BlendingEnsemble
        )
        return {
            'voting': VotingEnsemble,
            'stacking': StackingEnsemble,
            'blending': BlendingEnsemble
        }
    except ImportError as e:
        logger.warning(f"Ensemble models not available: {e}")
        return {}


class LogisticRegressionModel(TraditionalMLModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression(
            **config.hyperparameters
        )


class RandomForestModel(TraditionalMLModel):
    """Random Forest model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            **config.hyperparameters
        )


class XGBoostModel(TraditionalMLModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**config.hyperparameters)
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")


class LightGBMModel(TraditionalMLModel):
    """LightGBM model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(**config.hyperparameters)
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")


class SVMModel(TraditionalMLModel):
    """Support Vector Machine model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.svm import SVC
        
        self.model = SVC(
            probability=True,  # Enable probability estimates
            **config.hyperparameters
        )


class ModelFactory:
    """
    Factory class for creating different types of models.
    Supports traditional ML, deep learning, and transformer models.
    """
    
    # Registry of available model types
    _model_registry = {
        # Traditional ML models
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'svm': SVMModel,
    }
    
    # Flags to track if models have been loaded
    _dl_models_loaded = False
    _transformer_models_loaded = False
    _ensemble_models_loaded = False
    
    @classmethod
    def _ensure_dl_models_loaded(cls):
        """Ensure deep learning models are loaded into registry."""
        if not cls._dl_models_loaded:
            dl_models = _import_deep_learning_models()
            cls._model_registry.update(dl_models)
            cls._dl_models_loaded = True
    
    @classmethod
    def _ensure_transformer_models_loaded(cls):
        """Ensure transformer models are loaded into registry."""
        if not cls._transformer_models_loaded:
            transformer_models = _import_transformer_models()
            cls._model_registry.update(transformer_models)
            cls._transformer_models_loaded = True
    
    @classmethod
    def _ensure_ensemble_models_loaded(cls):
        """Ensure ensemble models are loaded into registry."""
        if not cls._ensemble_models_loaded:
            ensemble_models = _import_ensemble_models()
            cls._model_registry.update(ensemble_models)
            cls._ensemble_models_loaded = True
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[ModelConfig] = None, 
                     **kwargs) -> BaseModel:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create (e.g., 'logistic_regression', 'xgboost')
            config: ModelConfig object with model parameters
            **kwargs: Additional parameters to override config
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If model_type is not supported
        """
        # Load deep learning, transformer, and ensemble models if needed
        cls._ensure_dl_models_loaded()
        cls._ensure_transformer_models_loaded()
        cls._ensure_ensemble_models_loaded()
        
        if model_type not in cls._model_registry:
            available_models = ', '.join(cls._model_registry.keys())
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Available models: {available_models}"
            )
        
        # Create default config if not provided
        if config is None:
            config = cls._create_default_config(model_type, **kwargs)
        
        # Override config with kwargs if provided
        if kwargs:
            config.hyperparameters.update(kwargs)
        
        model_class = cls._model_registry[model_type]
        logger.info(f"Creating model: {model_type} with config: {config}")
        
        return model_class(config)
    
    @classmethod
    def _create_default_config(cls, model_type: str, **kwargs) -> ModelConfig:
        """
        Create default configuration for a model type.
        
        Args:
            model_type: Type of model
            **kwargs: Additional hyperparameters
            
        Returns:
            ModelConfig with default parameters
        """
        default_configs = {
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42,
                'n_jobs': -1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'random_state': 42
            },
            'lstm': {
                'vocab_size': 10000,
                'num_classes': 3,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'n_layers': 2,
                'dropout': 0.5,
                'bidirectional': False,
                'use_attention': True,
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'gru': {
                'vocab_size': 10000,
                'num_classes': 3,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'n_layers': 2,
                'dropout': 0.5,
                'bidirectional': False,
                'use_attention': True,
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'cnn': {
                'vocab_size': 10000,
                'num_classes': 3,
                'embedding_dim': 128,
                'n_filters': 100,
                'filter_sizes': [3, 4, 5],
                'dropout': 0.5,
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'bilstm': {
                'vocab_size': 10000,
                'num_classes': 3,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'n_layers': 2,
                'dropout': 0.5,
                'bidirectional': True,
                'use_attention': True,
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'bert': {
                'num_classes': 3,
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'dropout': 0.1,
                'batch_size': 16,
                'epochs': 3,
                'learning_rate': 2e-5
            },
            'roberta': {
                'num_classes': 3,
                'model_name': 'roberta-base',
                'max_length': 512,
                'dropout': 0.1,
                'batch_size': 16,
                'epochs': 3,
                'learning_rate': 2e-5
            },
            'distilbert': {
                'num_classes': 3,
                'model_name': 'distilbert-base-uncased',
                'max_length': 512,
                'dropout': 0.1,
                'batch_size': 16,
                'epochs': 3,
                'learning_rate': 2e-5
            },
            'voting': {
                'voting_type': 'soft',
                'weights': None
            },
            'stacking': {
                'use_probabilities': True
            },
            'blending': {
                'blend_ratio': 0.2
            }
        }
        
        hyperparameters = default_configs.get(model_type, {})
        hyperparameters.update(kwargs)
        
        return ModelConfig(
            model_type=model_type,
            model_name=f"{model_type}_model",
            hyperparameters=hyperparameters,
            version="1.0.0"
        )
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a new model type in the factory.
        
        Args:
            model_type: Unique identifier for the model type
            model_class: Class that implements BaseModel interface
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel")
        
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered new model type: {model_type}")
    
    @classmethod
    def list_available_models(cls) -> list:
        """
        Get list of all available model types.
        
        Returns:
            List of model type identifiers
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Model type '{model_type}' not found")
        
        model_class = cls._model_registry[model_type]
        default_config = cls._create_default_config(model_type)
        
        return {
            'model_type': model_type,
            'model_class': model_class.__name__,
            'default_hyperparameters': default_config.hyperparameters,
            'base_class': model_class.__bases__[0].__name__
        }
