"""
Model module for the ML pipeline.
Provides base model interfaces, model factory, and configuration validation.
"""

from src.model.base_model import (
    BaseModel,
    ModelConfig,
    PredictionResult,
    TraditionalMLModel,
    DeepLearningModel,
    TransformerModel
)

from src.model.model_factory import ModelFactory

from src.model.config_validator import ConfigValidator

# Deep learning models are imported lazily through ModelFactory
# to avoid dependency issues if PyTorch is not installed

__all__ = [
    'BaseModel',
    'ModelConfig',
    'PredictionResult',
    'TraditionalMLModel',
    'DeepLearningModel',
    'TransformerModel',
    'ModelFactory',
    'ConfigValidator'
]
