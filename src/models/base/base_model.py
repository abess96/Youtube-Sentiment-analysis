"""
Base model interfaces for the ML pipeline.
Provides abstract base classes for all model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for model initialization and training."""
    model_type: str
    model_name: str
    hyperparameters: Dict[str, Any]
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Structured prediction result with metadata."""
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """Abstract base class for all models in the pipeline."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model with configuration.
        
        Args:
            config: ModelConfig object containing model parameters
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and metadata
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of class probabilities
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: File path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: File path to load the model from
        """
        pass
    
    def get_config(self) -> ModelConfig:
        """Get the model configuration."""
        return self.config
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get the feature names used by the model."""
        return self.feature_names
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """Set the feature names for the model."""
        self.feature_names = feature_names


class TraditionalMLModel(BaseModel):
    """Base class for traditional ML models (sklearn-based)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the traditional ML model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return {
            "status": "success",
            "model_type": self.config.model_type,
            "training_samples": len(X)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """Save the model using joblib."""
        import joblib
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """Load the model using joblib."""
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True


class DeepLearningModel(BaseModel):
    """Base class for deep learning models (PyTorch/TensorFlow)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = None
        self.optimizer = None
        self.loss_fn = None
        
    @abstractmethod
    def _build_architecture(self) -> Any:
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Prepare data for training (batching, tensors, etc.)."""
        pass


class TransformerModel(BaseModel):
    """Base class for transformer-based models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self.max_length = config.hyperparameters.get('max_length', 512)
        
    @abstractmethod
    def _load_pretrained_model(self) -> Any:
        """Load the pre-trained transformer model."""
        pass
    
    @abstractmethod
    def _tokenize_text(self, texts: List[str]) -> Any:
        """Tokenize text inputs for the transformer."""
        pass
