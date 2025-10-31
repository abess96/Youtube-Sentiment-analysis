"""
Ensemble models for sentiment analysis.
Implements voting, stacking, and blending ensemble methods.
"""

import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

from src.model.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.
    Supports both hard voting (majority vote) and soft voting (average probabilities).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models: List[BaseModel] = []
        self.voting_type = config.hyperparameters.get('voting_type', 'soft')
        self.weights = config.hyperparameters.get('weights', None)
        
    def add_model(self, model: BaseModel) -> None:
        """Add a base model to the ensemble."""
        self.base_models.append(model)
        logger.info(f"Added model to ensemble: {model.config.model_name}")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train all base models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results for each model
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added to ensemble. Use add_model() first.")
        
        results = {}
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.config.model_name}")
            
            try:
                train_result = model.train(X, y, **kwargs)
                results[f"model_{i}_{model.config.model_name}"] = train_result
            except Exception as e:
                logger.error(f"Error training model {model.config.model_name}: {e}")
                results[f"model_{i}_{model.config.model_name}"] = {"status": "failed", "error": str(e)}
        
        self.is_trained = True
        
        return {
            "status": "success",
            "ensemble_type": "voting",
            "voting_type": self.voting_type,
            "num_models": len(self.base_models),
            "model_results": results
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble voting.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        if self.voting_type == 'hard':
            return self._hard_voting_predict(X)
        else:
            return self._soft_voting_predict(X)
    
    def _hard_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Hard voting: majority vote from all models."""
        predictions = []
        
        for model in self.base_models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {model.config.model_name} prediction failed: {e}")
        
        if len(predictions) == 0:
            raise ValueError("All models failed to make predictions.")
        
        # Stack predictions and take majority vote
        predictions = np.array(predictions)
        
        # Apply weights if provided
        if self.weights is not None:
            weighted_predictions = []
            for i in range(X.shape[0]):
                votes = []
                for j, pred in enumerate(predictions[:, i]):
                    weight = self.weights[j] if j < len(self.weights) else 1.0
                    votes.extend([pred] * int(weight * 10))
                weighted_predictions.append(Counter(votes).most_common(1)[0][0])
            return np.array(weighted_predictions)
        else:
            # Simple majority vote
            final_predictions = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                final_predictions.append(Counter(votes).most_common(1)[0][0])
            return np.array(final_predictions)
    
    def _soft_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: average probabilities from all models."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted average.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        all_probas = []
        
        for model in self.base_models:
            try:
                proba = model.predict_proba(X)
                all_probas.append(proba)
            except Exception as e:
                logger.warning(f"Model {model.config.model_name} probability prediction failed: {e}")
        
        if len(all_probas) == 0:
            raise ValueError("All models failed to predict probabilities.")
        
        # Average probabilities with optional weights
        if self.weights is not None:
            weights = np.array(self.weights[:len(all_probas)])
            weights = weights / weights.sum()
            weighted_probas = np.average(all_probas, axis=0, weights=weights)
            return weighted_probas
        else:
            return np.mean(all_probas, axis=0)
    
    def save(self, path: str) -> None:
        """Save the ensemble and all base models."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble.")
        
        ensemble_data = {
            'config': self.config,
            'voting_type': self.voting_type,
            'weights': self.weights,
            'num_models': len(self.base_models)
        }
        
        # Save ensemble metadata
        joblib.dump(ensemble_data, f"{path}_ensemble.pkl")
        
        # Save each base model
        for i, model in enumerate(self.base_models):
            model.save(f"{path}_model_{i}.pkl")
        
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the ensemble and all base models."""
        # Load ensemble metadata
        ensemble_data = joblib.load(f"{path}_ensemble.pkl")
        self.config = ensemble_data['config']
        self.voting_type = ensemble_data['voting_type']
        self.weights = ensemble_data['weights']
        
        # Load base models
        num_models = ensemble_data['num_models']
        self.base_models = []
        
        for i in range(num_models):
            # Note: This requires knowing the model type
            # In practice, you'd save model type info and recreate appropriately
            model_path = f"{path}_model_{i}.pkl"
            # Placeholder - actual implementation would need model type info
            logger.warning(f"Loading model {i} from {model_path} - implement model recreation")
        
        self.is_trained = True
        logger.info(f"Ensemble loaded from {path}")


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble that uses a meta-model to combine base model predictions.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models: List[BaseModel] = []
        self.meta_model: Optional[BaseModel] = None
        self.use_probabilities = config.hyperparameters.get('use_probabilities', True)
        
    def add_base_model(self, model: BaseModel) -> None:
        """Add a base model to the ensemble."""
        self.base_models.append(model)
        logger.info(f"Added base model: {model.config.model_name}")
    
    def set_meta_model(self, model: BaseModel) -> None:
        """Set the meta-model for stacking."""
        self.meta_model = model
        logger.info(f"Set meta-model: {model.config.model_name}")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the stacking ensemble.
        
        1. Train all base models
        2. Generate meta-features from base model predictions
        3. Train meta-model on meta-features
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added. Use add_base_model() first.")
        
        if self.meta_model is None:
            raise ValueError("No meta-model set. Use set_meta_model() first.")
        
        # Step 1: Train base models
        logger.info("Training base models...")
        base_results = {}
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.config.model_name}")
            train_result = model.train(X, y, **kwargs)
            base_results[f"base_model_{i}"] = train_result
        
        # Step 2: Generate meta-features
        logger.info("Generating meta-features...")
        meta_features = self._generate_meta_features(X)
        
        # Step 3: Train meta-model
        logger.info("Training meta-model...")
        meta_result = self.meta_model.train(meta_features, y)
        
        self.is_trained = True
        
        return {
            "status": "success",
            "ensemble_type": "stacking",
            "num_base_models": len(self.base_models),
            "base_model_results": base_results,
            "meta_model_result": meta_result
        }
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features_list = []
        
        for model in self.base_models:
            if self.use_probabilities:
                # Use probability predictions as meta-features
                proba = model.predict_proba(X)
                meta_features_list.append(proba)
            else:
                # Use class predictions as meta-features
                pred = model.predict(X).reshape(-1, 1)
                meta_features_list.append(pred)
        
        # Concatenate all meta-features
        meta_features = np.hstack(meta_features_list)
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to make final prediction
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the stacking ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to predict probabilities
        return self.meta_model.predict_proba(meta_features)
    
    def save(self, path: str) -> None:
        """Save the stacking ensemble."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble.")
        
        ensemble_data = {
            'config': self.config,
            'use_probabilities': self.use_probabilities,
            'num_base_models': len(self.base_models)
        }
        
        joblib.dump(ensemble_data, f"{path}_stacking.pkl")
        
        # Save base models
        for i, model in enumerate(self.base_models):
            model.save(f"{path}_base_{i}.pkl")
        
        # Save meta-model
        self.meta_model.save(f"{path}_meta.pkl")
        
        logger.info(f"Stacking ensemble saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the stacking ensemble."""
        ensemble_data = joblib.load(f"{path}_stacking.pkl")
        self.config = ensemble_data['config']
        self.use_probabilities = ensemble_data['use_probabilities']
        
        # Load models (placeholder - needs proper implementation)
        logger.warning("Load functionality needs model type information for proper reconstruction")
        
        self.is_trained = True
        logger.info(f"Stacking ensemble loaded from {path}")


class BlendingEnsemble(BaseModel):
    """
    Blending ensemble that uses a holdout validation set to train the meta-model.
    Similar to stacking but uses a separate validation set instead of cross-validation.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models: List[BaseModel] = []
        self.meta_model: Optional[BaseModel] = None
        self.blend_ratio = config.hyperparameters.get('blend_ratio', 0.2)
        
    def add_base_model(self, model: BaseModel) -> None:
        """Add a base model to the ensemble."""
        self.base_models.append(model)
    
    def set_meta_model(self, model: BaseModel) -> None:
        """Set the meta-model for blending."""
        self.meta_model = model
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the blending ensemble.
        
        1. Split data into train and blend sets
        2. Train base models on train set
        3. Generate predictions on blend set
        4. Train meta-model on blend set predictions
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        if len(self.base_models) == 0 or self.meta_model is None:
            raise ValueError("Base models and meta-model must be set.")
        
        # Split data
        split_idx = int(len(X) * (1 - self.blend_ratio))
        X_train, X_blend = X[:split_idx], X[split_idx:]
        y_train, y_blend = y[:split_idx], y[split_idx:]
        
        logger.info(f"Split data: {len(X_train)} train, {len(X_blend)} blend")
        
        # Train base models on train set
        logger.info("Training base models...")
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            model.train(X_train, y_train, **kwargs)
        
        # Generate predictions on blend set
        logger.info("Generating blend predictions...")
        blend_features = []
        for model in self.base_models:
            proba = model.predict_proba(X_blend)
            blend_features.append(proba)
        
        blend_features = np.hstack(blend_features)
        
        # Train meta-model on blend set
        logger.info("Training meta-model...")
        meta_result = self.meta_model.train(blend_features, y_blend)
        
        self.is_trained = True
        
        return {
            "status": "success",
            "ensemble_type": "blending",
            "num_base_models": len(self.base_models),
            "train_size": len(X_train),
            "blend_size": len(X_blend),
            "meta_model_result": meta_result
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the blending ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # Get predictions from base models
        blend_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)
            blend_features.append(proba)
        
        blend_features = np.hstack(blend_features)
        
        # Use meta-model for final prediction
        return self.meta_model.predict(blend_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the blending ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # Get predictions from base models
        blend_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)
            blend_features.append(proba)
        
        blend_features = np.hstack(blend_features)
        
        # Use meta-model for probability prediction
        return self.meta_model.predict_proba(blend_features)
    
    def save(self, path: str) -> None:
        """Save the blending ensemble."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble.")
        
        ensemble_data = {
            'config': self.config,
            'blend_ratio': self.blend_ratio,
            'num_base_models': len(self.base_models)
        }
        
        joblib.dump(ensemble_data, f"{path}_blending.pkl")
        
        for i, model in enumerate(self.base_models):
            model.save(f"{path}_base_{i}.pkl")
        
        self.meta_model.save(f"{path}_meta.pkl")
        
        logger.info(f"Blending ensemble saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the blending ensemble."""
        ensemble_data = joblib.load(f"{path}_blending.pkl")
        self.config = ensemble_data['config']
        self.blend_ratio = ensemble_data['blend_ratio']
        
        self.is_trained = True
        logger.info(f"Blending ensemble loaded from {path}")
