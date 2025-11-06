"""
Transformer-based models for sentiment analysis.
Implements BERT, RoBERTa, and DistilBERT fine-tuning using Hugging Face transformers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.models.base.base_model import TransformerModel, ModelConfig

logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """Wrapper for transformer models with classification head."""
    
    def __init__(self, transformer_model, num_classes: int, dropout: float = 0.1):
        super(TransformerClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout)
        
        # Get hidden size from transformer config
        hidden_size = transformer_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits


class BERTModel(TransformerModel):
    """BERT model for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        self.model_name = config.hyperparameters.get('model_name', 'bert-base-uncased')
        
    def _load_pretrained_model(self) -> nn.Module:
        """Load pre-trained BERT model."""
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Load pre-trained BERT
        bert_model = BertModel.from_pretrained(self.model_name)
        
        # Wrap with classification head
        dropout = self.config.hyperparameters.get('dropout', 0.1)
        model = TransformerClassifier(bert_model, self.num_classes, dropout)
        
        return model.to(self.device)
    
    def _tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs for BERT."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call _load_pretrained_model first.")
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        # X should be text strings or already tokenized
        if isinstance(X[0], str):
            # Tokenize texts
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            # Assume already tokenized
            input_ids = torch.LongTensor(X)
            attention_mask = torch.ones_like(input_ids)
        
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(input_ids, attention_mask, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 16)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the BERT model."""
        if self.model is None:
            self.model = self._load_pretrained_model()
        
        # Setup optimizer
        learning_rate = self.config.hyperparameters.get('learning_rate', 2e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Prepare data
        train_loader = self._prepare_data(X, y)
        
        # Training loop
        epochs = self.config.hyperparameters.get('epochs', 3)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "status": "success",
            "model_type": self.config.model_type,
            "epochs": epochs,
            "final_loss": avg_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        # Prepare input
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        # Prepare input
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint['model_name']
        
        self.model = self._load_pretrained_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class RoBERTaModel(TransformerModel):
    """RoBERTa model for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        self.model_name = config.hyperparameters.get('model_name', 'roberta-base')
        
    def _load_pretrained_model(self) -> nn.Module:
        """Load pre-trained RoBERTa model."""
        try:
            from transformers import RobertaModel, RobertaTokenizer
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        roberta_model = RobertaModel.from_pretrained(self.model_name)
        
        dropout = self.config.hyperparameters.get('dropout', 0.1)
        model = TransformerClassifier(roberta_model, self.num_classes, dropout)
        
        return model.to(self.device)
    
    def _tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs for RoBERTa."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            input_ids = torch.LongTensor(X)
            attention_mask = torch.ones_like(input_ids)
        
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(input_ids, attention_mask, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 16)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the RoBERTa model."""
        if self.model is None:
            self.model = self._load_pretrained_model()
        
        learning_rate = self.config.hyperparameters.get('learning_rate', 2e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        train_loader = self._prepare_data(X, y)
        epochs = self.config.hyperparameters.get('epochs', 3)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "status": "success",
            "model_type": self.config.model_type,
            "epochs": epochs,
            "final_loss": avg_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint['model_name']
        
        self.model = self._load_pretrained_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class DistilBERTModel(TransformerModel):
    """DistilBERT model for sentiment analysis (lighter and faster than BERT)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        self.model_name = config.hyperparameters.get('model_name', 'distilbert-base-uncased')
        
    def _load_pretrained_model(self) -> nn.Module:
        """Load pre-trained DistilBERT model."""
        try:
            from transformers import DistilBertModel, DistilBertTokenizer
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        distilbert_model = DistilBertModel.from_pretrained(self.model_name)
        
        dropout = self.config.hyperparameters.get('dropout', 0.1)
        model = TransformerClassifier(distilbert_model, self.num_classes, dropout)
        
        return model.to(self.device)
    
    def _tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs for DistilBERT."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            input_ids = torch.LongTensor(X)
            attention_mask = torch.ones_like(input_ids)
        
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(input_ids, attention_mask, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 16)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the DistilBERT model."""
        if self.model is None:
            self.model = self._load_pretrained_model()
        
        learning_rate = self.config.hyperparameters.get('learning_rate', 2e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        train_loader = self._prepare_data(X, y)
        epochs = self.config.hyperparameters.get('epochs', 3)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "status": "success",
            "model_type": self.config.model_type,
            "epochs": epochs,
            "final_loss": avg_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        
        if isinstance(X[0], str):
            tokenized = self._tokenize_text(X.tolist())
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
        else:
            input_ids = torch.LongTensor(X).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint['model_name']
        
        self.model = self._load_pretrained_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
