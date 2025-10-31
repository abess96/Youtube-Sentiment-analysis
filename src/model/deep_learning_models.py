"""
Deep Learning models for sentiment analysis.
Implements LSTM, GRU, CNN, and BiLSTM architectures using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
import logging

from src.model.base_model import DeepLearningModel, ModelConfig

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int = 2, dropout: float = 0.5,
                 bidirectional: bool = False, use_attention: bool = True):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.use_attention = use_attention
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
    def attention_mechanism(self, lstm_output):
        """Apply attention mechanism to LSTM outputs."""
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        if self.use_attention:
            output = self.attention_mechanism(lstm_output)
        else:
            # Use last hidden state
            if self.lstm.bidirectional:
                output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                output = hidden[-1,:,:]
        
        output = self.dropout(output)
        return self.fc(output)


class GRUClassifier(nn.Module):
    """GRU-based text classifier with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int = 2, dropout: float = 0.5,
                 bidirectional: bool = False, use_attention: bool = True):
        super(GRUClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.use_attention = use_attention
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if use_attention:
            self.attention = nn.Linear(gru_output_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, output_dim)
        
    def attention_mechanism(self, gru_output):
        """Apply attention mechanism to GRU outputs."""
        attention_weights = torch.softmax(self.attention(gru_output), dim=1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        return context_vector
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        gru_output, hidden = self.gru(embedded)
        
        if self.use_attention:
            output = self.attention_mechanism(gru_output)
        else:
            if self.gru.bidirectional:
                output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                output = hidden[-1,:,:]
        
        output = self.dropout(output)
        return self.fc(output)


class CNNClassifier(nn.Module):
    """CNN-based text classifier with multiple filter sizes."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float = 0.5):
        super(CNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=n_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)  # [batch, embedding_dim, seq_len]
        
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class LSTMModel(DeepLearningModel):
    """LSTM model implementation for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = config.hyperparameters.get('vocab_size', 10000)
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        
    def _build_architecture(self) -> nn.Module:
        """Build LSTM architecture."""
        embedding_dim = self.config.hyperparameters.get('embedding_dim', 128)
        hidden_dim = self.config.hyperparameters.get('hidden_dim', 256)
        n_layers = self.config.hyperparameters.get('n_layers', 2)
        dropout = self.config.hyperparameters.get('dropout', 0.5)
        bidirectional = self.config.hyperparameters.get('bidirectional', False)
        use_attention = self.config.hyperparameters.get('use_attention', True)
        
        model = LSTMClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_classes,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention
        )
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 32)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the LSTM model."""
        if self.model is None:
            self.model = self._build_architecture()
        
        # Setup optimizer and loss
        learning_rate = self.config.hyperparameters.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Prepare data
        train_loader = self._prepare_data(X, y)
        
        # Training loop
        epochs = self.config.hyperparameters.get('epochs', 10)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 2 == 0:
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
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']
        self.num_classes = checkpoint['num_classes']
        
        self.model = self._build_architecture()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class GRUModel(DeepLearningModel):
    """GRU model implementation for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = config.hyperparameters.get('vocab_size', 10000)
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        
    def _build_architecture(self) -> nn.Module:
        """Build GRU architecture."""
        embedding_dim = self.config.hyperparameters.get('embedding_dim', 128)
        hidden_dim = self.config.hyperparameters.get('hidden_dim', 256)
        n_layers = self.config.hyperparameters.get('n_layers', 2)
        dropout = self.config.hyperparameters.get('dropout', 0.5)
        bidirectional = self.config.hyperparameters.get('bidirectional', False)
        use_attention = self.config.hyperparameters.get('use_attention', True)
        
        model = GRUClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_classes,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention
        )
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 32)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the GRU model."""
        if self.model is None:
            self.model = self._build_architecture()
        
        learning_rate = self.config.hyperparameters.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        train_loader = self._prepare_data(X, y)
        epochs = self.config.hyperparameters.get('epochs', 10)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 2 == 0:
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
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']
        self.num_classes = checkpoint['num_classes']
        
        self.model = self._build_architecture()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class CNNModel(DeepLearningModel):
    """CNN model implementation for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = config.hyperparameters.get('vocab_size', 10000)
        self.num_classes = config.hyperparameters.get('num_classes', 3)
        
    def _build_architecture(self) -> nn.Module:
        """Build CNN architecture."""
        embedding_dim = self.config.hyperparameters.get('embedding_dim', 128)
        n_filters = self.config.hyperparameters.get('n_filters', 100)
        filter_sizes = self.config.hyperparameters.get('filter_sizes', [3, 4, 5])
        dropout = self.config.hyperparameters.get('dropout', 0.5)
        
        model = CNNClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            filter_sizes=filter_sizes,
            output_dim=self.num_classes,
            dropout=dropout
        )
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Prepare data for training."""
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.config.hyperparameters.get('batch_size', 32)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the CNN model."""
        if self.model is None:
            self.model = self._build_architecture()
        
        learning_rate = self.config.hyperparameters.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        train_loader = self._prepare_data(X, y)
        epochs = self.config.hyperparameters.get('epochs', 10)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 2 == 0:
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
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']
        self.num_classes = checkpoint['num_classes']
        
        self.model = self._build_architecture()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class BiLSTMModel(LSTMModel):
    """Bidirectional LSTM model for sentiment analysis."""
    
    def __init__(self, config: ModelConfig):
        # Force bidirectional to True
        config.hyperparameters['bidirectional'] = True
        super().__init__(config)
