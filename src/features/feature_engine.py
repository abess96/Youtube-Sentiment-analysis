"""
Multi-Modal Feature Extraction Engine for Sentiment Analysis

This module implements advanced feature extraction techniques including:
- TF-IDF vectorization
- Word2Vec embeddings
- FastText embeddings
- Transformer-based embeddings (BERT, RoBERTa)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import pickle

# NLP and ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports for Word2Vec/FastText (not required for FeatureSelector)
try:
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None
    FastText = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalFeatureExtractor:
    """
    Multi-modal feature extraction engine that combines multiple text representation methods.
    
    Supports:
    - TF-IDF features
    - Word2Vec embeddings
    - FastText embeddings
    - Transformer-based embeddings (BERT, RoBERTa)
    """
    
    def __init__(
        self,
        tfidf_max_features: int = 5000,
        embedding_dim: int = 100,
        transformer_model: str = "bert-base-uncased",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the multi-modal feature extractor.
        
        Args:
            tfidf_max_features: Maximum number of TF-IDF features
            embedding_dim: Dimension for Word2Vec and FastText embeddings
            transformer_model: Name of the transformer model to use
            cache_dir: Directory to cache models and vectorizers
        """
        self.tfidf_max_features = tfidf_max_features
        self.embedding_dim = embedding_dim
        self.transformer_model = transformer_model
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/feature_extractors")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.word2vec_model: Optional[Word2Vec] = None
        self.fasttext_model: Optional[FastText] = None
        self.transformer_tokenizer: Optional[Any] = None
        self.transformer_model_obj: Optional[Any] = None
        
        # Device for transformer models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def extract_tfidf_features(
        self, 
        texts: List[str],
        fit: bool = False
    ) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer on the texts
            
        Returns:
            TF-IDF feature matrix
        """
        logger.info(f"Extracting TF-IDF features from {len(texts)} texts")
        
        if self.tfidf_vectorizer is None or fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
            self._save_tfidf_vectorizer()
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        logger.info(f"TF-IDF features shape: {features.shape}")
        return features.toarray()
    
    def extract_word2vec_features(
        self,
        texts: List[str],
        fit: bool = False,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4
    ) -> np.ndarray:
        """
        Extract Word2Vec embeddings from texts.
        
        Args:
            texts: List of text documents
            fit: Whether to train a new Word2Vec model
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            
        Returns:
            Word2Vec feature matrix (averaged word vectors)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2Vec features. Install with: pip install gensim")
        
        logger.info(f"Extracting Word2Vec features from {len(texts)} texts")
        
        # Tokenize texts
        tokenized_texts = [text.lower().split() for text in texts]
        
        if self.word2vec_model is None or fit:
            logger.info("Training Word2Vec model...")
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.embedding_dim,
                window=window,
                min_count=min_count,
                workers=workers,
                sg=1,  # Skip-gram
                epochs=10
            )
            self._save_word2vec_model()
        
        # Generate document embeddings by averaging word vectors
        features = []
        for tokens in tokenized_texts:
            word_vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[token])
            
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.embedding_dim)
            
            features.append(doc_vector)
        
        features_array = np.array(features)
        logger.info(f"Word2Vec features shape: {features_array.shape}")
        return features_array
    
    def extract_fasttext_features(
        self,
        texts: List[str],
        fit: bool = False,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4
    ) -> np.ndarray:
        """
        Extract FastText embeddings from texts.
        
        Args:
            texts: List of text documents
            fit: Whether to train a new FastText model
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            
        Returns:
            FastText feature matrix (averaged word vectors)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for FastText features. Install with: pip install gensim")
        
        logger.info(f"Extracting FastText features from {len(texts)} texts")
        
        # Tokenize texts
        tokenized_texts = [text.lower().split() for text in texts]
        
        if self.fasttext_model is None or fit:
            logger.info("Training FastText model...")
            self.fasttext_model = FastText(
                sentences=tokenized_texts,
                vector_size=self.embedding_dim,
                window=window,
                min_count=min_count,
                workers=workers,
                sg=1,  # Skip-gram
                epochs=10
            )
            self._save_fasttext_model()
        
        # Generate document embeddings by averaging word vectors
        features = []
        for tokens in tokenized_texts:
            word_vectors = []
            for token in tokens:
                # FastText can generate vectors for OOV words
                word_vectors.append(self.fasttext_model.wv[token])
            
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.embedding_dim)
            
            features.append(doc_vector)
        
        features_array = np.array(features)
        logger.info(f"FastText features shape: {features_array.shape}")
        return features_array
    
    def extract_transformer_features(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 128,
        pooling_strategy: str = "mean"
    ) -> np.ndarray:
        """
        Extract transformer-based embeddings (BERT, RoBERTa, etc.).
        
        Args:
            texts: List of text documents
            model_name: Name of the transformer model (overrides default)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            pooling_strategy: How to pool token embeddings ("mean", "cls", "max")
            
        Returns:
            Transformer feature matrix
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        model_name = model_name or self.transformer_model
        logger.info(f"Extracting transformer features using {model_name}")
        
        # Load model and tokenizer if not already loaded
        if self.transformer_tokenizer is None or self.transformer_model_obj is None:
            self._load_transformer_model(model_name)
        
        # Process texts in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.transformer_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {key: val.to(self.device) for key, val in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.transformer_model_obj(**encoded)
                
                # Apply pooling strategy
                if pooling_strategy == "cls":
                    # Use [CLS] token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling_strategy == "mean":
                    # Mean pooling over all tokens
                    attention_mask = encoded["attention_mask"].unsqueeze(-1)
                    masked_embeddings = outputs.last_hidden_state * attention_mask
                    sum_embeddings = torch.sum(masked_embeddings, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                elif pooling_strategy == "max":
                    # Max pooling over all tokens
                    embeddings = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                
                all_embeddings.append(embeddings)
        
        features_array = np.vstack(all_embeddings)
        logger.info(f"Transformer features shape: {features_array.shape}")
        return features_array
    
    def combine_features(
        self,
        feature_sets: List[np.ndarray],
        method: str = "concatenate"
    ) -> np.ndarray:
        """
        Combine multiple feature sets into a single feature matrix.
        
        Args:
            feature_sets: List of feature matrices to combine
            method: Combination method ("concatenate", "average", "weighted_average")
            
        Returns:
            Combined feature matrix
        """
        logger.info(f"Combining {len(feature_sets)} feature sets using {method}")
        
        if method == "concatenate":
            combined = np.hstack(feature_sets)
        elif method == "average":
            # Ensure all feature sets have the same shape
            combined = np.mean(feature_sets, axis=0)
        elif method == "weighted_average":
            # Simple weighted average (can be customized)
            weights = [1.0 / len(feature_sets)] * len(feature_sets)
            combined = np.average(feature_sets, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        logger.info(f"Combined features shape: {combined.shape}")
        return combined
    
    def extract_all_features(
        self,
        texts: List[str],
        feature_types: List[str] = ["tfidf", "word2vec", "fasttext", "transformer"],
        fit: bool = False,
        combine: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract all specified feature types from texts.
        
        Args:
            texts: List of text documents
            feature_types: List of feature types to extract
            fit: Whether to fit/train models on the texts
            combine: Whether to combine features into a single matrix
            
        Returns:
            Combined feature matrix or dictionary of feature matrices
        """
        logger.info(f"Extracting features: {feature_types}")
        
        features_dict = {}
        
        if "tfidf" in feature_types:
            features_dict["tfidf"] = self.extract_tfidf_features(texts, fit=fit)
        
        if "word2vec" in feature_types:
            features_dict["word2vec"] = self.extract_word2vec_features(texts, fit=fit)
        
        if "fasttext" in feature_types:
            features_dict["fasttext"] = self.extract_fasttext_features(texts, fit=fit)
        
        if "transformer" in feature_types:
            features_dict["transformer"] = self.extract_transformer_features(texts)
        
        if combine and len(features_dict) > 1:
            return self.combine_features(list(features_dict.values()))
        elif combine and len(features_dict) == 1:
            return list(features_dict.values())[0]
        else:
            return features_dict
    
    def _load_transformer_model(self, model_name: str):
        """Load transformer model and tokenizer."""
        logger.info(f"Loading transformer model: {model_name}")
        
        try:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model_obj = AutoModel.from_pretrained(model_name)
            self.transformer_model_obj.to(self.device)
            self.transformer_model_obj.eval()
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise
    
    def _save_tfidf_vectorizer(self):
        """Save TF-IDF vectorizer to disk."""
        path = self.cache_dir / "tfidf_vectorizer.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer to {path}")
    
    def _save_word2vec_model(self):
        """Save Word2Vec model to disk."""
        path = self.cache_dir / "word2vec_model.bin"
        self.word2vec_model.save(str(path))
        logger.info(f"Saved Word2Vec model to {path}")
    
    def _save_fasttext_model(self):
        """Save FastText model to disk."""
        path = self.cache_dir / "fasttext_model.bin"
        self.fasttext_model.save(str(path))
        logger.info(f"Saved FastText model to {path}")
    
    def load_tfidf_vectorizer(self, path: Optional[str] = None):
        """Load TF-IDF vectorizer from disk."""
        path = Path(path) if path else self.cache_dir / "tfidf_vectorizer.pkl"
        with open(path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        logger.info(f"Loaded TF-IDF vectorizer from {path}")
    
    def load_word2vec_model(self, path: Optional[str] = None):
        """Load Word2Vec model from disk."""
        path = Path(path) if path else self.cache_dir / "word2vec_model.bin"
        self.word2vec_model = Word2Vec.load(str(path))
        logger.info(f"Loaded Word2Vec model from {path}")
    
    def load_fasttext_model(self, path: Optional[str] = None):
        """Load FastText model from disk."""
        path = Path(path) if path else self.cache_dir / "fasttext_model.bin"
        self.fasttext_model = FastText.load(str(path))
        logger.info(f"Loaded FastText model from {path}")



class SentimentFeatureExtractor:
    """
    Extracts sentiment-specific features from text including:
    - Negation patterns and intensifiers
    - Emoji sentiment scoring
    - Punctuation and capitalization patterns
    """
    
    # Negation words that flip sentiment
    NEGATION_WORDS = {
        'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
        'none', 'nor', 'cannot', "can't", "won't", "shouldn't", "wouldn't",
        "couldn't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't", "shan't", "mightn't", "mustn't",
        "needn't", "oughtn't"
    }
    
    # Intensifier words that amplify sentiment
    INTENSIFIERS = {
        'very', 'extremely', 'absolutely', 'completely', 'totally', 'utterly',
        'highly', 'really', 'incredibly', 'amazingly', 'exceptionally', 'particularly',
        'especially', 'remarkably', 'extraordinarily', 'tremendously', 'enormously',
        'immensely', 'intensely', 'deeply', 'profoundly', 'seriously', 'genuinely',
        'truly', 'quite', 'rather', 'pretty', 'fairly', 'somewhat', 'super',
        'mega', 'ultra', 'so', 'too', 'enough'
    }
    
    # Diminisher words that reduce sentiment intensity
    DIMINISHERS = {
        'barely', 'hardly', 'scarcely', 'slightly', 'somewhat', 'little',
        'bit', 'kind of', 'sort of', 'kinda', 'sorta', 'almost', 'nearly',
        'practically', 'virtually', 'relatively', 'moderately', 'mildly'
    }
    
    # Emoji sentiment mappings (positive, negative, neutral)
    EMOJI_SENTIMENT = {
        # Positive emojis
        '😀': 0.8, '😃': 0.8, '😄': 0.9, '😁': 0.8, '😆': 0.7, '😅': 0.6,
        '😂': 0.9, '🤣': 0.9, '😊': 0.8, '😇': 0.7, '🙂': 0.6, '🙃': 0.5,
        '😉': 0.6, '😌': 0.6, '😍': 0.9, '🥰': 0.9, '😘': 0.8, '😗': 0.7,
        '😙': 0.7, '😚': 0.7, '😋': 0.7, '😛': 0.6, '😝': 0.6, '😜': 0.6,
        '🤪': 0.6, '🤗': 0.8, '🤩': 0.9, '🥳': 0.9, '😺': 0.8, '😸': 0.8,
        '😹': 0.8, '😻': 0.9, '😽': 0.7, '🙌': 0.8, '👏': 0.8, '🎉': 0.9,
        '🎊': 0.8, '❤️': 0.9, '💕': 0.8, '💖': 0.8, '💗': 0.8, '💙': 0.8,
        '💚': 0.8, '💛': 0.8, '🧡': 0.8, '💜': 0.8, '🤎': 0.7, '🖤': 0.6,
        '🤍': 0.7, '💯': 0.9, '✨': 0.7, '⭐': 0.7, '🌟': 0.8, '💫': 0.7,
        '👍': 0.8, '👌': 0.7, '✌️': 0.7, '🤞': 0.6, '🙏': 0.7, '👑': 0.8,
        
        # Negative emojis
        '😠': -0.8, '😡': -0.9, '🤬': -0.9, '😤': -0.7, '😒': -0.6, '🙄': -0.6,
        '😞': -0.7, '😔': -0.7, '😟': -0.7, '😕': -0.6, '🙁': -0.6, '☹️': -0.7,
        '😣': -0.7, '😖': -0.7, '😫': -0.8, '😩': -0.8, '🥺': -0.6, '😢': -0.8,
        '😭': -0.9, '😤': -0.7, '😰': -0.7, '😨': -0.7, '😱': -0.8, '😳': -0.5,
        '🤯': -0.7, '😬': -0.6, '🙃': -0.5, '😑': -0.5, '😐': -0.4, '😶': -0.4,
        '🤐': -0.5, '😷': -0.5, '🤢': -0.8, '🤮': -0.9, '🤧': -0.6, '😵': -0.7,
        '🥴': -0.6, '😪': -0.6, '😴': -0.4, '💔': -0.9, '💀': -0.7, '👎': -0.8,
        '🖕': -0.9, '😾': -0.8, '😿': -0.8, '🙀': -0.7, '💩': -0.8,
        
        # Neutral emojis
        '😐': 0.0, '😑': 0.0, '😶': 0.0, '🤔': 0.0, '🤨': 0.0, '🧐': 0.0,
        '😏': 0.0, '😪': 0.0, '😴': 0.0, '🤷': 0.0, '🙈': 0.0, '🙉': 0.0,
        '🙊': 0.0, '👀': 0.0, '💭': 0.0, '💬': 0.0
    }
    
    def __init__(self):
        """Initialize the sentiment feature extractor."""
        logger.info("Initialized SentimentFeatureExtractor")
    
    def extract_negation_features(self, text: str) -> Dict[str, Any]:
        """
        Extract negation-related features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing negation features
        """
        tokens = text.lower().split()
        
        # Count negation words
        negation_count = sum(1 for token in tokens if token in self.NEGATION_WORDS)
        
        # Detect negation patterns (negation word followed by sentiment word)
        negation_patterns = []
        for i in range(len(tokens) - 1):
            if tokens[i] in self.NEGATION_WORDS:
                # Look ahead up to 3 words for context
                context_window = tokens[i:min(i+4, len(tokens))]
                negation_patterns.append(' '.join(context_window))
        
        # Calculate negation density (negations per word)
        negation_density = negation_count / len(tokens) if tokens else 0.0
        
        return {
            'negation_count': negation_count,
            'negation_density': negation_density,
            'negation_patterns': negation_patterns,
            'has_negation': negation_count > 0
        }
    
    def extract_intensifier_features(self, text: str) -> Dict[str, Any]:
        """
        Extract intensifier and diminisher features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing intensifier features
        """
        tokens = text.lower().split()
        
        # Count intensifiers and diminishers
        intensifier_count = sum(1 for token in tokens if token in self.INTENSIFIERS)
        diminisher_count = sum(1 for token in tokens if token in self.DIMINISHERS)
        
        # Calculate densities
        intensifier_density = intensifier_count / len(tokens) if tokens else 0.0
        diminisher_density = diminisher_count / len(tokens) if tokens else 0.0
        
        # Net intensity (intensifiers - diminishers)
        net_intensity = intensifier_count - diminisher_count
        
        return {
            'intensifier_count': intensifier_count,
            'diminisher_count': diminisher_count,
            'intensifier_density': intensifier_density,
            'diminisher_density': diminisher_density,
            'net_intensity': net_intensity,
            'has_intensifiers': intensifier_count > 0,
            'has_diminishers': diminisher_count > 0
        }
    
    def extract_emoji_sentiment_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji sentiment features from text.
        
        Args:
            text: Input text containing emojis
            
        Returns:
            Dictionary containing emoji sentiment features
        """
        # Extract all emojis from text
        emojis = [char for char in text if char in self.EMOJI_SENTIMENT]
        
        if not emojis:
            return {
                'emoji_count': 0,
                'emoji_sentiment_score': 0.0,
                'positive_emoji_count': 0,
                'negative_emoji_count': 0,
                'neutral_emoji_count': 0,
                'emoji_sentiment_mean': 0.0,
                'emoji_sentiment_sum': 0.0,
                'has_emojis': False
            }
        
        # Calculate sentiment scores
        sentiment_scores = [self.EMOJI_SENTIMENT[emoji] for emoji in emojis]
        
        positive_count = sum(1 for score in sentiment_scores if score > 0.3)
        negative_count = sum(1 for score in sentiment_scores if score < -0.3)
        neutral_count = sum(1 for score in sentiment_scores if -0.3 <= score <= 0.3)
        
        return {
            'emoji_count': len(emojis),
            'emoji_sentiment_score': sum(sentiment_scores),
            'positive_emoji_count': positive_count,
            'negative_emoji_count': negative_count,
            'neutral_emoji_count': neutral_count,
            'emoji_sentiment_mean': np.mean(sentiment_scores),
            'emoji_sentiment_sum': sum(sentiment_scores),
            'has_emojis': True,
            'emoji_polarity': 'positive' if sum(sentiment_scores) > 0 else 'negative' if sum(sentiment_scores) < 0 else 'neutral'
        }
    
    def extract_punctuation_features(self, text: str) -> Dict[str, Any]:
        """
        Extract punctuation pattern features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing punctuation features
        """
        # Count different punctuation types
        exclamation_count = text.count('!')
        question_count = text.count('?')
        period_count = text.count('.')
        comma_count = text.count(',')
        ellipsis_count = text.count('...')
        
        # Multiple punctuation patterns (e.g., "!!!", "???")
        multiple_exclamation = len([m for m in text.split() if '!!' in m])
        multiple_question = len([m for m in text.split() if '??' in m])
        
        # Mixed punctuation (e.g., "?!", "!?")
        mixed_punctuation = text.count('?!') + text.count('!?')
        
        # Total punctuation count
        total_punctuation = exclamation_count + question_count + period_count + comma_count
        
        # Punctuation density
        text_length = len(text)
        punctuation_density = total_punctuation / text_length if text_length > 0 else 0.0
        
        return {
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'period_count': period_count,
            'comma_count': comma_count,
            'ellipsis_count': ellipsis_count,
            'multiple_exclamation': multiple_exclamation,
            'multiple_question': multiple_question,
            'mixed_punctuation': mixed_punctuation,
            'total_punctuation': total_punctuation,
            'punctuation_density': punctuation_density,
            'has_exclamation': exclamation_count > 0,
            'has_question': question_count > 0
        }
    
    def extract_capitalization_features(self, text: str) -> Dict[str, Any]:
        """
        Extract capitalization pattern features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing capitalization features
        """
        if not text:
            return {
                'all_caps_word_count': 0,
                'capitalized_word_count': 0,
                'all_caps_ratio': 0.0,
                'capitalized_ratio': 0.0,
                'has_all_caps': False,
                'has_title_case': False
            }
        
        words = text.split()
        
        # Count all-caps words (excluding single letters and common acronyms)
        all_caps_words = [w for w in words if w.isupper() and len(w) > 1]
        all_caps_count = len(all_caps_words)
        
        # Count capitalized words (first letter uppercase)
        capitalized_words = [w for w in words if w and w[0].isupper()]
        capitalized_count = len(capitalized_words)
        
        # Calculate ratios
        total_words = len(words)
        all_caps_ratio = all_caps_count / total_words if total_words > 0 else 0.0
        capitalized_ratio = capitalized_count / total_words if total_words > 0 else 0.0
        
        # Check if entire text is uppercase
        is_all_caps = text.isupper() and len(text) > 3
        
        # Check for title case
        is_title_case = text.istitle()
        
        return {
            'all_caps_word_count': all_caps_count,
            'capitalized_word_count': capitalized_count,
            'all_caps_ratio': all_caps_ratio,
            'capitalized_ratio': capitalized_ratio,
            'has_all_caps': all_caps_count > 0,
            'has_title_case': is_title_case,
            'is_shouting': is_all_caps  # All caps often indicates shouting/strong emotion
        }
    
    def extract_all_sentiment_features(self, text: str) -> Dict[str, Any]:
        """
        Extract all sentiment-specific features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing all sentiment features
        """
        features = {}
        
        # Extract all feature types
        features.update(self.extract_negation_features(text))
        features.update(self.extract_intensifier_features(text))
        features.update(self.extract_emoji_sentiment_features(text))
        features.update(self.extract_punctuation_features(text))
        features.update(self.extract_capitalization_features(text))
        
        return features
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract sentiment features from a batch of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame containing sentiment features for all texts
        """
        logger.info(f"Extracting sentiment features from {len(texts)} texts")
        
        features_list = []
        for text in texts:
            features = self.extract_all_sentiment_features(text)
            # Remove non-numeric features for DataFrame
            numeric_features = {k: v for k, v in features.items() 
                              if isinstance(v, (int, float, bool))}
            features_list.append(numeric_features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(df.columns)} sentiment features")
        return df
    
    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Get a numeric feature vector for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of sentiment features
        """
        features = self.extract_all_sentiment_features(text)
        # Extract only numeric values
        numeric_values = [v for v in features.values() 
                         if isinstance(v, (int, float, bool))]
        return np.array(numeric_values, dtype=float)


class FeatureSelector:
    """
    Feature selection and optimization module that implements multiple selection algorithms,
    feature importance ranking, visualization, and feature interaction detection.
    
    Supports:
    - Univariate feature selection (chi-square, ANOVA F-test, mutual information)
    - Model-based feature selection (tree-based importance, L1 regularization)
    - Recursive feature elimination (RFE)
    - Feature importance ranking and visualization
    - Feature combination and interaction detection
    """
    
    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        selection_method: str = "mutual_info",
        threshold: Optional[float] = None
    ):
        """
        Initialize the feature selector.
        
        Args:
            n_features_to_select: Number of top features to select (None = auto)
            selection_method: Feature selection method to use
            threshold: Threshold for feature selection (method-dependent)
        """
        self.n_features_to_select = n_features_to_select
        self.selection_method = selection_method
        self.threshold = threshold
        
        # Store feature scores and rankings
        self.feature_scores_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.selected_features_: Optional[List[int]] = None
        self.feature_importances_: Optional[Dict[str, float]] = None
        
        logger.info(f"Initialized FeatureSelector with method: {selection_method}")
    
    def select_features_univariate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "mutual_info",
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Select features using univariate statistical tests.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Statistical test method ("chi2", "f_classif", "mutual_info")
            k: Number of top features to select
            
        Returns:
            Indices of selected features
        """
        from sklearn.feature_selection import (
            chi2, f_classif, mutual_info_classif,
            SelectKBest
        )
        
        k = k or self.n_features_to_select or int(X.shape[1] * 0.5)
        
        logger.info(f"Selecting {k} features using {method}")
        
        # Choose scoring function
        if method == "chi2":
            # Ensure non-negative features for chi2
            X_positive = X - X.min() + 1e-10
            score_func = chi2
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X_positive, y)
        elif method == "f_classif":
            score_func = f_classif
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X, y)
        elif method == "mutual_info":
            score_func = mutual_info_classif
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X, y)
        else:
            raise ValueError(f"Unknown univariate method: {method}")
        
        # Store scores and selected features
        self.feature_scores_ = selector.scores_
        self.selected_features_ = selector.get_support(indices=True)
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self.selected_features_
    
    def select_features_model_based(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
        threshold: Optional[Union[str, float]] = "mean"
    ) -> np.ndarray:
        """
        Select features using model-based feature importance.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model ("random_forest", "gradient_boosting", "lasso")
            threshold: Threshold for feature selection ("mean", "median", or float)
            
        Returns:
            Indices of selected features
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import SelectFromModel
        
        logger.info(f"Selecting features using {model_type}")
        
        # Choose model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "lasso":
            model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model and select features
        selector = SelectFromModel(model, threshold=threshold, prefit=False)
        selector.fit(X, y)
        
        # Store feature importances
        if hasattr(selector.estimator_, "feature_importances_"):
            self.feature_scores_ = selector.estimator_.feature_importances_
        elif hasattr(selector.estimator_, "coef_"):
            self.feature_scores_ = np.abs(selector.estimator_.coef_).flatten()
        
        self.selected_features_ = selector.get_support(indices=True)
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self.selected_features_
    
    def select_features_rfe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: Optional[int] = None,
        step: int = 1
    ) -> np.ndarray:
        """
        Select features using Recursive Feature Elimination (RFE).
        
        Args:
            X: Feature matrix
            y: Target labels
            n_features: Number of features to select
            step: Number of features to remove at each iteration
            
        Returns:
            Indices of selected features
        """
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        n_features = n_features or self.n_features_to_select or int(X.shape[1] * 0.5)
        
        logger.info(f"Selecting {n_features} features using RFE")
        
        # Use Random Forest as the base estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Perform RFE
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        
        # Store rankings
        self.feature_scores_ = selector.ranking_
        self.selected_features_ = selector.get_support(indices=True)
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self.selected_features_
    
    def select_features_variance_threshold(
        self,
        X: np.ndarray,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Remove features with low variance.
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            Indices of selected features
        """
        from sklearn.feature_selection import VarianceThreshold
        
        logger.info(f"Removing features with variance < {threshold}")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        self.feature_scores_ = selector.variances_
        self.selected_features_ = selector.get_support(indices=True)
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self.selected_features_
    
    def rank_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "mutual_info",
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Rank all features by importance.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Ranking method
            feature_names: Names of features
            
        Returns:
            DataFrame with feature rankings
        """
        logger.info(f"Ranking features using {method}")
        
        # Select features to get scores
        if method in ["chi2", "f_classif", "mutual_info"]:
            self.select_features_univariate(X, y, method=method, k=X.shape[1])
        elif method in ["random_forest", "gradient_boosting", "lasso"]:
            self.select_features_model_based(X, y, model_type=method, threshold=-np.inf)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names_ = feature_names
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance_score': self.feature_scores_,
            'rank': np.argsort(-self.feature_scores_) + 1
        })
        
        ranking_df = ranking_df.sort_values('importance_score', ascending=False)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            ranking_df['feature_name'],
            ranking_df['importance_score']
        ))
        
        logger.info(f"Ranked {len(ranking_df)} features")
        return ranking_df
    
    def visualize_feature_importance(
        self,
        ranking_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Visualize feature importance rankings.
        
        Args:
            ranking_df: DataFrame with feature rankings
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info(f"Visualizing top {top_n} features")
        
        # Select top N features
        top_features = ranking_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=top_features,
            y='feature_name',
            x='importance_score',
            palette='viridis'
        )
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def detect_feature_interactions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_interactions: int = 10,
        method: str = "random_forest"
    ) -> List[Dict[str, Any]]:
        """
        Detect important feature interactions.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            max_interactions: Maximum number of interactions to return
            method: Method for detecting interactions
            
        Returns:
            List of feature interaction dictionaries
        """
        from sklearn.ensemble import RandomForestClassifier
        from itertools import combinations
        
        logger.info(f"Detecting feature interactions using {method}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        interactions = []
        
        if method == "random_forest":
            # Train a Random Forest to get feature importances
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            base_importances = rf.feature_importances_
            
            # Test pairwise interactions
            logger.info("Testing pairwise feature interactions...")
            
            # Get top features to test interactions
            top_features_idx = np.argsort(-base_importances)[:min(20, X.shape[1])]
            
            for i, j in combinations(top_features_idx, 2):
                # Create interaction feature
                X_interaction = X.copy()
                interaction_feature = X[:, i] * X[:, j]
                X_interaction = np.column_stack([X_interaction, interaction_feature])
                
                # Train model with interaction
                rf_interaction = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=42, 
                    n_jobs=-1
                )
                rf_interaction.fit(X_interaction, y)
                
                # Get importance of interaction feature
                interaction_importance = rf_interaction.feature_importances_[-1]
                
                # Calculate interaction strength
                individual_importance = base_importances[i] + base_importances[j]
                interaction_strength = interaction_importance / (individual_importance + 1e-10)
                
                interactions.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'feature_1_idx': i,
                    'feature_2_idx': j,
                    'interaction_importance': interaction_importance,
                    'interaction_strength': interaction_strength,
                    'individual_importance_sum': individual_importance
                })
        
        # Sort by interaction strength
        interactions = sorted(
            interactions,
            key=lambda x: x['interaction_strength'],
            reverse=True
        )[:max_interactions]
        
        logger.info(f"Found {len(interactions)} significant feature interactions")
        return interactions
    
    def create_interaction_features(
        self,
        X: np.ndarray,
        interactions: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create new features based on detected interactions.
        
        Args:
            X: Original feature matrix
            interactions: List of interaction dictionaries
            feature_names: Names of original features
            
        Returns:
            Tuple of (augmented feature matrix, augmented feature names)
        """
        logger.info(f"Creating {len(interactions)} interaction features")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        X_augmented = X.copy()
        augmented_names = feature_names.copy()
        
        for interaction in interactions:
            i = interaction['feature_1_idx']
            j = interaction['feature_2_idx']
            
            # Create interaction feature (multiplication)
            interaction_feature = X[:, i] * X[:, j]
            X_augmented = np.column_stack([X_augmented, interaction_feature])
            
            # Create interaction name
            interaction_name = f"{interaction['feature_1']}_x_{interaction['feature_2']}"
            augmented_names.append(interaction_name)
        
        logger.info(f"Augmented feature matrix shape: {X_augmented.shape}")
        return X_augmented, augmented_names
    
    def select_features_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: List[str] = ["mutual_info", "random_forest", "rfe"],
        voting: str = "union"
    ) -> np.ndarray:
        """
        Select features using ensemble of multiple methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            methods: List of selection methods to use
            voting: How to combine selections ("union", "intersection", "majority")
            
        Returns:
            Indices of selected features
        """
        logger.info(f"Ensemble feature selection with methods: {methods}")
        
        all_selections = []
        
        for method in methods:
            if method in ["chi2", "f_classif", "mutual_info"]:
                selected = self.select_features_univariate(X, y, method=method)
            elif method in ["random_forest", "gradient_boosting", "lasso"]:
                selected = self.select_features_model_based(X, y, model_type=method)
            elif method == "rfe":
                selected = self.select_features_rfe(X, y)
            else:
                logger.warning(f"Unknown method: {method}, skipping")
                continue
            
            all_selections.append(set(selected))
        
        # Combine selections based on voting strategy
        if voting == "union":
            selected_features = set.union(*all_selections)
        elif voting == "intersection":
            selected_features = set.intersection(*all_selections)
        elif voting == "majority":
            # Features selected by majority of methods
            feature_votes = {}
            for selection in all_selections:
                for feature in selection:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            threshold = len(methods) / 2
            selected_features = {
                f for f, votes in feature_votes.items() 
                if votes > threshold
            }
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")
        
        self.selected_features_ = np.array(sorted(selected_features))
        
        logger.info(f"Ensemble selected {len(self.selected_features_)} features")
        return self.selected_features_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix by selecting only the selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("No features selected. Call a selection method first.")
        
        return X[:, self.selected_features_]
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Fit the selector and transform the data in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Selection method (uses default if None)
            
        Returns:
            Transformed feature matrix
        """
        method = method or self.selection_method
        
        if method in ["chi2", "f_classif", "mutual_info"]:
            self.select_features_univariate(X, y, method=method)
        elif method in ["random_forest", "gradient_boosting", "lasso"]:
            self.select_features_model_based(X, y, model_type=method)
        elif method == "rfe":
            self.select_features_rfe(X, y)
        elif method == "variance":
            self.select_features_variance_threshold(X)
        elif method == "ensemble":
            self.select_features_ensemble(X, y)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return self.transform(X)


class EmbeddingGenerator:
    """
    Specialized class for generating and managing embeddings.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize the embedding generator.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.extractor = MultiModalFeatureExtractor(embedding_dim=embedding_dim)
    
    def generate_word2vec_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> np.ndarray:
        """Generate Word2Vec embeddings."""
        return self.extractor.extract_word2vec_features(texts, **kwargs)
    
    def generate_fasttext_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> np.ndarray:
        """Generate FastText embeddings."""
        return self.extractor.extract_fasttext_features(texts, **kwargs)
    
    def generate_transformer_embeddings(
        self,
        texts: List[str],
        model_name: str = "bert-base-uncased",
        **kwargs
    ) -> np.ndarray:
        """Generate transformer-based embeddings."""
        return self.extractor.extract_transformer_features(
            texts, 
            model_name=model_name,
            **kwargs
        )
    
    def generate_sentence_embeddings(
        self,
        texts: List[str],
        method: str = "transformer"
    ) -> np.ndarray:
        """
        Generate sentence-level embeddings using specified method.
        
        Args:
            texts: List of text documents
            method: Embedding method ("word2vec", "fasttext", "transformer")
            
        Returns:
            Sentence embeddings
        """
        if method == "word2vec":
            return self.generate_word2vec_embeddings(texts)
        elif method == "fasttext":
            return self.generate_fasttext_embeddings(texts)
        elif method == "transformer":
            return self.generate_transformer_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding method: {method}")


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This movie is absolutely amazing! I loved every minute of it. 😍🎉",
        "Terrible experience. Would NOT recommend to anyone. 😠👎",
        "It was okay, nothing special but not bad either. 🤷"
    ]
    
    # Initialize extractor
    extractor = MultiModalFeatureExtractor(
        tfidf_max_features=1000,
        embedding_dim=50,
        transformer_model="bert-base-uncased"
    )
    
    # Extract TF-IDF features
    tfidf_features = extractor.extract_tfidf_features(sample_texts, fit=True)
    print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # Extract Word2Vec features
    word2vec_features = extractor.extract_word2vec_features(sample_texts, fit=True)
    print(f"Word2Vec features shape: {word2vec_features.shape}")
    
    # Extract FastText features
    fasttext_features = extractor.extract_fasttext_features(sample_texts, fit=True)
    print(f"FastText features shape: {fasttext_features.shape}")
    
    # Extract transformer features
    transformer_features = extractor.extract_transformer_features(sample_texts)
    print(f"Transformer features shape: {transformer_features.shape}")
    
    # Extract and combine all features
    combined_features = extractor.extract_all_features(
        sample_texts,
        feature_types=["tfidf", "word2vec", "fasttext"],
        fit=True,
        combine=True
    )
    print(f"Combined features shape: {combined_features.shape}")
    
    # Test sentiment feature extraction
    print("\n" + "="*50)
    print("Sentiment Feature Extraction Examples")
    print("="*50)
    
    sentiment_extractor = SentimentFeatureExtractor()
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nText {i}: {text}")
        features = sentiment_extractor.extract_all_sentiment_features(text)
        print(f"  Negation count: {features['negation_count']}")
        print(f"  Intensifier count: {features['intensifier_count']}")
        print(f"  Emoji sentiment score: {features['emoji_sentiment_score']:.2f}")
        print(f"  Exclamation count: {features['exclamation_count']}")
        print(f"  All caps words: {features['all_caps_word_count']}")
    
    # Batch extraction
    print("\n" + "="*50)
    print("Batch Feature Extraction")
    print("="*50)
    sentiment_df = sentiment_extractor.extract_features_batch(sample_texts)
    print(f"Feature DataFrame shape: {sentiment_df.shape}")
    print(f"Features: {list(sentiment_df.columns)}")
    
    # Test feature selection
    print("\n" + "="*50)
    print("Feature Selection Examples")
    print("="*50)
    
    # Create synthetic data for demonstration
    from sklearn.datasets import make_classification
    X_demo, y_demo = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=15,
        n_redundant=10,
        random_state=42
    )
    
    # Initialize feature selector
    feature_selector = FeatureSelector(
        n_features_to_select=20,
        selection_method="mutual_info"
    )
    
    # Test univariate selection
    print("\n1. Univariate Feature Selection (Mutual Information):")
    selected_features = feature_selector.select_features_univariate(X_demo, y_demo, method="mutual_info", k=20)
    print(f"   Selected {len(selected_features)} features: {selected_features[:10]}...")
    
    # Test model-based selection
    print("\n2. Model-Based Feature Selection (Random Forest):")
    selected_features = feature_selector.select_features_model_based(X_demo, y_demo, model_type="random_forest")
    print(f"   Selected {len(selected_features)} features: {selected_features[:10]}...")
    
    # Test feature ranking
    print("\n3. Feature Ranking:")
    feature_names = [f"feature_{i}" for i in range(X_demo.shape[1])]
    ranking_df = feature_selector.rank_features(X_demo, y_demo, method="random_forest", feature_names=feature_names)
    print(f"   Top 10 features:")
    print(ranking_df.head(10)[['feature_name', 'importance_score', 'rank']])
    
    # Test feature interaction detection
    print("\n4. Feature Interaction Detection:")
    interactions = feature_selector.detect_feature_interactions(
        X_demo, y_demo, 
        feature_names=feature_names,
        max_interactions=5
    )
    print(f"   Found {len(interactions)} significant interactions:")
    for i, interaction in enumerate(interactions[:3], 1):
        print(f"   {i}. {interaction['feature_1']} x {interaction['feature_2']}: "
              f"strength={interaction['interaction_strength']:.4f}")
    
    # Test ensemble selection
    print("\n5. Ensemble Feature Selection:")
    selected_features = feature_selector.select_features_ensemble(
        X_demo, y_demo,
        methods=["mutual_info", "random_forest"],
        voting="intersection"
    )
    print(f"   Ensemble selected {len(selected_features)} features")
    
    # Test fit_transform
    print("\n6. Fit-Transform:")
    X_transformed = feature_selector.fit_transform(X_demo, y_demo, method="mutual_info")
    print(f"   Original shape: {X_demo.shape}, Transformed shape: {X_transformed.shape}")
