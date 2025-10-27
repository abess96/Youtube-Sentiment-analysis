"""
Advanced Text Preprocessing Module

This module implements the AdvancedPreprocessor class with multiple preprocessing strategies
for YouTube sentiment analysis, including emoji handling, negation pattern detection,
and various cleaning techniques.
"""

import re
import string
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import emoji

# Configure logging
logger = logging.getLogger('advanced_preprocessing')
logger.setLevel('DEBUG')

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class PreprocessingMode(Enum):
    """Enumeration for different preprocessing strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


class AdvancedPreprocessor:
    """
    Advanced text preprocessor with multiple preprocessing strategies.
    
    Supports aggressive and conservative preprocessing modes, emoji handling,
    negation pattern detection, and various text cleaning techniques.
    """
    
    def __init__(self):
        """Initialize the AdvancedPreprocessor with required components."""
        self.lemmatizer = WordNetLemmatizer()
        
        # Define stopwords but preserve sentiment-important words
        self.stop_words = set(stopwords.words('english')) - {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'but', 'however', 'yet', 'although', 'though',
            'very', 'really', 'quite', 'rather', 'extremely', 'incredibly'
        }
        
        # Negation patterns for detection
        self.negation_patterns = [
            r'\b(?:not|no|never|nothing|nowhere|nobody|none|neither|nor)\b',
            r'\b(?:don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|cannot|couldn\'t|shouldn\'t|mustn\'t)\b',
            r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t)\b',
            r'\b(?:barely|hardly|scarcely|seldom|rarely)\b'
        ]
        
        # Intensifier patterns
        self.intensifier_patterns = [
            r'\b(?:very|really|extremely|incredibly|absolutely|totally|completely|utterly)\b',
            r'\b(?:quite|rather|pretty|fairly|somewhat|slightly|a bit|a little)\b',
            r'\b(?:super|mega|ultra|hyper|so|such|too|way)\b'
        ]
        
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # User mention patterns (YouTube, Twitter, etc.)
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Repeated character pattern (e.g., "sooooo" -> "so")
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
        logger.info("AdvancedPreprocessor initialized successfully")
    
    def handle_emojis(self, text: str, mode: str = "convert") -> str:
        """
        Handle emojis in text by converting them to text descriptions or removing them.
        
        Args:
            text: Input text containing emojis
            mode: Either "convert" to convert emojis to text or "remove" to remove them
            
        Returns:
            Text with emojis handled according to the specified mode
        """
        try:
            if mode == "convert":
                # Convert emojis to text descriptions
                text = emoji.demojize(text, delimiters=(" ", " "))
                # Clean up the emoji text format
                text = re.sub(r':[a-z_]+:', lambda m: m.group(0).replace('_', ' ').strip(':'), text)
            elif mode == "remove":
                # Remove all emojis
                text = emoji.replace_emoji(text, replace='')
            
            return text.strip()
        except Exception as e:
            logger.warning(f"Error handling emojis: {e}")
            return text
    
    def extract_negation_patterns(self, text: str) -> List[str]:
        """
        Extract negation patterns from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected negation patterns
        """
        negations = []
        text_lower = text.lower()
        
        for pattern in self.negation_patterns:
            matches = re.findall(pattern, text_lower)
            negations.extend(matches)
        
        return negations
    
    def detect_intensifiers(self, text: str) -> List[str]:
        """
        Detect intensifier words in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected intensifiers
        """
        intensifiers = []
        text_lower = text.lower()
        
        for pattern in self.intensifier_patterns:
            matches = re.findall(pattern, text_lower)
            intensifiers.extend(matches)
        
        return intensifiers
    
    def clean_urls_and_mentions(self, text: str) -> str:
        """
        Remove URLs, user mentions, hashtags, and email addresses from text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Text with URLs, mentions, hashtags, and emails removed
        """
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove user mentions
        text = self.mention_pattern.sub('', text)
        
        # Remove hashtags (optional - might want to keep for sentiment)
        text = self.hashtag_pattern.sub('', text)
        
        # Remove email addresses
        text = self.email_pattern.sub('', text)
        
        return text.strip()
    
    def normalize_repeated_characters(self, text: str) -> str:
        """
        Normalize repeated characters (e.g., "sooooo" -> "so").
        
        Args:
            text: Input text to normalize
            
        Returns:
            Text with repeated characters normalized
        """
        # Replace repeated characters with maximum of 2 repetitions
        return self.repeated_char_pattern.sub(r'\1\1', text)
    
    def preprocess_aggressive(self, text: str) -> str:
        """
        Apply aggressive preprocessing strategy.
        
        This mode applies extensive cleaning including:
        - Emoji conversion to text
        - URL and mention removal
        - Aggressive punctuation removal
        - Stopword removal
        - Lemmatization
        - Character normalization
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Aggressively preprocessed text
        """
        try:
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Handle emojis by converting to text
            text = self.handle_emojis(text, mode="convert")
            
            # Clean URLs, mentions, hashtags, emails
            text = self.clean_urls_and_mentions(text)
            
            # Normalize repeated characters
            text = self.normalize_repeated_characters(text)
            
            # Remove newlines and extra whitespace
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Remove most punctuation but keep sentence-ending punctuation
            text = re.sub(r'[^\w\s!?.]', '', text)
            
            # Remove extra punctuation (multiple consecutive punctuation marks)
            text = re.sub(r'[!?.]{2,}', '.', text)
            
            # Simple tokenization and remove stopwords
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple split if NLTK tokenizer fails
                tokens = text.split()
            
            tokens = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]
            
            # Lemmatize tokens
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except Exception:
                # Skip lemmatization if it fails
                pass
            
            # Join tokens back
            text = ' '.join(tokens)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in aggressive preprocessing: {e}")
            return text if isinstance(text, str) else ""
    
    def preprocess_conservative(self, text: str) -> str:
        """
        Apply conservative preprocessing strategy.
        
        This mode applies minimal cleaning to preserve original meaning:
        - Emoji conversion to text (preserving sentiment)
        - Basic URL removal
        - Minimal punctuation cleaning
        - Preserve negations and intensifiers
        - No stopword removal
        - No lemmatization
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Conservatively preprocessed text
        """
        try:
            if not isinstance(text, str):
                return ""
            
            # Handle emojis by converting to text (preserve sentiment)
            text = self.handle_emojis(text, mode="convert")
            
            # Remove only URLs and emails (keep mentions for context)
            text = self.url_pattern.sub('', text)
            text = self.email_pattern.sub('', text)
            
            # Normalize repeated characters but less aggressively
            text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
            
            # Clean up whitespace
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Remove only clearly non-informative characters
            text = re.sub(r'[^\w\s!?.,@#\'-]', '', text)
            
            # Preserve case for proper nouns and emphasis
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in conservative preprocessing: {e}")
            return text if isinstance(text, str) else ""
    
    def preprocess_text(self, text: str, mode: PreprocessingMode = PreprocessingMode.CONSERVATIVE) -> Dict[str, Any]:
        """
        Preprocess text using the specified mode and return detailed results.
        
        Args:
            text: Input text to preprocess
            mode: Preprocessing mode (AGGRESSIVE or CONSERVATIVE)
            
        Returns:
            Dictionary containing preprocessed text and extracted features
        """
        if not isinstance(text, str):
            return {
                'original_text': text,
                'preprocessed_text': "",
                'negation_patterns': [],
                'intensifiers': [],
                'preprocessing_mode': mode.value,
                'text_length_original': 0,
                'text_length_processed': 0
            }
        
        # Extract patterns before preprocessing
        negation_patterns = self.extract_negation_patterns(text)
        intensifiers = self.detect_intensifiers(text)
        
        # Apply preprocessing based on mode
        if mode == PreprocessingMode.AGGRESSIVE:
            preprocessed_text = self.preprocess_aggressive(text)
        else:
            preprocessed_text = self.preprocess_conservative(text)
        
        return {
            'original_text': text,
            'preprocessed_text': preprocessed_text,
            'negation_patterns': negation_patterns,
            'intensifiers': intensifiers,
            'preprocessing_mode': mode.value,
            'text_length_original': len(text),
            'text_length_processed': len(preprocessed_text)
        }
    
    def batch_preprocess(self, texts: List[str], mode: PreprocessingMode = PreprocessingMode.CONSERVATIVE) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            mode: Preprocessing mode to apply
            
        Returns:
            List of preprocessing results
        """
        results = []
        for text in texts:
            result = self.preprocess_text(text, mode)
            results.append(result)
        
        logger.info(f"Batch preprocessed {len(texts)} texts using {mode.value} mode")
        return results


def main():
    """Example usage of the AdvancedPreprocessor."""
    preprocessor = AdvancedPreprocessor()
    
    # Example texts with various challenges
    sample_texts = [
        "This movie is sooooo good! 😍 I love it!!! @user123 check this out #awesome",
        "I don't think this is not bad, but it's not great either... 🤔",
        "Visit https://example.com for more info! Email me at test@example.com",
        "AMAZING!!!! This is absolutely incredible and super fantastic!!!",
        "not good at all... really disappointing 😞"
    ]
    
    print("=== Conservative Preprocessing ===")
    for text in sample_texts:
        result = preprocessor.preprocess_text(text, PreprocessingMode.CONSERVATIVE)
        print(f"Original: {result['original_text']}")
        print(f"Processed: {result['preprocessed_text']}")
        print(f"Negations: {result['negation_patterns']}")
        print(f"Intensifiers: {result['intensifiers']}")
        print("-" * 50)
    
    print("\n=== Aggressive Preprocessing ===")
    for text in sample_texts:
        result = preprocessor.preprocess_text(text, PreprocessingMode.AGGRESSIVE)
        print(f"Original: {result['original_text']}")
        print(f"Processed: {result['preprocessed_text']}")
        print(f"Negations: {result['negation_patterns']}")
        print(f"Intensifiers: {result['intensifiers']}")
        print("-" * 50)


if __name__ == "__main__":
    main()