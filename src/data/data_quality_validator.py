"""
Data Quality Validation Module

This module provides comprehensive data quality validation for text data,
including language detection, spam detection, and various text quality metrics.

Example usage:
    from src.data.data_quality_validator import DataQualityValidator
    
    validator = DataQualityValidator()
    
    # Validate text quality
    quality_metrics = validator.validate_text_quality("This is a sample comment.")
    print(f"Overall quality: {quality_metrics['overall_quality']}")
    
    # Detect language
    language = validator.detect_language("This is an English comment.")
    print(f"Detected language: {language}")
    
    # Check for spam
    is_spam = validator.check_spam_indicators("BUY NOW! Click here for deals!")
    print(f"Is spam: {is_spam}")
    
    # Get detailed spam analysis
    spam_analysis = validator.get_detailed_spam_analysis("Subscribe to my channel!")
    print(f"Spam confidence: {spam_analysis['confidence']}")
"""

import re
import string
from typing import Dict, List, Optional, Tuple
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import numpy as np


class DataQualityValidator:
    """
    A comprehensive data quality validator for text data with focus on 
    YouTube comments and social media content.
    """
    
    def __init__(self):
        """Initialize the DataQualityValidator with predefined patterns and thresholds."""
        # Set seed for consistent language detection results
        DetectorFactory.seed = 0
        
        # Spam detection patterns
        self.spam_patterns = [
            r'(?i)\b(buy|purchase|sale|discount|offer|deal|free|win|winner|prize)\b.*\b(now|today|click|link|visit)\b',
            r'(?i)\b(viagra|cialis|pharmacy|pills|medication)\b',
            r'(?i)\b(earn|money|cash|income|profit)\b.*\b(home|online|easy|fast)\b',
            r'(?i)\b(subscribe|follow|like|share)\b.*\b(channel|page|account)\b',
            r'(?i)\b(check|visit|go to)\b.*\b(my|our)\b.*\b(channel|website|page|profile)\b',
            r'(?i)\b(first|early|new)\b.*\b(comment|viewer|subscriber)\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_spam_patterns = [re.compile(pattern) for pattern in self.spam_patterns]
        
        # Quality thresholds
        self.min_length = 3
        self.max_length = 5000
        self.max_repetition_ratio = 0.7
        self.max_caps_ratio = 0.8
        self.max_special_char_ratio = 0.5
        
    def validate_text_quality(self, text: str) -> Dict[str, float]:
        """
        Validate text quality and return comprehensive quality metrics.
        
        Args:
            text (str): Input text to validate
            
        Returns:
            Dict[str, float]: Dictionary containing quality metrics:
                - length_score: Score based on text length (0-1)
                - readability_score: Basic readability score (0-1)
                - repetition_score: Score based on character repetition (0-1)
                - caps_score: Score based on capitalization patterns (0-1)
                - special_char_score: Score based on special character usage (0-1)
                - overall_quality: Weighted average of all scores (0-1)
        """
        if not text or not isinstance(text, str):
            return self._get_zero_quality_scores()
        
        text = text.strip()
        if not text:
            return self._get_zero_quality_scores()
        
        # Calculate individual quality metrics
        length_score = self._calculate_length_score(text)
        readability_score = self._calculate_readability_score(text)
        repetition_score = self._calculate_repetition_score(text)
        caps_score = self._calculate_caps_score(text)
        special_char_score = self._calculate_special_char_score(text)
        
        # Calculate weighted overall quality score
        weights = {
            'length': 0.2,
            'readability': 0.3,
            'repetition': 0.2,
            'caps': 0.15,
            'special_char': 0.15
        }
        
        overall_quality = (
            weights['length'] * length_score +
            weights['readability'] * readability_score +
            weights['repetition'] * repetition_score +
            weights['caps'] * caps_score +
            weights['special_char'] * special_char_score
        )
        
        return {
            'length_score': length_score,
            'readability_score': readability_score,
            'repetition_score': repetition_score,
            'caps_score': caps_score,
            'special_char_score': special_char_score,
            'overall_quality': overall_quality
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text for language detection
            
        Returns:
            str: Detected language code (e.g., 'en', 'es', 'fr') or 'unknown' if detection fails
        """
        if not text or not isinstance(text, str):
            return 'unknown'
        
        # Clean text for better language detection
        cleaned_text = self._clean_text_for_language_detection(text)
        
        if len(cleaned_text.strip()) < 3:
            return 'unknown'
        
        # Check for repetitive text that might confuse language detection
        if self._is_repetitive_text(cleaned_text):
            return 'unknown'
        
        try:
            detected_lang = detect(cleaned_text)
            # Additional validation for common false positives
            if self._validate_language_detection(cleaned_text, detected_lang):
                return detected_lang
            else:
                return 'unknown'
        except LangDetectException:
            return 'unknown'
    
    def check_spam_indicators(self, text: str) -> bool:
        """
        Check if text contains spam indicators based on predefined patterns.
        
        Args:
            text (str): Input text to check for spam
            
        Returns:
            bool: True if spam indicators are found, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        
        # Check against spam patterns
        for pattern in self.compiled_spam_patterns:
            if pattern.search(text):
                return True
        
        # Additional spam indicators
        spam_indicators = [
            self._check_excessive_repetition(text),
            self._check_excessive_caps(text),
            self._check_suspicious_urls(text),
            self._check_excessive_special_chars(text),
            self._check_promotional_keywords(text_lower),
            self._is_repetitive_text(text)  # Add repetitive text check
        ]
        
        # Return True if multiple indicators are present or if highly repetitive
        return sum(spam_indicators) >= 2 or self._is_repetitive_text(text)
    
    def get_detailed_spam_analysis(self, text: str) -> Dict[str, any]:
        """
        Get detailed spam analysis with individual indicator scores.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, any]: Detailed spam analysis results
        """
        if not text or not isinstance(text, str):
            return {'is_spam': False, 'confidence': 0.0, 'indicators': {}}
        
        text_lower = text.lower()
        
        indicators = {
            'pattern_matches': sum(1 for pattern in self.compiled_spam_patterns if pattern.search(text)),
            'excessive_repetition': self._check_excessive_repetition(text),
            'excessive_caps': self._check_excessive_caps(text),
            'suspicious_urls': self._check_suspicious_urls(text),
            'excessive_special_chars': self._check_excessive_special_chars(text),
            'promotional_keywords': self._check_promotional_keywords(text_lower),
            'repetitive_text': self._is_repetitive_text(text)
        }
        
        # Calculate spam confidence score
        total_indicators = sum(indicators.values())
        confidence = min(total_indicators / 3.0, 1.0)  # Normalize to 0-1
        
        is_spam = confidence > 0.5
        
        return {
            'is_spam': is_spam,
            'confidence': confidence,
            'indicators': indicators
        }
    
    def _get_zero_quality_scores(self) -> Dict[str, float]:
        """Return zero scores for invalid input."""
        return {
            'length_score': 0.0,
            'readability_score': 0.0,
            'repetition_score': 0.0,
            'caps_score': 0.0,
            'special_char_score': 0.0,
            'overall_quality': 0.0
        }
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length."""
        length = len(text)
        
        if length < self.min_length:
            return 0.0
        elif length > self.max_length:
            return 0.3  # Very long texts get low score
        elif self.min_length <= length <= 100:
            return 1.0  # Optimal length
        elif 100 < length <= 500:
            return 0.8  # Good length
        else:
            return 0.6  # Acceptable length
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate basic readability score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Basic readability metrics
        avg_word_length = sum(len(word.strip(string.punctuation)) for word in words) / len(words)
        sentence_count = max(1, len(re.findall(r'[.!?]+', text)))
        avg_sentence_length = len(words) / sentence_count
        
        # Score based on reasonable ranges
        word_length_score = 1.0 if 3 <= avg_word_length <= 7 else 0.5
        sentence_length_score = 1.0 if 5 <= avg_sentence_length <= 20 else 0.5
        
        return (word_length_score + sentence_length_score) / 2
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate score based on character repetition."""
        if len(text) < 3:
            return 1.0
        
        # Count consecutive repeated characters
        repetition_count = 0
        for i in range(len(text) - 2):
            if text[i] == text[i + 1] == text[i + 2]:
                repetition_count += 1
        
        repetition_ratio = repetition_count / len(text)
        
        if repetition_ratio > self.max_repetition_ratio:
            return 0.0
        else:
            return 1.0 - (repetition_ratio / self.max_repetition_ratio)
    
    def _calculate_caps_score(self, text: str) -> float:
        """Calculate score based on capitalization patterns."""
        if not text:
            return 1.0
        
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 1.0
        
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        
        if caps_ratio > self.max_caps_ratio:
            return 0.0
        else:
            return 1.0 - (caps_ratio / self.max_caps_ratio)
    
    def _calculate_special_char_score(self, text: str) -> float:
        """Calculate score based on special character usage."""
        if not text:
            return 1.0
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / len(text)
        
        if special_char_ratio > self.max_special_char_ratio:
            return 0.0
        else:
            return 1.0 - (special_char_ratio / self.max_special_char_ratio)
    
    def _clean_text_for_language_detection(self, text: str) -> str:
        """Clean text for better language detection."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _check_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        if len(text) < 10:
            return False
        
        repetition_count = 0
        for i in range(len(text) - 2):
            if text[i] == text[i + 1] == text[i + 2]:
                repetition_count += 1
        
        return (repetition_count / len(text)) > 0.3
    
    def _check_excessive_caps(self, text: str) -> bool:
        """Check for excessive capitalization."""
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) < 5:
            return False
        
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        return caps_ratio > 0.7
    
    def _check_suspicious_urls(self, text: str) -> bool:
        """Check for suspicious URL patterns."""
        url_patterns = [
            r'bit\.ly',
            r'tinyurl',
            r'goo\.gl',
            r't\.co',
            r'short\.link'
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for multiple URLs
        url_count = len(re.findall(r'http[s]?://\S+', text))
        return url_count > 2
    
    def _check_excessive_special_chars(self, text: str) -> bool:
        """Check for excessive special characters."""
        if not text:
            return False
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return (special_chars / len(text)) > 0.4
    
    def _check_promotional_keywords(self, text_lower: str) -> bool:
        """Check for promotional keywords."""
        promotional_keywords = [
            'subscribe', 'follow', 'like and subscribe', 'check out my',
            'visit my channel', 'free money', 'easy money', 'work from home',
            'make money online', 'click here', 'limited time', 'act now',
            'special offer', 'exclusive deal'
        ]
        
        return any(keyword in text_lower for keyword in promotional_keywords)
    
    def _is_repetitive_text(self, text: str) -> bool:
        """Check if text is highly repetitive."""
        if len(text) < 5:
            return False
        
        # Check for repeated characters
        unique_chars = len(set(text.lower().replace(' ', '')))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return True
        
        # If less than 30% unique characters, consider it repetitive
        return (unique_chars / total_chars) < 0.3
    
    def _validate_language_detection(self, text: str, detected_lang: str) -> bool:
        """Validate language detection results for common false positives."""
        # For very short text, be more conservative
        if len(text.split()) < 3:
            # Only accept English for very short text if it contains common English words
            if detected_lang == 'en':
                common_english_words = ['the', 'and', 'is', 'to', 'a', 'in', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also', 'back', 'after', 'use', 'work', 'life', 'only', 'new', 'way', 'may', 'say']
                text_words = text.lower().split()
                return any(word in common_english_words for word in text_words)
            else:
                return False
        
        return True