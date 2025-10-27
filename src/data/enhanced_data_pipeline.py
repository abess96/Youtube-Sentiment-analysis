"""
Enhanced Data Processing Pipeline

This module integrates the advanced data validation and preprocessing components
to create a comprehensive data processing pipeline with A/B testing capabilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import os
from pathlib import Path

from .data_quality_validator import DataQualityValidator
from .advanced_preprocessing import AdvancedPreprocessor, PreprocessingMode

# Configure logging
logger = logging.getLogger('enhanced_data_pipeline')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('enhanced_preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class DataProcessingStrategy(Enum):
    """Enumeration for different data processing strategies."""
    QUALITY_FIRST = "quality_first"
    SPEED_FIRST = "speed_first"
    BALANCED = "balanced"


class EnhancedDataPipeline:
    """
    Enhanced data processing pipeline that integrates data quality validation
    and advanced preprocessing with A/B testing capabilities.
    """
    
    def __init__(self, quality_threshold: float = 0.5, language_filter: str = 'en'):
        """
        Initialize the enhanced data pipeline.
        
        Args:
            quality_threshold: Minimum quality score for text acceptance (0-1)
            language_filter: Language code to filter for ('en' for English, None for all)
        """
        self.validator = DataQualityValidator()
        self.preprocessor = AdvancedPreprocessor()
        self.quality_threshold = quality_threshold
        self.language_filter = language_filter
        
        logger.info(f"Enhanced data pipeline initialized with quality threshold: {quality_threshold}")
    
    def process_dataset(
        self, 
        data: pd.DataFrame, 
        text_column: str = 'clean_comment',
        strategy: DataProcessingStrategy = DataProcessingStrategy.BALANCED,
        enable_ab_testing: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete dataset with quality validation and preprocessing.
        
        Args:
            data: Input DataFrame containing text data
            text_column: Name of the column containing text to process
            strategy: Processing strategy to use
            enable_ab_testing: Whether to create A/B test splits
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Processing dataset with {len(data)} records using {strategy.value} strategy")
        
        if text_column not in data.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset")
        
        # Step 1: Data Quality Validation
        logger.info("Step 1: Performing data quality validation...")
        quality_results = self._validate_dataset_quality(data, text_column)
        
        # Step 2: Filter data based on quality and language
        logger.info("Step 2: Filtering data based on quality criteria...")
        filtered_data = self._filter_data(data, text_column, quality_results)
        
        # Step 3: Apply preprocessing based on strategy
        logger.info("Step 3: Applying preprocessing...")
        if enable_ab_testing:
            processed_results = self._process_with_ab_testing(filtered_data, text_column, strategy)
        else:
            processed_results = self._process_single_strategy(filtered_data, text_column, strategy)
        
        # Step 4: Generate processing report
        logger.info("Step 4: Generating processing report...")
        report = self._generate_processing_report(data, filtered_data, processed_results, quality_results)
        
        return {
            'processed_data': processed_results,
            'quality_report': report,
            'original_count': len(data),
            'filtered_count': len(filtered_data),
            'processing_strategy': strategy.value,
            'ab_testing_enabled': enable_ab_testing
        }
    
    def _validate_dataset_quality(self, data: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Validate quality of entire dataset."""
        quality_scores = []
        language_detections = []
        spam_detections = []
        
        for idx, text in enumerate(data[text_column]):
            if pd.isna(text) or not isinstance(text, str):
                quality_scores.append(0.0)
                language_detections.append('unknown')
                spam_detections.append(True)
                continue
            
            # Quality validation
            quality_metrics = self.validator.validate_text_quality(text)
            quality_scores.append(quality_metrics['overall_quality'])
            
            # Language detection
            language = self.validator.detect_language(text)
            language_detections.append(language)
            
            # Spam detection
            is_spam = self.validator.check_spam_indicators(text)
            spam_detections.append(is_spam)
            
            if (idx + 1) % 1000 == 0:
                logger.debug(f"Processed {idx + 1} records for quality validation")
        
        return {
            'quality_scores': quality_scores,
            'languages': language_detections,
            'spam_flags': spam_detections
        }
    
    def _filter_data(self, data: pd.DataFrame, text_column: str, quality_results: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on quality criteria."""
        # Create filter mask
        quality_mask = np.array(quality_results['quality_scores']) >= self.quality_threshold
        
        if self.language_filter:
            language_mask = np.array(quality_results['languages']) == self.language_filter
        else:
            language_mask = np.ones(len(data), dtype=bool)
        
        spam_mask = ~np.array(quality_results['spam_flags'])
        
        # Combine all filters
        final_mask = quality_mask & language_mask & spam_mask
        
        filtered_data = data[final_mask].copy()
        
        logger.info(f"Filtered dataset: {len(data)} -> {len(filtered_data)} records")
        logger.info(f"Quality filter removed: {np.sum(~quality_mask)} records")
        logger.info(f"Language filter removed: {np.sum(~language_mask)} records")
        logger.info(f"Spam filter removed: {np.sum(~spam_mask)} records")
        
        return filtered_data
    
    def _process_single_strategy(
        self, 
        data: pd.DataFrame, 
        text_column: str, 
        strategy: DataProcessingStrategy
    ) -> pd.DataFrame:
        """Process data using a single preprocessing strategy."""
        # Determine preprocessing mode based on strategy
        if strategy == DataProcessingStrategy.QUALITY_FIRST:
            mode = PreprocessingMode.CONSERVATIVE
        elif strategy == DataProcessingStrategy.SPEED_FIRST:
            mode = PreprocessingMode.AGGRESSIVE
        else:  # BALANCED
            mode = PreprocessingMode.CONSERVATIVE
        
        processed_data = data.copy()
        processed_texts = []
        preprocessing_metadata = []
        
        for text in data[text_column]:
            result = self.preprocessor.preprocess_text(text, mode)
            processed_texts.append(result['preprocessed_text'])
            preprocessing_metadata.append({
                'negation_patterns': result['negation_patterns'],
                'intensifiers': result['intensifiers'],
                'original_length': result['text_length_original'],
                'processed_length': result['text_length_processed']
            })
        
        processed_data['processed_text'] = processed_texts
        processed_data['preprocessing_metadata'] = preprocessing_metadata
        processed_data['preprocessing_mode'] = mode.value
        
        return processed_data
    
    def _process_with_ab_testing(
        self, 
        data: pd.DataFrame, 
        text_column: str, 
        strategy: DataProcessingStrategy
    ) -> Dict[str, pd.DataFrame]:
        """Process data with A/B testing using different preprocessing modes."""
        # Split data for A/B testing
        np.random.seed(42)  # For reproducible splits
        split_mask = np.random.random(len(data)) < 0.5
        
        group_a = data[split_mask].copy()
        group_b = data[~split_mask].copy()
        
        # Process Group A with Conservative mode
        logger.info("Processing Group A with Conservative preprocessing...")
        group_a_processed = []
        group_a_metadata = []
        
        for text in group_a[text_column]:
            result = self.preprocessor.preprocess_text(text, PreprocessingMode.CONSERVATIVE)
            group_a_processed.append(result['preprocessed_text'])
            group_a_metadata.append({
                'negation_patterns': result['negation_patterns'],
                'intensifiers': result['intensifiers'],
                'original_length': result['text_length_original'],
                'processed_length': result['text_length_processed']
            })
        
        group_a['processed_text'] = group_a_processed
        group_a['preprocessing_metadata'] = group_a_metadata
        group_a['preprocessing_mode'] = 'conservative'
        group_a['ab_group'] = 'A'
        
        # Process Group B with Aggressive mode
        logger.info("Processing Group B with Aggressive preprocessing...")
        group_b_processed = []
        group_b_metadata = []
        
        for text in group_b[text_column]:
            result = self.preprocessor.preprocess_text(text, PreprocessingMode.AGGRESSIVE)
            group_b_processed.append(result['preprocessed_text'])
            group_b_metadata.append({
                'negation_patterns': result['negation_patterns'],
                'intensifiers': result['intensifiers'],
                'original_length': result['text_length_original'],
                'processed_length': result['text_length_processed']
            })
        
        group_b['processed_text'] = group_b_processed
        group_b['preprocessing_metadata'] = group_b_metadata
        group_b['preprocessing_mode'] = 'aggressive'
        group_b['ab_group'] = 'B'
        
        return {
            'group_a': group_a,
            'group_b': group_b,
            'combined': pd.concat([group_a, group_b], ignore_index=True)
        }
    
    def _generate_processing_report(
        self, 
        original_data: pd.DataFrame, 
        filtered_data: pd.DataFrame, 
        processed_results: Any,
        quality_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        # Basic statistics
        original_count = len(original_data)
        filtered_count = len(filtered_data)
        filter_rate = (original_count - filtered_count) / original_count if original_count > 0 else 0
        
        # Quality statistics
        quality_scores = quality_results['quality_scores']
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Language distribution
        language_dist = {}
        for lang in quality_results['languages']:
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # Spam statistics
        spam_count = sum(quality_results['spam_flags'])
        spam_rate = spam_count / original_count if original_count > 0 else 0
        
        report = {
            'processing_summary': {
                'original_records': original_count,
                'filtered_records': filtered_count,
                'filter_rate': filter_rate,
                'records_processed': filtered_count
            },
            'quality_metrics': {
                'average_quality_score': avg_quality,
                'quality_threshold': self.quality_threshold,
                'low_quality_filtered': sum(1 for score in quality_scores if score < self.quality_threshold)
            },
            'language_distribution': language_dist,
            'spam_detection': {
                'spam_records_detected': spam_count,
                'spam_rate': spam_rate
            }
        }
        
        # Add A/B testing specific metrics if applicable
        if isinstance(processed_results, dict) and 'group_a' in processed_results:
            report['ab_testing'] = {
                'group_a_size': len(processed_results['group_a']),
                'group_b_size': len(processed_results['group_b']),
                'group_a_mode': 'conservative',
                'group_b_mode': 'aggressive'
            }
        
        return report
    
    def save_processed_data(
        self, 
        processed_results: Dict[str, Any], 
        output_dir: str = './data/interim',
        filename_prefix: str = 'enhanced_processed'
    ) -> None:
        """Save processed data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_data = processed_results['processed_data']
        
        if isinstance(processed_data, dict) and 'group_a' in processed_data:
            # Save A/B testing results
            processed_data['group_a'].to_csv(
                output_path / f'{filename_prefix}_group_a.csv', 
                index=False
            )
            processed_data['group_b'].to_csv(
                output_path / f'{filename_prefix}_group_b.csv', 
                index=False
            )
            processed_data['combined'].to_csv(
                output_path / f'{filename_prefix}_combined.csv', 
                index=False
            )
            logger.info(f"A/B testing data saved to {output_path}")
        else:
            # Save single strategy results
            processed_data.to_csv(
                output_path / f'{filename_prefix}.csv', 
                index=False
            )
            logger.info(f"Processed data saved to {output_path}")
        
        # Save quality report
        import json
        with open(output_path / f'{filename_prefix}_report.json', 'w') as f:
            json.dump(processed_results['quality_report'], f, indent=2)
        
        logger.info("Processing report saved")


def main():
    """Example usage of the enhanced data pipeline."""
    try:
        logger.info("Starting enhanced data processing pipeline...")
        
        # Initialize pipeline
        pipeline = EnhancedDataPipeline(quality_threshold=0.6, language_filter='en')
        
        # Load data
        train_data = pd.read_csv('./data/raw/train.csv')
        logger.info(f"Loaded {len(train_data)} training records")
        
        # Process with A/B testing
        results = pipeline.process_dataset(
            train_data, 
            text_column='clean_comment',
            strategy=DataProcessingStrategy.BALANCED,
            enable_ab_testing=True
        )
        
        # Save results
        pipeline.save_processed_data(results, filename_prefix='train_enhanced')
        
        # Print summary
        report = results['quality_report']
        print("\n=== Enhanced Data Processing Summary ===")
        print(f"Original records: {report['processing_summary']['original_records']}")
        print(f"Filtered records: {report['processing_summary']['filtered_records']}")
        print(f"Filter rate: {report['processing_summary']['filter_rate']:.2%}")
        print(f"Average quality score: {report['quality_metrics']['average_quality_score']:.3f}")
        print(f"Spam rate: {report['spam_detection']['spam_rate']:.2%}")
        
        if 'ab_testing' in report:
            print(f"A/B Group A size: {report['ab_testing']['group_a_size']}")
            print(f"A/B Group B size: {report['ab_testing']['group_b_size']}")
        
        logger.info("Enhanced data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in enhanced data processing: {e}")
        raise


if __name__ == "__main__":
    main()