
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import yaml
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from pathlib import Path

# Import enhanced pipeline components
try:
    from src.data.enhanced_data_pipeline import EnhancedDataPipeline, DataProcessingStrategy
    from src.data.data_quality_validator import DataQualityValidator
    from src.data.advanced_preprocessing import AdvancedPreprocessor, PreprocessingMode
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    try:
        from .enhanced_data_pipeline import EnhancedDataPipeline, DataProcessingStrategy
        from .data_quality_validator import DataQualityValidator
        from .advanced_preprocessing import AdvancedPreprocessor, PreprocessingMode
        ENHANCED_PIPELINE_AVAILABLE = True
    except ImportError:
        ENHANCED_PIPELINE_AVAILABLE = False
        print("Enhanced pipeline components not available, falling back to legacy preprocessing")

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def load_params():
    """Load parameters from params.yaml file."""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params.get('data_preprocessing', {})
    except Exception as e:
        logger.warning(f"Could not load params.yaml: {e}. Using default parameters.")
        return {}

def run_enhanced_preprocessing(train_data, test_data, params):
    """Run enhanced preprocessing pipeline with MLflow integration."""
    logger.info("Using enhanced data preprocessing pipeline")
    
    # Initialize enhanced pipeline with parameters
    quality_threshold = params.get('quality_threshold', 0.6)
    language_filter = params.get('language_filter', 'en')
    processing_strategy = params.get('processing_strategy', 'balanced')
    enable_ab_testing = params.get('enable_ab_testing', False)
    enable_mlflow = params.get('enable_mlflow_logging', True)
    
    # Map string strategy to enum
    strategy_map = {
        'quality_first': DataProcessingStrategy.QUALITY_FIRST,
        'speed_first': DataProcessingStrategy.SPEED_FIRST,
        'balanced': DataProcessingStrategy.BALANCED
    }
    strategy = strategy_map.get(processing_strategy, DataProcessingStrategy.BALANCED)
    
    # Initialize pipeline with MLflow integration
    pipeline = EnhancedDataPipeline(
        quality_threshold=quality_threshold,
        language_filter=language_filter,
        enable_mlflow=enable_mlflow
    )
    
    # Process training data
    logger.info("Processing training data with enhanced pipeline...")
    train_results = pipeline.process_dataset(
        train_data,
        text_column='clean_comment',
        strategy=strategy,
        enable_ab_testing=enable_ab_testing,
        run_name="data_preprocessing_train"
    )
    
    # Process test data (no A/B testing for test data)
    logger.info("Processing test data with enhanced pipeline...")
    test_results = pipeline.process_dataset(
        test_data,
        text_column='clean_comment',
        strategy=strategy,
        enable_ab_testing=False,
        run_name="data_preprocessing_test"
    )
    
    # Save enhanced results (will also log artifacts to MLflow if enabled)
    save_enhanced_data(train_results, test_results, './data')
    
    return train_results, test_results

def save_enhanced_data(train_results, test_results, data_path):
    """Save enhanced preprocessing results."""
    interim_data_path = os.path.join(data_path, 'interim')
    os.makedirs(interim_data_path, exist_ok=True)
    
    # Save training data
    train_processed = train_results['processed_data']
    if isinstance(train_processed, dict) and 'combined' in train_processed:
        # A/B testing results
        train_processed['combined'].to_csv(
            os.path.join(interim_data_path, "train_processed.csv"), 
            index=False
        )
        # Save A/B groups separately for analysis
        train_processed['group_a'].to_csv(
            os.path.join(interim_data_path, "train_processed_group_a.csv"), 
            index=False
        )
        train_processed['group_b'].to_csv(
            os.path.join(interim_data_path, "train_processed_group_b.csv"), 
            index=False
        )
    else:
        # Single strategy results
        train_processed.to_csv(
            os.path.join(interim_data_path, "train_processed.csv"), 
            index=False
        )
    
    # Save test data
    test_processed = test_results['processed_data']
    test_processed.to_csv(
        os.path.join(interim_data_path, "test_processed.csv"), 
        index=False
    )
    
    # Save quality reports
    with open(os.path.join(interim_data_path, "train_quality_report.json"), 'w') as f:
        json.dump(train_results['quality_report'], f, indent=2)
    
    with open(os.path.join(interim_data_path, "test_quality_report.json"), 'w') as f:
        json.dump(test_results['quality_report'], f, indent=2)
    
    logger.info(f"Enhanced processed data saved to {interim_data_path}")

def run_legacy_preprocessing(train_data, test_data):
    """Run legacy preprocessing pipeline."""
    logger.info("Using legacy data preprocessing pipeline")
    
    # Preprocess the data using legacy method
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    
    # Save the processed data using legacy method
    save_data(train_processed_data, test_processed_data, data_path='./data')
    
    return train_processed_data, test_processed_data

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Load parameters
        params = load_params()
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')
        
        # Check if enhanced pipeline should be used
        use_enhanced = params.get('enable_enhanced_pipeline', True) and ENHANCED_PIPELINE_AVAILABLE
        
        if use_enhanced:
            train_results, test_results = run_enhanced_preprocessing(train_data, test_data, params)
            
            # Print summary statistics
            train_report = train_results['quality_report']
            test_report = test_results['quality_report']
            
            print("\n=== Enhanced Preprocessing Summary ===")
            print(f"Training data: {train_report['processing_summary']['original_records']} -> {train_report['processing_summary']['filtered_records']} records")
            print(f"Test data: {test_report['processing_summary']['original_records']} -> {test_report['processing_summary']['filtered_records']} records")
            print(f"Training filter rate: {train_report['processing_summary']['filter_rate']:.2%}")
            print(f"Test filter rate: {test_report['processing_summary']['filter_rate']:.2%}")
            print(f"Training avg quality: {train_report['quality_metrics']['average_quality_score']:.3f}")
            print(f"Test avg quality: {test_report['quality_metrics']['average_quality_score']:.3f}")
            
            if 'ab_testing' in train_report:
                print(f"A/B Testing - Group A: {train_report['ab_testing']['group_a_size']} records")
                print(f"A/B Testing - Group B: {train_report['ab_testing']['group_b_size']} records")
        else:
            train_processed_data, test_processed_data = run_legacy_preprocessing(train_data, test_data)
            print(f"\n=== Legacy Preprocessing Summary ===")
            print(f"Training data processed: {len(train_processed_data)} records")
            print(f"Test data processed: {len(test_processed_data)} records")
        
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
