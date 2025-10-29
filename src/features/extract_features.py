"""
Feature Extraction Pipeline with MLflow Integration
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engine import MultiModalFeatureExtractor, SentimentFeatureExtractor
from utils.mlflow_config import setup_mlflow, get_or_create_experiment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params.get('feature_extraction', {})

def main():
    """Main feature extraction pipeline"""
    try:
        # Setup MLflow
        setup_mlflow()
        experiment_id = get_or_create_experiment("feature-engineering-pipeline")
        
        # Load parameters
        params = load_params()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="feature_extraction"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tag("pipeline_stage", "feature_extraction")
            
            # Load processed data
            train_data = pd.read_csv('./data/interim/train_processed.csv')
            test_data = pd.read_csv('./data/interim/test_processed.csv')
            
            logger.info(f"Loaded {len(train_data)} train and {len(test_data)} test samples")
            mlflow.log_metric("train_samples", len(train_data))
            mlflow.log_metric("test_samples", len(test_data))
            
            # Initialize extractors
            multimodal_extractor = MultiModalFeatureExtractor(
                tfidf_max_features=params.get('tfidf_max_features', 5000),
                embedding_dim=params.get('embedding_dim', 100)
            )
            
            sentiment_extractor = SentimentFeatureExtractor()
            
            # Extract TF-IDF features
            logger.info("Extracting TF-IDF features...")
            train_tfidf = multimodal_extractor.extract_tfidf_features(
                train_data['clean_comment'].tolist(), fit=True
            )
            test_tfidf = multimodal_extractor.extract_tfidf_features(
                test_data['clean_comment'].tolist(), fit=False
            )
            
            # Extract sentiment features
            logger.info("Extracting sentiment features...")
            train_sentiment_df = sentiment_extractor.extract_features_batch(
                train_data['clean_comment'].tolist()
            )
            test_sentiment_df = sentiment_extractor.extract_features_batch(
                test_data['clean_comment'].tolist()
            )
            
            # Log feature metrics
            mlflow.log_metric("tfidf_features", train_tfidf.shape[1])
            mlflow.log_metric("sentiment_features", train_sentiment_df.shape[1])
            mlflow.log_metric("total_features", train_tfidf.shape[1] + train_sentiment_df.shape[1])
            
            # Create output directory
            features_dir = Path('./data/features')
            features_dir.mkdir(exist_ok=True)
            
            # Save features
            feature_data = {
                'train_tfidf': train_tfidf,
                'test_tfidf': test_tfidf,
                'train_sentiment': train_sentiment_df.values,
                'test_sentiment': test_sentiment_df.values,
                'sentiment_feature_names': train_sentiment_df.columns.tolist(),
                'train_labels': train_data['category'].values,
                'test_labels': test_data['category'].values
            }
            
            with open(features_dir / 'extracted_features.pkl', 'wb') as f:
                pickle.dump(feature_data, f)
            
            # Log artifacts
            mlflow.log_artifact(str(features_dir / 'extracted_features.pkl'))
            
            logger.info("Feature extraction completed successfully")
            
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()