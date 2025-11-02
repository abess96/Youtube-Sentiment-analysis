"""
Feature Selection Pipeline with MLflow Integration
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import mlflow
from pathlib import Path
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engine import FeatureSelector
from utils.mlflow_config import setup_mlflow, get_or_create_experiment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params.get('feature_selection', {})

def main():
    """Main feature selection pipeline"""
    try:
        # Setup MLflow
        setup_mlflow()
        experiment_id = get_or_create_experiment("feature-engineering-pipeline")
        
        # Load parameters
        params = load_params()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="feature_selection"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tag("pipeline_stage", "feature_selection")
            
            # Load extracted features
            with open('./data/features/extracted_features.pkl', 'rb') as f:
                feature_data = pickle.load(f)
            
            # Combine features
            train_features = np.hstack([
                feature_data['train_tfidf'],
                feature_data['train_sentiment']
            ])
            test_features = np.hstack([
                feature_data['test_tfidf'],
                feature_data['test_sentiment']
            ])
            
            logger.info(f"Combined features shape: {train_features.shape}")
            mlflow.log_metric("combined_features", train_features.shape[1])
            
            # Initialize feature selector
            selector = FeatureSelector(
                n_features_to_select=params.get('n_features_to_select', 1000),
                selection_method=params.get('selection_method', 'mutual_info')
            )
            
            # Select features
            logger.info("Performing feature selection...")
            selected_train = selector.fit_transform(
                train_features, 
                feature_data['train_labels'],
                method=params.get('selection_method', 'mutual_info')
            )
            selected_test = selector.transform(test_features)
            
            # Log selection metrics
            mlflow.log_metric("selected_features", selected_train.shape[1])
            mlflow.log_metric("feature_reduction_ratio", selected_train.shape[1] / train_features.shape[1])
            
            # Create feature names
            tfidf_names = [f"tfidf_{i}" for i in range(feature_data['train_tfidf'].shape[1])]
            all_feature_names = tfidf_names + feature_data['sentiment_feature_names']
            
            # Rank features
            ranking_df = selector.rank_features(
                train_features,
                feature_data['train_labels'],
                method=params.get('selection_method', 'mutual_info'),
                feature_names=all_feature_names
            )
            
            # Log top feature importances
            top_features = ranking_df.head(10)
            for idx, row in top_features.iterrows():
                mlflow.log_metric(f"feature_importance_{row['feature_name']}", row['importance_score'])
            
            # Save selected features
            selected_data = {
                'train_features': selected_train,
                'test_features': selected_test,
                'train_labels': feature_data['train_labels'],
                'test_labels': feature_data['test_labels'],
                'selected_feature_indices': selector.selected_features_
            }
            
            features_dir = Path('./data/features')
            with open(features_dir / 'selected_features.pkl', 'wb') as f:
                pickle.dump(selected_data, f)
            
            # Save feature ranking
            ranking_df.to_csv(features_dir / 'feature_ranking.csv', index=False)
            
            # Log artifacts
            mlflow.log_artifact(str(features_dir / 'selected_features.pkl'))
            mlflow.log_artifact(str(features_dir / 'feature_ranking.csv'))
            
            logger.info(f"Feature selection completed. Selected {selected_train.shape[1]} features")
            
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        raise

if __name__ == "__main__":
    main()