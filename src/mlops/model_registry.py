# register model

import json
import mlflow
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.mlflow_config import setup_mlflow, get_or_create_experiment


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # For local MLflow, we'll just log the model info instead of registering
        # Model registry requires MLflow server with backend store
        if 'MLFLOW_TRACKING_URI' in os.environ and 'dagshub.com' in os.environ.get('MLFLOW_TRACKING_URI', ''):
            # DagsHub MLflow - try to register
            model_uri = f"runs:/{model_info['run_id']}/model"
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        else:
            # Local MLflow - just log the information
            logger.info(f'Local MLflow detected. Model info logged: {model_info}')
            logger.info(f'Model would be registered as: {model_name}')
            logger.info('To enable model registry, configure DagsHub MLflow tracking.')
            
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        # Setup MLflow configuration
        setup_mlflow()
        
        # Get structured experiment name
        import yaml
        try:
            with open('params.yaml', 'r') as f:
                params = yaml.safe_load(f)
            experiment_name = params.get('mlflow', {}).get('experiments', {}).get('model_registry', '05_Model_Registry')
            get_or_create_experiment(experiment_name)
        except:
            pass
        
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()