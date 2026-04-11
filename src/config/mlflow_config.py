"""
MLflow configuration for DagsHub integration
"""
import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_mlflow():
    """
    Configure MLflow to work with DagsHub or fallback to local
    """
    # Set MLflow tracking URI
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Set credentials if using PAT
        dagshub_pat = os.getenv('DAGSHUB_PAT')
        if dagshub_pat:
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_pat
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_pat
        
        # Alternative: use username/password
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        if username and password:
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    else:
        # Fallback to local MLflow tracking
        print("No DagsHub configuration found. Using local MLflow tracking.")
        print("To use DagsHub, create a .env file with your DagsHub credentials.")

def get_or_create_experiment(experiment_name: str):
    """
    Get existing experiment or create new one
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
        else:
            print(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
    except Exception as e:
        print(f"Error with experiment setup: {e}")
        return None