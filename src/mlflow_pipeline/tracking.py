import mlflow
import mlflow.keras
from pathlib import Path
import tensorflow as tf
from ..utils.config import config

class MLflowTracker:
    """MLflow tracking class for experiment management"""
    
    def __init__(self):
        self.experiment_name = config.base['mlflow']['experiment_name']
        self.tracking_uri = config.base['mlflow']['tracking_uri']
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"✓ MLflow tracking setup: {self.tracking_uri}")
        print(f"✓ Experiment: {self.experiment_name}")
    
    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: dict):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"✓ Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: dict, step=None):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        print(f"✓ Logged {len(metrics)} metrics")
    
    def log_model(self, model, model_name="mnist_cnn"):
        """Log model to MLflow"""
        mlflow.keras.log_model(model, model_name)
        print(f"✓ Model logged: {model_name}")
    
    def log_artifact(self, artifact_path):
        """Log artifact to MLflow"""
        mlflow.log_artifact(artifact_path)
        print(f"✓ Artifact logged: {artifact_path}")
    
    def register_model(self, model_uri, model_name):
        """Register model in MLflow model registry"""
        mlflow.register_model(model_uri, model_name)
        print(f"✓ Model registered: {model_name}")
    
    def get_experiment_runs(self):
        """Get all runs for current experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return None
    