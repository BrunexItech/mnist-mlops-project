import mlflow
from mlflow.tracking import MlflowClient
from ..utils.config import config

class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()
        self.experiment_name = config.base['mlflow']['experiment_name']
    
    def register_model(self, run_id, model_name="mnist-cnn"):
        """Register a model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        print(f"Model registered: {result.name} version {result.version}")
        return result
    
    def transition_stage(self, model_name, version, stage="Staging"):
        """Transition model to different stage (Staging/Production/Archived)"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"✅ Model {model_name} v{version} moved to {stage}")