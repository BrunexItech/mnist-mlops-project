import tensorflow as tf
import mlflow
import mlflow.keras
from pathlib import Path
from ..utils.config import config

class ModelTrainer:
    """Model training class with MLflow tracking"""
    
    def __init__(self):
        self.history = None
        self.callbacks = []
    
    def setup_callbacks(self, checkpoint_path='best_model.h5', patience=10):
        """Setup training callbacks"""
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        return self.callbacks
    
    def train_model(self, model, train_dataset, val_dataset, epochs=10):
        """Train the model with MLflow tracking"""
        # Start MLflow run
        mlflow.set_tracking_uri(config.base['mlflow']['tracking_uri'])
        mlflow.set_experiment(config.base['mlflow']['experiment_name'])
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", config.base['model']['batch_size'])
            mlflow.log_param("learning_rate", config.base['model']['learning_rate'])
            
            # Train model
            self.history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=self.callbacks,
                verbose=1
            )
            
            # Log metrics
            for epoch in range(len(self.history.history['accuracy'])):
                mlflow.log_metric("train_accuracy", self.history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", self.history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric("train_loss", self.history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_loss", self.history.history['val_loss'][epoch], step=epoch)
            
            # Log model
            mlflow.keras.log_model(model, "model")
            
        return self.history
    
    def evaluate_model(self, model, test_dataset):
        """Evaluate model on test dataset"""
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss, test_accuracy