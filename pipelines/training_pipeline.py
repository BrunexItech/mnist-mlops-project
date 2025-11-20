import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
# ... rest of imports

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_builder import ModelBuilder
from src.training.trainer import ModelTrainer
from src.mlflow_pipeline.tracking import MLflowTracker
import tensorflow as tf

def main():
    """Main training pipeline"""
    print("🚀 Starting MNIST Training Pipeline...")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_builder = ModelBuilder()
    trainer = ModelTrainer()
    mlflow_tracker = MLflowTracker()
    
    # Step 1: Load data
    print("\n Step 1: Loading data...")
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    
    # Step 2: Preprocess data
    print("\n Step 2: Preprocessing data...")
    x_train_clean, y_train_clean, x_test_clean, y_test_clean = preprocessor.clean_data(
        x_train, y_train, x_test, y_test
    )
    
    x_train_split, y_train_split, x_val, y_val = preprocessor.split_data(
        x_train_clean, y_train_clean
    )
    
    x_train_norm, x_val_norm, x_test_norm = preprocessor.normalize_data(
        x_train_split, x_val, x_test_clean
    )
    
    x_train_final, x_val_final, x_test_final = preprocessor.reshape_for_cnn(
        x_train_norm, x_val_norm, x_test_norm
    )
    
    # Step 3: Create datasets
    print("\n Step 3: Creating datasets...")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_final, y_train_split))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_final, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_final, y_test_clean))
    
    # Apply batching and prefetching
    BATCH_SIZE = 32
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Step 4: Build and compile model
    print("\n Step 4: Building model...")
    model = model_builder.create_cnn_model()
    model_builder.compile_model()
    model_builder.get_model_summary()
    
    # Step 5: Setup training
    print("\n⚡ Step 5: Setting up training...")
    trainer.setup_callbacks()
    
    # Step 6: Train model with MLflow tracking
    print("\n Step 6: Training model...")
    history = trainer.train_model(model, train_dataset, val_dataset, epochs=10)
    
    # Step 7: Evaluate model
    print("\n Step 7: Evaluating model...")
    test_loss, test_accuracy = trainer.evaluate_model(model, test_dataset)
    
    print(f"\nTraining pipeline completed!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    return model, history, test_accuracy

if __name__ == "__main__":
    model, history, test_accuracy = main()