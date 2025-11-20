import numpy as np
from sklearn.model_selection import train_test_split
from ..utils.config import config

class DataPreprocessor:
    """Data preprocessing class for MNIST dataset"""
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        
    def clean_data(self, x_train, y_train, x_test, y_test):
        """Clean data by removing corrupted images"""
        # Remove completely black images
        zero_images_train = np.all(x_train == 0, axis=(1, 2))
        zero_images_test = np.all(x_test == 0, axis=(1, 2))
        
        valid_indices_train = ~zero_images_train
        valid_indices_test = ~zero_images_test
        
        x_train_clean = x_train[valid_indices_train]
        y_train_clean = y_train[valid_indices_train]
        x_test_clean = x_test[valid_indices_test]
        y_test_clean = y_test[valid_indices_test]
        
        print(f"After cleaning - Training: {x_train_clean.shape}, Test: {x_test_clean.shape}")
        return x_train_clean, y_train_clean, x_test_clean, y_test_clean
    
    def split_data(self, x_train, y_train, test_size=0.2, random_state=42):
        """Split data into training and validation sets"""
        x_temp, x_val, y_temp, y_val = train_test_split(
            x_train, y_train, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_train
        )
        
        self.x_train = x_temp
        self.y_train = y_temp
        self.x_val = x_val
        self.y_val = y_val
        
        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        return self.x_train, self.y_train, self.x_val, self.y_val
    
    def normalize_data(self, x_train, x_val, x_test):
        """Normalize pixel values to [0, 1] range"""
        x_train_normalized = x_train.astype('float32') / 255.0
        x_val_normalized = x_val.astype('float32') / 255.0
        x_test_normalized = x_test.astype('float32') / 255.0
        
        print(f"Normalized pixel range: [{x_train_normalized.min():.3f}, {x_train_normalized.max():.3f}]")
        return x_train_normalized, x_val_normalized, x_test_normalized
    
    def reshape_for_cnn(self, x_train, x_val, x_test):
        """Reshape data for CNN input"""
        x_train_reshaped = x_train.reshape(-1, 28, 28, 1)
        x_val_reshaped = x_val.reshape(-1, 28, 28, 1)
        x_test_reshaped = x_test.reshape(-1, 28, 28, 1)
        
        print(f"Training shape after reshaping: {x_train_reshaped.shape}")
        print(f"Validation shape after reshaping: {x_val_reshaped.shape}")
        print(f"Test shape after reshaping: {x_test_reshaped.shape}")
        return x_train_reshaped, x_val_reshaped, x_test_reshaped