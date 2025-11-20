import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from ..utils.config import config

class DataLoader:
    """Data loader class for MNIST dataset"""
    
    def __init__(self):
        self.raw_data_path = config.data_paths['raw'] / 'mnist_dataset.npz'
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load_data(self) -> tuple:
        """Load MNIST dataset from local raw data only"""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"MNIST data not found at {self.raw_data_path}. "
                "Please run the notebook first to download and save the data."
            )
        
        self._load_from_local()
        self._print_data_shapes()
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def _load_from_local(self):
        """Load dataset from local raw data"""
        with np.load(self.raw_data_path) as data:
            self.x_train = data['x_train']
            self.y_train = data['y_train']
            self.x_test = data['x_test']
            self.y_test = data['y_test']
        print("✓ Loaded MNIST from local raw data")
    
    def _print_data_shapes(self):
        """Print data shapes for understanding"""
        print(f'Training data shape: {self.x_train.shape}')
        print(f'Training label shape: {self.y_train.shape}')
        print(f'Testing data shape: {self.x_test.shape}')
        print(f'Testing label shape: {self.y_test.shape}')
    
    def get_data_summary(self) -> dict:
        """Get comprehensive data summary"""
        return {
            'train_samples': len(self.x_train),
            'test_samples': len(self.x_test),
            'image_shape': self.x_train.shape[1:],
            'num_classes': len(np.unique(self.y_train))
        }