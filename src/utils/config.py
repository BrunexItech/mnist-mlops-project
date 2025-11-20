import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for managing project settings."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    def __init__(self):
        self.load_configs()
    
    def load_configs(self):
        """Load configuration from YAML files."""
        config_dir = self.BASE_DIR / "config"
        
        # Load base config
        base_config_path = config_dir / "base.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self.base = yaml.safe_load(f)
        else:
            self.base = {}
        
        # Set default values
        self.data_paths = {
            'raw': self.DATA_DIR / "raw",
            'processed': self.DATA_DIR / "processed",
            'features': self.DATA_DIR / "features"
        }
        
        self.model_paths = {
            'trained': self.MODELS_DIR / "trained",
            'deployed': self.MODELS_DIR / "deployed"
        }

# Global config instance
config = Config()