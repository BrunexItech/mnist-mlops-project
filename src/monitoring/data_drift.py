import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

class DataDriftMonitor:
    def __init__(self):
        self.reference_data = None
    
    def set_reference_data(self, x_data, y_data):
        """Set reference dataset for drift comparison"""
        x_flat = x_data.reshape(x_data.shape[0], -1)
        feature_cols = [f'pixel_{i}' for i in range(x_flat.shape[1])]
        
        self.reference_data = pd.DataFrame(x_flat, columns=feature_cols)
        self.reference_data['target'] = y_data
        print(f"✓ Reference data set: {self.reference_data.shape}")
    
    def check_drift(self, current_x, current_y):
        """Check data drift against reference data"""
        current_flat = current_x.reshape(current_x.shape[0], -1)
        feature_cols = [f'pixel_{i}' for i in range(current_flat.shape[1])]
        
        current_data = pd.DataFrame(current_flat, columns=feature_cols)
        current_data['target'] = current_y
        
        # Generate drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        return drift_report