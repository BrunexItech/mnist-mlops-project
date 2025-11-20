import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from ..utils.config import config

class DataDriftMonitor:
    """Data drift monitoring class using Evidently AI"""
    
    def __init__(self):
        self.reference_data = None
        self.current_data = None
    
    def set_reference_data(self, x_data, y_data):
        """Set reference data for drift comparison"""
        # Convert to DataFrame for Evidently
        x_flat = x_data.reshape(x_data.shape[0], -1)
        feature_columns = [f'pixel_{i}' for i in range(x_flat.shape[1])]
        
        self.reference_data = pd.DataFrame(x_flat, columns=feature_columns)
        self.reference_data['target'] = y_data
        print(f"✓ Reference data set: {self.reference_data.shape}")
    
    def set_current_data(self, x_data, y_data):
        """Set current data for drift detection"""
        x_flat = x_data.reshape(x_data.shape[0], -1)
        feature_columns = [f'pixel_{i}' for i in range(x_flat.shape[1])]
        
        self.current_data = pd.DataFrame(x_flat, columns=feature_columns)
        self.current_data['target'] = y_data
        print(f"✓ Current data set: {self.current_data.shape}")
    
    def check_data_drift(self):
        """Check for data drift between reference and current data"""
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Reference and current data must be set first")
        
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        
        return data_drift_report
    
    def generate_drift_report(self, report, save_path=None):
        """Generate and save drift report"""
        drift_result = report.as_dict()
        drift_detected = drift_result['metrics'][0]['result']['dataset_drift']
        
        print(f"Data Drift Detected: {drift_detected}")
        
        if save_path:
            report.save_html(save_path)
            print(f"✓ Drift report saved to: {save_path}")
        
        return drift_detected