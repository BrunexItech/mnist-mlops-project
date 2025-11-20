import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetSummaryMetric, DataQualityMetrics

class DataValidator:
    """Data validation using Evidently AI"""
    
    def validate_mnist_data(self, x_data, y_data):
        """Validate MNIST dataset quality"""
        # Convert to DataFrame
        x_flat = x_data.reshape(x_data.shape[0], -1)
        feature_columns = [f'pixel_{i}' for i in range(x_flat.shape[1])]
        
        df = pd.DataFrame(x_flat, columns=feature_columns)
        df['target'] = y_data
        
        # Create validation report
        report = Report(metrics=[DataQualityMetrics()])
        report.run(current_data=df, reference_data=None)
        
        # Basic quality checks
        checks = {
            "shape_correct": x_data.shape[1:] == (28, 28),
            "pixel_range": (x_data.min() >= 0) and (x_data.max() <= 255),
            "no_nulls": not np.isnan(x_data).any(),
            "labels_range": (y_data.min() >= 0) and (y_data.max() <= 9),
            "num_classes": len(np.unique(y_data)) == 10
        }
        
        print("Data Quality Checks:")
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check}: {result}")
        
        return all(checks.values()), report