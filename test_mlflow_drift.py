import pandas as pd
import numpy as np
import mlflow
import os
from src.monitoring.drift_detection import detect_drift

def main():
    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'age': np.random.normal(35, 5, 200),
        'salary': np.random.normal(50000, 10000, 200),
        'satisfaction': np.random.normal(0.7, 0.1, 200),
        'department': np.random.choice(['HR', 'Sales', 'IT'], 200),
        'years_service': np.random.poisson(5, 200)
    })

    # Create current data with drift
    np.random.seed(100)
    current_data = pd.DataFrame({
        'age': np.random.normal(38, 6, 200),  # Shifted distribution
        'salary': np.random.normal(55000, 12000, 200),  # Shifted with wider variance
        'satisfaction': np.random.normal(0.6, 0.15, 200),  # Lower satisfaction
        'department': np.random.choice(['HR', 'Sales', 'IT'], 200, p=[0.4, 0.4, 0.2]),  # Changed distribution
        'years_service': np.random.poisson(4, 200)  # Slightly lower
    })

    # Set up MLflow tracking
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment('drift_detection_test')

    # Run drift detection with MLflow tracking
    with mlflow.start_run(run_name='drift_detection_demo'):
        results = detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_features=['age', 'salary', 'satisfaction', 'years_service'],
            categorical_features=['department'],
            mlflow_tracking=True
        )
        
        # Print the results
        print('\nDrift Detection Results:')
        print(f'Drift detected: {results["drift_detected"]}')
        print(f'Drift score: {results["drift_score"]}')
        print(f'Drifted features: {results["drifted_features"]}')

    # Print MLflow tracking URL
    print('\nMLflow UI can be started with: mlflow ui')
    print('Then view the results at: http://localhost:5000')

if __name__ == "__main__":
    main() 