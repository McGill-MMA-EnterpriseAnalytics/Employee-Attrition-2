import pandas as pd
import numpy as np
import mlflow
import os
from src.monitoring.drift_detection import detect_drift

def main():
    # Create reference data with predictions
    np.random.seed(42)
    features = {
        'feature1': np.random.normal(0, 1, 300),
        'feature2': np.random.normal(5, 2, 300),
        'feature3': np.random.choice(['A', 'B', 'C'], 300, p=[0.6, 0.3, 0.1])
    }
    
    # Generate predictions (30% class 1 rate)
    predictions = np.random.binomial(1, 0.3, 300)
    
    reference_data = pd.DataFrame({
        **features,
        'prediction': predictions
    })
    
    # Create current data with prediction drift
    np.random.seed(100)
    current_features = {
        'feature1': np.random.normal(0.1, 1.1, 300),  # Slight feature drift
        'feature2': np.random.normal(5, 2, 300),      # No drift 
        'feature3': np.random.choice(['A', 'B', 'C'], 300, p=[0.55, 0.35, 0.1])  # Slight drift
    }
    
    # Generate predictions with drift (70% class 1 rate - significant prediction drift)
    current_predictions = np.random.binomial(1, 0.7, 300)
    
    current_data = pd.DataFrame({
        **current_features,
        'prediction': current_predictions
    })

    # Set up MLflow tracking
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment('prediction_drift_test')

    # Run drift detection with MLflow tracking
    with mlflow.start_run(run_name='prediction_drift_demo'):
        results = detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_features=['feature1', 'feature2'],
            categorical_features=['feature3'],
            prediction_column='prediction',
            mlflow_tracking=True
        )
        
        # Print the results
        print('\nPrediction Drift Detection Results:')
        print(f'Drift detected: {results["drift_detected"]}')
        print(f'Drift score: {results["drift_score"]}')
        print(f'Drifted features: {results["drifted_features"]}')
        
        # MLflow will have logged:
        # - drift_detected: 1 (True)
        # - drift_score: value between 0 and 1
        # - drifted_features_count: number of drifted features
        # - Individual test results

    # Print MLflow tracking URL
    print('\nOpen MLflow UI (which should be running) at: http://localhost:5000')
    print('Look for the prediction_drift_test experiment')

if __name__ == "__main__":
    main() 