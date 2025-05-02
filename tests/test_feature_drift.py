import pytest
import pandas as pd
import numpy as np
import mlflow
from src.monitoring.drift_detection import detect_drift

def test_feature_drift():
    # Generate reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Generate current data with drift
    current_data = pd.DataFrame({
        'feature1': np.random.normal(1, 1, 1000),  # Shifted mean
        'feature2': np.random.normal(0, 2, 1000),  # Increased variance
        'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.4, 0.4, 0.2])  # Changed distribution
    })
    
    # Run drift detection
    results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_features=['feature1', 'feature2'],
        categorical_features=['feature3']
    )
    
    # Check results
    assert results['drift_detected'] is True
    assert results['drift_score'] > 0
    assert len(results['drifted_features']) > 0
    assert len(results['test_results']) > 0

def test_feature_drift_with_mlflow():
    # Set up MLflow
    mlflow.set_experiment("test_drift_detection")
    
    # Generate reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Generate current data with drift
    current_data = pd.DataFrame({
        'feature1': np.random.normal(1, 1, 1000),  # Shifted mean
        'feature2': np.random.normal(0, 2, 1000),  # Increased variance
        'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.4, 0.4, 0.2])  # Changed distribution
    })
    
    # Run drift detection with MLflow tracking
    with mlflow.start_run():
        results = detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_features=['feature1', 'feature2'],
            categorical_features=['feature3'],
            mlflow_tracking=True
        )
        
        # Check results
        assert results['drift_detected'] is True
        assert results['drift_score'] > 0
        assert len(results['drifted_features']) > 0
        assert len(results['test_results']) > 0
        
        # Verify MLflow metrics
        run = mlflow.active_run()
        assert run is not None
        metrics = mlflow.get_run(run.info.run_id).data.metrics
        assert 'drift_detected' in metrics
        assert 'drift_score' in metrics 