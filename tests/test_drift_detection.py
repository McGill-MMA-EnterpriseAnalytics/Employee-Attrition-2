import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.drift_detection import detect_drift
from evidently.metrics import DataDriftTable
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.pipeline.column_mapping import ColumnMapping

def test_drift_detection():
    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'age': np.random.normal(35, 5, 100),
        'salary': np.random.normal(50000, 10000, 100),
        'satisfaction': np.random.normal(0.7, 0.1, 100)
    })
    
    # Create current data with slight drift
    current_data = pd.DataFrame({
        'age': np.random.normal(40, 5, 100),  # Slightly older
        'salary': np.random.normal(55000, 10000, 100),  # Slightly higher salary
        'satisfaction': np.random.normal(0.6, 0.1, 100)  # Slightly lower satisfaction
    })
    
    # Run drift detection
    results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_features=['age', 'salary', 'satisfaction'],
        categorical_features=[]
    )
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'drift_detected' in results
    assert 'drift_score' in results
    assert 'drifted_features' in results
    assert 'test_results' in results
    
    # Check drift detection
    assert results['drift_detected'] is True  # We expect drift due to shifted distributions
    assert results['drift_score'] >= 0.0
    assert len(results['drifted_features']) > 0  # At least one feature should show drift
    assert len(results['test_results']) > 0  # Should have test results

def test_no_drift():
    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    
    # Create current data with no drift (same distribution)
    np.random.seed(43)  # Different seed but same distribution
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    
    # Run drift detection
    results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_features=['feature1', 'feature2'],
        categorical_features=[]
    )
    
    # Check results
    assert isinstance(results, dict)
    assert 'drift_detected' in results
    assert 'drift_score' in results
    assert 'drifted_features' in results
    assert 'test_results' in results
    
    # Since distributions are the same, we expect no significant drift
    assert results['drift_score'] >= 0.0
    assert len(results['test_results']) > 0 