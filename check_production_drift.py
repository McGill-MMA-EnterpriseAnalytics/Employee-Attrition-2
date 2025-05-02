import pandas as pd
import numpy as np
import os
import glob
import mlflow
from datetime import datetime
from src.monitoring.drift_detection import detect_drift
from src.config.config import settings

def get_latest_reference_data():
    """Get the latest reference data file."""
    files = glob.glob(str(settings.DRIFT_REFERENCE_DIR / "reference_data_*.parquet"))
    if not files:
        raise FileNotFoundError("No reference data files found. Run save_reference_data.py first.")
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Using reference data: {latest_file}")
    return pd.read_parquet(latest_file)

def generate_production_data(num_samples=500, drift_level='none'):
    """
    Generate production data with various drift levels.
    
    Args:
        num_samples: Number of data points to generate
        drift_level: 'none', 'low', 'medium', or 'high'
    """
    np.random.seed(int(datetime.now().timestamp()))  # Random seed based on current time
    
    # Drift parameters based on level
    if drift_level == 'none':
        age_mean, age_std = 35, 5
        salary_mean, salary_std = 50000, 10000
        satisfaction_mean, satisfaction_std = 3.5, 0.8
        dept_probs = None  # Default distribution
        attrition_rate = 0.2
    elif drift_level == 'low':
        age_mean, age_std = 36, 5.5
        salary_mean, salary_std = 52000, 11000
        satisfaction_mean, satisfaction_std = 3.3, 0.9
        dept_probs = [0.15, 0.3, 0.25, 0.2, 0.1]  # Slight change in department distribution
        attrition_rate = 0.25
    elif drift_level == 'medium':
        age_mean, age_std = 38, 6
        salary_mean, salary_std = 55000, 12000
        satisfaction_mean, satisfaction_std = 3.0, 1.0
        dept_probs = [0.1, 0.35, 0.3, 0.15, 0.1]  # More change in department distribution
        attrition_rate = 0.3
    elif drift_level == 'high':
        age_mean, age_std = 40, 7
        salary_mean, salary_std = 60000, 15000
        satisfaction_mean, satisfaction_std = 2.5, 1.2
        dept_probs = [0.05, 0.4, 0.35, 0.1, 0.1]  # Significant change in department distribution
        attrition_rate = 0.4
    else:
        raise ValueError("drift_level must be 'none', 'low', 'medium', or 'high'")
    
    # Generate data
    production_data = pd.DataFrame({
        # Numerical features
        'age': np.random.normal(age_mean, age_std, num_samples),
        'salary': np.random.normal(salary_mean, salary_std, num_samples),
        'satisfaction_score': np.random.normal(satisfaction_mean, satisfaction_std, num_samples),
        'years_at_company': np.random.poisson(4, num_samples),
        'last_evaluation': np.random.normal(0.7, 0.15, num_samples),
        
        # Categorical features
        'department': np.random.choice(
            ['HR', 'Sales', 'Engineering', 'Marketing', 'IT'], 
            num_samples,
            p=dept_probs
        ) if dept_probs else np.random.choice(['HR', 'Sales', 'Engineering', 'Marketing', 'IT'], num_samples),
        
        'job_level': np.random.choice(['Entry', 'Mid', 'Senior', 'Executive'], num_samples, p=[0.3, 0.4, 0.2, 0.1]),
        
        # Target variable (0: stayed, 1: left)
        'attrition': np.random.binomial(1, attrition_rate, num_samples)
    })
    
    # Make some logical correlations
    # Lower satisfaction tends to lead to higher attrition
    for i in range(num_samples):
        if production_data.loc[i, 'satisfaction_score'] < 2.5:
            production_data.loc[i, 'attrition'] = np.random.choice([0, 1], p=[0.3, 0.7])
    
    return production_data

def check_drift_and_log(reference_data, production_data, drift_level):
    """Check for drift and log results to MLflow."""
    # Set up MLflow
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment('production_drift_monitoring')
    
    # Define features
    numerical_features = ['age', 'salary', 'satisfaction_score', 'years_at_company', 'last_evaluation']
    categorical_features = ['department', 'job_level']
    target_column = 'attrition'
    
    # Start MLflow run
    with mlflow.start_run(run_name=f'drift_check_{drift_level}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        # Log parameters
        mlflow.log_param('drift_level', drift_level)
        mlflow.log_param('samples_count', len(production_data))
        mlflow.log_param('check_timestamp', datetime.now().isoformat())
        
        # Check feature drift
        print(f"\nChecking feature drift (drift level: {drift_level})...")
        feature_drift_results = detect_drift(
            reference_data=reference_data,
            current_data=production_data,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            mlflow_tracking=True
        )
        
        # Check target drift
        print(f"\nChecking target drift (drift level: {drift_level})...")
        target_drift_results = detect_drift(
            reference_data=reference_data,
            current_data=production_data,
            numerical_features=[], 
            categorical_features=[],
            prediction_column=target_column,
            mlflow_tracking=True
        )
        
        # Combine and log overall results
        overall_drift_detected = feature_drift_results['drift_detected'] or target_drift_results['drift_detected']
        overall_drift_score = max(feature_drift_results['drift_score'], target_drift_results['drift_score'])
        
        mlflow.log_metric('overall_drift_detected', int(overall_drift_detected))
        mlflow.log_metric('overall_drift_score', overall_drift_score)
        
        # Print results
        print(f"\nOverall drift results (drift level: {drift_level}):")
        print(f"Drift detected: {overall_drift_detected}")
        print(f"Drift score: {overall_drift_score}")
        print(f"Feature drift detected: {feature_drift_results['drift_detected']}")
        print(f"Target drift detected: {target_drift_results['drift_detected']}")
        print(f"Drifted features: {feature_drift_results['drifted_features']}")
        
        return overall_drift_detected, overall_drift_score

def main():
    """Main function to check for drift in production data."""
    # Get reference data
    try:
        reference_data = get_latest_reference_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run save_reference_data.py first to create reference data.")
        return
    
    # Run drift checks with different drift levels
    print("\n=== SIMULATING PRODUCTION DRIFT MONITORING ===")
    
    # Test with no drift
    production_data_no_drift = generate_production_data(drift_level='none')
    check_drift_and_log(reference_data, production_data_no_drift, 'none')
    
    # Test with low drift
    production_data_low_drift = generate_production_data(drift_level='low')
    check_drift_and_log(reference_data, production_data_low_drift, 'low')
    
    # Test with medium drift
    production_data_medium_drift = generate_production_data(drift_level='medium')
    check_drift_and_log(reference_data, production_data_medium_drift, 'medium')
    
    # Test with high drift
    production_data_high_drift = generate_production_data(drift_level='high')
    check_drift_and_log(reference_data, production_data_high_drift, 'high')
    
    print("\n=== DRIFT MONITORING COMPLETE ===")
    print("View the results in MLflow UI (experiment: production_drift_monitoring)")
    print("MLflow URL: http://localhost:5000")

if __name__ == "__main__":
    main() 