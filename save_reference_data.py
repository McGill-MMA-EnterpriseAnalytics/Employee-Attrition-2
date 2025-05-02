import pandas as pd
import numpy as np
import os
import mlflow
from datetime import datetime
from src.config.config import settings

def main():
    """Generate and save reference data for future drift detection."""
    
    # Create reference data with appropriate schema
    np.random.seed(42)  # Fixed seed for reproducibility
    num_samples = 500
    
    reference_data = pd.DataFrame({
        # Numerical features
        'age': np.random.normal(35, 5, num_samples),
        'salary': np.random.normal(50000, 10000, num_samples),
        'satisfaction_score': np.random.normal(3.5, 0.8, num_samples),
        'years_at_company': np.random.poisson(4, num_samples),
        'last_evaluation': np.random.normal(0.7, 0.15, num_samples),
        
        # Categorical features
        'department': np.random.choice(['HR', 'Sales', 'Engineering', 'Marketing', 'IT'], num_samples),
        'job_level': np.random.choice(['Entry', 'Mid', 'Senior', 'Executive'], num_samples, p=[0.3, 0.4, 0.2, 0.1]),
        
        # Target variable (0: stayed, 1: left)
        'attrition': np.random.binomial(1, 0.2, num_samples)  # 20% attrition rate
    })
    
    # Make some logical correlations
    # Lower satisfaction tends to lead to higher attrition
    for i in range(num_samples):
        if reference_data.loc[i, 'satisfaction_score'] < 2.5:
            reference_data.loc[i, 'attrition'] = np.random.choice([0, 1], p=[0.3, 0.7])
    
    # Ensure directories exist
    drift_ref_dir = settings.DRIFT_REFERENCE_DIR
    os.makedirs(drift_ref_dir, exist_ok=True)
    
    # Save reference data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_file_path = drift_ref_dir / f"reference_data_{timestamp}.parquet"
    reference_data.to_parquet(ref_file_path)
    
    print(f"Reference data saved to: {ref_file_path}")
    
    # Log to MLflow
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment('reference_data')
    
    with mlflow.start_run(run_name='reference_data_creation'):
        # Log basic statistics
        mlflow.log_param('num_samples', num_samples)
        mlflow.log_param('num_features', len(reference_data.columns) - 1)  # Excluding target
        mlflow.log_param('timestamp', timestamp)
        
        # Log feature statistics
        for col in reference_data.columns:
            if reference_data[col].dtype in [np.float64, np.int64]:
                mlflow.log_metric(f'{col}_mean', reference_data[col].mean())
                mlflow.log_metric(f'{col}_std', reference_data[col].std())
        
        # Log reference data as an artifact
        mlflow.log_artifact(ref_file_path)
        
        print(f"Reference data statistics logged to MLflow")
        print(f"Reference data artifact saved in MLflow")
    
    print("\nYou can now use this reference data for drift detection with:")
    print(f"reference_data = pd.read_parquet('{ref_file_path}')")
    print("current_data = ... # Load your new data")
    print("detect_drift(reference_data, current_data, ...)")

if __name__ == "__main__":
    main() 