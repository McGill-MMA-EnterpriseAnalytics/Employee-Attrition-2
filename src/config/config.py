"""
Configuration settings for the project.
"""
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    """Project settings."""
    
    # Base directories
    PROJECT_ROOT = Path.cwd()
    DRIFT_REFERENCE_DIR = PROJECT_ROOT / "drift_reference"
    DRIFT_ARTIFACTS_DIR = PROJECT_ROOT / "drift_artifacts"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = "employee_attrition"
    
    # Drift detection settings
    DEFAULT_DRIFT_THRESHOLD = 0.05


# Export the settings object
settings = Settings() 