# Drift Detection Guide

This guide provides comprehensive instructions for using the Drift Detection system in the Employee Attrition MLOps project.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Configuration](#setup-and-configuration)
- [Running Drift Detection](#running-drift-detection)
- [Working with HTML Reports](#working-with-html-reports)
- [Using the API](#using-the-api)
- [GitHub Actions Integration](#github-actions-integration)
- [MLflow Maintenance](#mlflow-maintenance)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Introduction

The drift detection system monitors data and prediction drift over time, helping detect when your machine learning model might need retraining. The system uses:

- **Evidently**: For statistical tests and report generation
- **MLflow**: For experiment tracking and artifact storage
- **GitHub Actions**: For automated monitoring and notifications
- **FastAPI**: For API access to drift detection

### Key Concepts

- **Data Drift**: Changes in the distribution of input features over time
- **Prediction Drift**: Changes in the model's predictions over time
- **Reference Data**: Baseline data (usually training data) used for comparison
- **Drift Score**: Measure of how much drift has occurred (0-1)
- **Drift Threshold**: The value above which drift is considered significant (default: 0.05)

## Setup and Configuration

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- MLflow server running

### Setting Up Reference Data

Before you can detect drift, you need reference data:

```bash
# Save reference data from your training dataset
python save_reference_data.py

# Or use a specific input file
python save_reference_data.py --input-file path/to/training_data.csv
```

The reference data will be saved in the `drift_reference/` directory and also logged to MLflow for tracking. The script generates synthetic employee data by default if no input file is provided.

## Running Drift Detection

### Basic Drift Detection

```bash
# Run drift detection with default settings
python check_production_drift.py

# Run drift detection on a specific file
python check_production_drift.py --input-file path/to/production_data.csv

# Run drift detection with a specific model
python check_production_drift.py --run-id <mlflow_run_id>

# Run drift detection with a custom threshold
python check_production_drift.py --threshold 0.01
```

### Interpreting Results

Drift detection outputs:
- Drift detected (yes/no)
- Drift score (0-1)
- List of drifted features
- Number of drifted features
- Drift share (percentage of features that drifted)

A higher drift score indicates more significant drift in your data. The default threshold is 0.05, meaning any score above this will trigger a drift detection alert.

## Working with HTML Reports

### Generating Reports

Drift detection can generate comprehensive HTML reports with feature-by-feature analysis:

```bash
# Generate a comprehensive HTML report
python scripts/generate_drift_report.py --current-data path/to/data.csv

# Generate a report with a custom name
python scripts/generate_drift_report.py --current-data path/to/data.csv --report-name monthly_drift_may

# Include target/prediction drift
python scripts/generate_drift_report.py --current-data path/to/data.csv --target target_column
```

### Viewing Reports

HTML reports are saved in the `reports/` directory and can be opened in any web browser. They include:

- Feature distribution comparisons with overlay charts
- Statistical test results for each feature
- Data quality metrics and changes
- Drift metrics and visualizations
- Recommendations for addressing drift issues

## Using the API

The project includes a FastAPI endpoint for on-demand drift detection.

### Starting the API Server

```bash
# Start the API server
python drift_api.py

# With a custom port
PORT=8080 python drift_api.py
```

### API Endpoints

The API is documented at http://localhost:8000/docs and includes:

- `GET /health`: Health check to verify the API is running
- `POST /detect-drift`: Detect drift from JSON data
- `GET /`: Root endpoint with API information

### Example API Usage

To test the API, you can use the included test client:

```bash
# Run the test client
python test_drift_api_client.py
```

For programmatic usage:

```python
import requests
import pandas as pd

# Load data
df = pd.read_csv("path/to/data.csv")

# Convert to format for API
data = df.to_dict(orient="records")

# Detect drift
response = requests.post(
    "http://localhost:8000/detect-drift",
    json={
        "data": data,
        "threshold": 0.05
    }
)

# Check results
result = response.json()
print(f"Drift detected: {result['drift_detected']}")
print(f"Drift score: {result['drift_score']}")
print(f"Drifted features: {result['drifted_features']}")
```

## GitHub Actions Integration

The project includes GitHub Actions workflows for automated drift detection:

### Scheduled Monitoring

The drift detection workflow runs automatically every Monday and:
1. Performs drift detection on the latest data
2. Creates GitHub issues if drift is detected
3. Generates HTML reports and attaches them as artifacts
4. Fixes MLflow metadata issues automatically

### Manual Triggering

You can manually trigger drift detection:
1. Go to the "Actions" tab in your GitHub repository
2. Select the "Drift Detection" workflow
3. Click "Run workflow"

## MLflow Maintenance

The drift detection system integrates with MLflow to track drift metrics over time.

### Viewing Drift Metrics in MLflow

1. Start the MLflow UI: `mlflow ui`
2. Navigate to the experiment containing your drift detection runs
3. View drift metrics including:
   - Drift score
   - Number of drifted features
   - Drift detected (0/1)

### Fixing MLflow Metadata

```bash
# Fix missing meta.yaml files
python scripts/mlflow_maintenance.py

# Fix run metadata issues
python scripts/mlflow_maintenance.py --fix-run-metadata

# Dry run (show what would be fixed without making changes)
python scripts/mlflow_maintenance.py --fix-run-metadata --dry-run
```

## Advanced Usage

### Custom Drift Detection

You can use the `DriftDetector` class directly in your Python code:

```python
from src.monitoring.drift_detection import DriftDetector
import pandas as pd

# Load reference and current data
reference_data = pd.read_parquet("drift_reference/reference_train_data.parquet")
current_data = pd.read_csv("path/to/current_data.csv")

# Create detector with custom threshold
detector = DriftDetector(drift_threshold=0.01, mlflow_tracking=True)

# Detect feature drift
drift_detected, drift_score, drifted_features = detector.detect_drift(
    reference_data=reference_data,
    current_data=current_data,
    features=["age", "salary", "department"]  # Optional: specify features to check
)

# Detect prediction drift
pred_drift, pred_score = detector.detect_prediction_drift(
    reference_data=reference_data,
    current_data=current_data,
    prediction_column="predicted_attrition"
)

print(f"Feature drift detected: {drift_detected}, score: {drift_score}")
print(f"Prediction drift detected: {pred_drift}, score: {pred_score}")
```

### Integrating with Model Monitoring

You can integrate drift detection into your model monitoring pipeline:

1. Set up scheduled drift detection using GitHub Actions
2. Configure alerting through GitHub Issues when drift is detected
3. Implement automatic model retraining when drift exceeds critical thresholds

## Troubleshooting

### Common Issues

- **Missing reference data**: Ensure you've run `save_reference_data.py` to create reference data
- **MLflow errors**: Run `mlflow_maintenance.py` to fix metadata issues
- **API connection errors**: Verify the MLflow server is running and accessible
- **No drift detected when expected**: Check if the threshold is too high (try lowering it)
- **All features showing drift**: Check for data type mismatches or preprocessing issues

### Logs

- Check MLflow logs for drift detection results
- API logs are available in the terminal when running the API server
- GitHub Actions logs can be found in the "Actions" tab of your repository 

### Getting Help

If you encounter issues not covered in this guide:
1. Check the issue tracker for similar problems
2. Review the implementation in `src/monitoring/drift_detection.py`
3. Run tests to verify your drift detection setup works correctly 