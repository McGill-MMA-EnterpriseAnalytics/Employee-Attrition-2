# Employee Attrition MLOps Project

A full-stack MLOps solution for employee attrition prediction with robust drift detection capabilities, featuring:
- Automated model training, retraining, and promotion
- Drift detection and monitoring
- API and Streamlit frontend
- MLflow tracking and artifact management
- CI/CD with GitHub Actions and Docker Compose

## Features

- **ML Model Training**: Automated training and validation of employee attrition prediction models
- **MLflow Integration**: Tracking experiments, model registration, and model versioning
- **Drift Detection System**: 
  - Feature drift monitoring with statistical tests
  - Prediction drift monitoring for model outputs
  - Automated alerts when drift is detected
  - Detailed HTML reports with feature-by-feature analysis
  - FastAPI endpoint for on-demand drift detection
  - MLflow integration for tracking drift metrics over time
  - Customizable drift thresholds for different sensitivity levels
- **GitHub Actions Workflows**:
  - Automated drift detection on schedule
  - Model promotion workflow
  - MLflow metadata maintenance
- **Visualization**: Comprehensive HTML reports for data and prediction drift
- **Frontend**: Streamlit app for live predictions and model info

## Architecture

- **API**: Serves predictions and model info (FastAPI)
- **Frontend**: Streamlit app for live predictions and model info
- **MLflow**: Model tracking and artifact storage
- **Automation**: All workflows managed by GitHub Actions

### Drift Detection Architecture

The drift detection system consists of:

1. **Reference Data Management**:
   - Saving baseline data for comparison
   - Storing feature distributions and statistics

2. **Drift Detection Pipeline**:
   - Feature drift detection using statistical tests
   - Prediction drift monitoring
   - HTML report generation

3. **Automation**:
   - GitHub Actions workflows for scheduled monitoring
   - Automatic issue creation for detected drift
   - Model retraining triggers

4. **API Layer**:
   - FastAPI endpoints for drift detection
   - Report generation and retrieval

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/BTCJULIAN/Employee-Attrition-2.git
cd Employee-Attrition-2

# Install dependencies with Poetry
poetry install
```

### Running Drift Detection

```bash
# Run drift detection with default settings
python check_production_drift.py

# Generate HTML report for current data
python scripts/generate_drift_report.py --current-data path/to/data.csv

# Save new reference data (baseline) for drift comparison
python save_reference_data.py
```

### Using the Drift Detection API

The project includes a FastAPI endpoint for drift detection:

```bash
# Start the API server
python drift_api.py

# Test the API with the test client
python test_drift_api_client.py

# Access the API documentation at http://localhost:8000/docs
```

Example API request:
```python
import requests
import pandas as pd

# Load data
df = pd.read_csv("path/to/data.csv")
data = df.to_dict(orient="records")

# Detect drift
response = requests.post(
    "http://localhost:8000/detect-drift",
    json={"data": data, "threshold": 0.05}
)

# Check results
result = response.json()
print(f"Drift detected: {result['drift_detected']}")
```

For comprehensive documentation on drift detection, see the [Drift Detection Guide](docs/drift_detection_guide.md).

### MLflow Maintenance

Repair and maintain MLflow metadata with:

```bash
python scripts/mlflow_maintenance.py --fix-run-metadata
```

## Quickstart with Docker

1. Clone the repo and set up `.env`:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Build and run all services:
   ```bash
   docker-compose up --build
   ```

3. Access:
   - API: http://localhost:8000
   - Frontend: http://localhost:8501
   - MLflow: http://localhost:5001

## Project Structure

/
├── .github/workflows/ # GitHub Actions workflows
├── scripts/ # Automation and utility scripts
├── src/ # Source code
│ ├── employee_attrition_mlops/ # Core ML logic
│ ├── monitoring/ # Drift detection
│ └── frontend/ # Streamlit app
├── tests/ # Test files
├── docs/ # Documentation
├── mlruns/ # MLflow tracking and artifacts
└── reports/ # Generated reports



## CI/CD & Automation

All automation is managed by GitHub Actions workflows:
- Testing and linting
- Drift detection
- Model retraining
- Batch prediction
- Model promotion
- API redeployment

## Environment Setup

### Required Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DATABASE_URL_PYMSSQL=mssql+pymssql://username:password@hostname:1433/database

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5001  # MLflow server
MLFLOW_MODEL_NAME=employee_attrition_model
MLFLOW_MODEL_STAGE=Production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## License

MIT