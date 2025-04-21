# scripts/monitor_drift.py
import os
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Set MLflow tracking URI (fallback to local mlruns folder)
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", f"file://{os.getcwd()}/mlruns")
)

# Configuration values
MODEL_NAME = os.getenv("PRODUCTION_MODEL_NAME", "AttritionProductionModel")
REPORTS_PATH = os.getenv("REPORTS_PATH", "reports")
REFERENCE_DATA_PATH = os.path.join(REPORTS_PATH, "reference_train_data.parquet")
REFERENCE_FEATURES_PATH = os.path.join(REPORTS_PATH, "reference_feature_names.json")
RECENT_DATA_PATH = "data/recent/recent_data.parquet"

# Ensure reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)


def load_reference_data():
    df_ref = pd.read_parquet(REFERENCE_DATA_PATH)
    with open(REFERENCE_FEATURES_PATH, "r") as f:
        features = json.load(f)
    return df_ref, features


def load_recent_data(features):
    if not os.path.exists(RECENT_DATA_PATH):
        raise FileNotFoundError(f"Recent data file not found: {RECENT_DATA_PATH}")
    df = pd.read_parquet(RECENT_DATA_PATH)
    return df[features]


def get_production_run_id(model_name: str):
    client = MlflowClient()
    # Try registry stages
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].run_id
    except Exception:
        pass
    # Fallback to latest training run
    exp = client.get_experiment_by_name("Attrition Final Model - logistic_regression")
    if exp:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if runs:
            return runs[0].info.run_id
    raise ValueError(f"No production or training run found for {model_name}")


def run_drift_detection():
    df_ref, features = load_reference_data()
    df_recent = load_recent_data(features)

    # Build Evidently report for data drift
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref, current_data=df_recent,
               column_mapping=ColumnMapping())

    # Save JSON report
    drift_report_path = os.path.join(REPORTS_PATH, "drift_report.json")
    report_dict = report.as_dict()
    with open(drift_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Drift report saved to {drift_report_path}")

    # Log to MLflow (artifact only)
    run_id = get_production_run_id(MODEL_NAME)
    with mlflow.start_run(run_name="drift_monitor", nested=False):
        try:
            mlflow.log_artifact(drift_report_path, artifact_path="drift_reports")
        except Exception as e:
            print(f"Warning: could not log artifact to MLflow: {e}")

    print("Drift detection complete and report logged to MLflow.")
