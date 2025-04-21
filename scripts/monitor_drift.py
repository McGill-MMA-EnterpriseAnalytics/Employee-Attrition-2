#!/usr/bin/env python
# scripts/monitor_drift.py

import os
import sys
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Ensure Python can import your package from <repo-root>/src
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from employee_attrition_mlops.config import (
    REPORTS_PATH,
    PRODUCTION_MODEL_NAME as MODEL_NAME,
    RETRAIN_TRIGGER_FEATURE_COUNT,
    RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE,
)

# Configure MLflow tracking URI (fallback to local mlruns)
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", f"file://{BASE_DIR}/mlruns")
)

# Ensure the reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

# File paths
REFERENCE_DATA_PATH     = os.path.join(REPORTS_PATH, "reference_train_data.parquet")
REFERENCE_FEATURES_PATH = os.path.join(REPORTS_PATH, "reference_feature_names.json")
RECENT_DATA_PATH        = os.path.join(BASE_DIR, "data", "recent", "recent_data.parquet")
DRIFT_REPORT_PATH       = os.path.join(REPORTS_PATH, "drift_report.json")


def load_reference_data():
    df_ref = pd.read_parquet(REFERENCE_DATA_PATH)
    with open(REFERENCE_FEATURES_PATH, "r") as f:
        features = json.load(f)
    return df_ref, features


def load_recent_data(features):
    if not os.path.exists(RECENT_DATA_PATH):
        raise FileNotFoundError(f"Recent data not found: {RECENT_DATA_PATH}")
    df = pd.read_parquet(RECENT_DATA_PATH)
    return df[features]


def get_production_run_id(model_name: str):
    client = MlflowClient()
    # Try registry stage first
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].run_id
    except Exception:
        pass
    # Fallback: latest final-training run
    exp = client.get_experiment_by_name("Attrition Final Model - logistic_regression")
    if exp:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
    raise ValueError(f"No production or training run found for {model_name}")


def run_drift_detection():
    # Load data
    df_ref, features = load_reference_data()
    df_recent = load_recent_data(features)

    # Execute data drift detection
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=df_ref,
        current_data=df_recent,
        column_mapping=ColumnMapping(),
    )
    report_dict = report.as_dict()

    # Save drift report
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"📄 Drift report saved to {DRIFT_REPORT_PATH}")

    # Evaluate retraining triggers
    nf = 0
    share = 0.0
    for m in report_dict.get("metrics", []):
        if not isinstance(m, dict):
            continue
        metric_info = m.get("metric")
        if not isinstance(metric_info, dict):
            continue
        name = metric_info.get("name", "")
        result = m.get("result", {})
        if name == "Number of drifted features":
            nf = result.get("number_of_drifted_features", 0)
        elif name == "Dataset drift":
            share = result.get("data_drift_share", 0.0)

    print(f"Detected {nf} drifted features; threshold is {RETRAIN_TRIGGER_FEATURE_COUNT}.")
    print(f"Dataset drift share: {share}; threshold is {RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE}.")

    retrain = (nf >= RETRAIN_TRIGGER_FEATURE_COUNT) or (share >= RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE)

    # Log the artifact to MLflow
    run_id = get_production_run_id(MODEL_NAME)
    with mlflow.start_run(run_name="drift_monitor", nested=False):
        try:
            mlflow.log_artifact(DRIFT_REPORT_PATH, artifact_path="drift_reports")
        except Exception as e:
            print(f"⚠️ Could not log artifact: {e}")
    print("✅ Drift report logged to MLflow.")

    # Exit with code: 2 for retrain needed, 0 for OK
    if retrain:
        print("🚨 Retraining needed. Exiting with code 2.")
        sys.exit(2)
    else:
        print("✔️ No retraining needed. Exiting with code 0.")
        sys.exit(0)


if __name__ == "__main__":
    run_drift_detection()
