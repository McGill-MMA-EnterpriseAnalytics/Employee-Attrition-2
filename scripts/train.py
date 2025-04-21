# scripts/train.py
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
# Import model classes needed based on potential best params
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
import os

# ensure Python can import from src/employee_attrition_mlops
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your source code
from employee_attrition_mlops.config import ( # Corrected package name
    RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    MLFLOW_TRACKING_URI, DEFAULT_EXPERIMENT_NAME, PRODUCTION_MODEL_NAME, # Use PRODUCTION_MODEL_NAME for registration
    BASELINE_PROFILE_FILENAME, REPORTS_PATH # Added REPORTS_PATH
)
from employee_attrition_mlops.data_processing import ( # Corrected package name
    load_and_clean_data, identify_column_types, find_skewed_columns,
    AddNewFeaturesTransformer
)
from employee_attrition_mlops.pipelines import create_preprocessing_pipeline, create_full_pipeline # Corrected package name
from employee_attrition_mlops.utils import save_json, load_json, generate_profile_dict # Corrected package name

# Map model type string to actual class (must match hpo.py)
CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

os.makedirs(REPORTS_PATH, exist_ok=True)


def train_final_model(params_file: str, register_model_as: str = None):
    """
    Loads best parameters from HPO, trains the final model on full training data,
    evaluates on test set, logs to MLflow, and optionally registers the model.
    """
    logger.info(f"Starting final model training using parameters from: {params_file}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # --- Load Best Parameters from HPO ---
    params = load_json(params_file)
    if params is None:
        logger.error(f"Could not load parameters from {params_file}. Exiting.")
        return

    logger.info(f"Loaded best parameters: {params}")

    # Extract parameters
    model_type = params.get('model_type')
    if not model_type or model_type not in CLASSIFIER_MAP:
        logger.error(f"Invalid or missing 'model_type' in parameters file: {model_type}")
        return
    classifier_class = CLASSIFIER_MAP[model_type]

    # Separate pipeline and model params based on prefixes used in HPO objective
    pipeline_params = {k.replace('pipe_', ''): v for k, v in params.items() if k.startswith('pipe_')}
    model_params = {k.replace('model_', ''): v for k, v in params.items() if k.startswith('model_')}
    
    # HPO may have returned 'penalty_saga' when using solver='saga'
    if 'penalty_saga' in model_params:
        # rename it to the correct sklearn arg 'penalty'
        model_params['penalty'] = model_params.pop('penalty_saga')
    # Remove keys that are not valid sklearn init args
    for drop in ['type', 'best_f2_cv_mean']:
        model_params.pop(drop, None)
    
    feature_selector_params = {k.replace('selector_', ''): v for k, v in params.items() if k.startswith('selector_')}
    # Add selector params under the main pipeline params dict for create_full_pipeline
    pipeline_params['feature_selector_params'] = feature_selector_params

    experiment_name = f"Attrition Final Model - {model_type}" # Set experiment based on model type
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_type}_final_training"):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID for final training: {run_id}")

        # Log HPO parameters used for this final model
        mlflow.log_params({f"hpo_best_{k}": v for k,v in params.items()})
        mlflow.log_param("source_params_file", params_file)
        mlflow.log_param("final_model_type", model_type)

        # --- Load and Prepare Data ---
        df = load_and_clean_data(RAW_DATA_PATH)
        feature_adder = AddNewFeaturesTransformer()
        df = feature_adder.fit_transform(df)
        col_types = identify_column_types(df, TARGET_COLUMN)
        numerical_cols = col_types['numerical']
        categorical_cols = col_types['categorical']
        ordinal_cols = col_types['ordinal']
        business_travel_col = col_types['business_travel']
        skewed_cols = find_skewed_columns(df, numerical_cols)
        mlflow.log_param("skewed_features_count", len(skewed_cols))

        # --- Train/Test Split ---
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN] # Already encoded
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Using Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # --- Create Pipelines using BEST HPO Params ---
        preprocessor = create_preprocessing_pipeline(
            numerical_cols=numerical_cols, categorical_cols=categorical_cols,
            ordinal_cols=ordinal_cols, business_travel_col=business_travel_col,
            skewed_cols=skewed_cols,
            # Get pipeline step choices from loaded params
            numeric_transformer_type=pipeline_params.get('num_transform'),
            numeric_scaler_type=pipeline_params.get('num_scaler'),
            business_encoder_type=pipeline_params.get('bt_encoder'),
        )

        full_pipeline = create_full_pipeline(
            classifier_class=classifier_class,
            model_params=model_params,
            preprocessor=preprocessor,
            feature_selector_type=pipeline_params.get('selector'),
            feature_selector_params=pipeline_params.get('feature_selector_params'),
            smote_active=pipeline_params.get('smote'),
        )
        logger.info("Created final pipeline using best HPO parameters.")

        # --- Training on FULL Training Data ---
        logger.info("Fitting final pipeline on full training data...")
        full_pipeline.fit(X_train, y_train)
        logger.info("Pipeline fitting complete.")
        
        # --- 4.1: Baseline profile ---
    try:
        import tensorflow_data_validation as tfdv
        # generate a TFDV statistics proto
        stats = tfdv.generate_statistics_from_dataframe(df)
        stats_path = os.path.join(REPORTS_PATH, "baseline_stats.pbtxt")
        tfdv.write_stats_text(stats, stats_path)
        mlflow.log_artifact(stats_path, artifact_path="baseline_profile")
        logger.info(f"Logged TFDV baseline profile to {stats_path}")
    except ImportError:
        # fallback to pandas if TFDV is unavailable
        profile = df.describe(include="all").to_json()
        stats_path = os.path.join(REPORTS_PATH, "baseline_profile.json")
        with open(stats_path, "w") as f:
            f.write(profile)
        mlflow.log_artifact(stats_path, artifact_path="baseline_profile")
        logger.info(f"Logged pandas baseline profile to {stats_path}")


        # --- Generate & Log Training Data Profile & Reference Data ---
        # (Copy the artifact logging code from train_py_v2 here)
        try:
            preprocessor_fitted = full_pipeline.named_steps['preprocessor']
            X_train_processed = preprocessor_fitted.transform(X_train)
            try: feature_names = preprocessor_fitted.get_feature_names_out()
            except Exception: feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
            X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)

            profile_dict = generate_profile_dict(X_train_processed_df)
            profile_path = os.path.join(REPORTS_PATH, BASELINE_PROFILE_FILENAME)
            save_json(profile_dict, profile_path)
            mlflow.log_artifact(profile_path, artifact_path="drift_reference")
            logger.info(f"Logged training data profile to {profile_path}")

            ref_data_path = "reference_train_data.parquet"
            X_train_processed_df.to_parquet(ref_data_path, index=False)
            mlflow.log_artifact(ref_data_path, artifact_path="drift_reference")
            logger.info(f"Logged reference training data to {ref_data_path}")

            ref_features_path = os.path.join(REPORTS_PATH, "reference_feature_names.json")
            save_json(list(X_train_processed_df.columns), ref_features_path)
            mlflow.log_artifact(ref_features_path, artifact_path="drift_reference")
            logger.info(f"Logged reference feature names to {ref_features_path}")
        except Exception as e:
            logger.error(f"Error logging profile/reference data: {e}", exc_info=True)


        # --- Evaluation on Test Set ---
        logger.info("Evaluating final model on test set...")
        y_pred_test = full_pipeline.predict(X_test)
        metrics = { # Calculate final test metrics
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_f1": f1_score(y_test, y_pred_test, pos_label="Yes", zero_division=0),
            "test_f2": fbeta_score(y_test, y_pred_test, beta=2, pos_label="Yes", zero_division=0),
            "test_precision": precision_score(y_test, y_pred_test, pos_label="Yes", zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, pos_label="Yes", zero_division=0),
        }
        if hasattr(full_pipeline, "predict_proba"):
            try:
                y_test_proba = full_pipeline.predict_proba(X_test)[:, 1]
                metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
            except Exception as e: logger.warning(f"Could not calculate AUC: {e}"); metrics["test_auc"] = -1
        else: metrics["test_auc"] = -1

        mlflow.log_metrics(metrics)
        logger.info(f"Logged final test metrics: {metrics}")

        # --- Log Baseline Test Predictions ---
        # (Copy the baseline prediction logging code from train_py_v2 here)
        try:
            baseline_preds_path = os.path.join(REPORTS_PATH, "baseline_test_predictions.json")
            preds_data = {"predictions": y_pred_test.tolist(), "true_labels": y_test.tolist()}
            save_json(preds_data, baseline_preds_path)
            mlflow.log_artifact(baseline_preds_path, artifact_path="drift_reference")
            logger.info(f"Logged baseline test predictions to {baseline_preds_path}")
        except Exception as e: logger.error(f"Could not log baseline test predictions: {e}")


        # --- Log Confusion Matrix & Feature Importance ---
        # (Copy the CM and Feature Importance logging code from train_py_v2 here)
        try: # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test)
            cm_path = os.path.join(REPORTS_PATH, "confusion_matrix.json")
            save_json({"labels": [0, 1], "matrix": cm.tolist()}, cm_path)
            mlflow.log_artifact(cm_path)
        except Exception as e: logger.error(f"Could not log confusion matrix: {e}")

        try: # Feature Importance (adapt based on final pipeline structure)
             # ... (logic copied and adapted from train_py_v2's feature importance section) ...
             # Make sure to get feature names *after* preprocessing AND feature selection
             final_estimator = full_pipeline.named_steps['classifier']
             # ... rest of importance logic ...
             pass # Placeholder for copied logic
        except Exception as e: logger.error(f"Could not log feature importances: {e}")


        # --- Log and Register Final Model ---
        artifact_path = f"{model_type}_final_pipeline" # Define artifact path
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path=artifact_path,
            # Register the model if a name is provided via CLI flag
            registered_model_name=register_model_as
        )
        if register_model_as:
            # Immediately transition the newly created version to Staging
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            # fetch the latest version just registered
            versions = client.get_latest_versions(register_model_as, stages=[])
            new_version = versions[-1].version
            client.transition_model_version_stage(
                name=register_model_as,
                version=new_version,
                stage="Staging",
                archive_existing_versions=False
            )
            logger.info(f"Registered model version {new_version} to 'Staging'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Final Employee Attrition Model using Best HPO Params")
    parser.add_argument(
        "--params-file",
        type=str,
        default=os.path.join(REPORTS_PATH, "best_hpo_params.json"), # Default to file saved by hpo.py
        help="Path to the JSON file containing the best parameters found by HPO.",
    )
    parser.add_argument(
        "--register-as",
        type=str,
        default=None, # Default: Do not register
        help=f"Register the trained model with this name. Defaults to None. Example: {PRODUCTION_MODEL_NAME}",
    )
    args = parser.parse_args()

    train_final_model(params_file=args.params_file, register_model_as=args.register_as)

