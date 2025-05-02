from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
import pandas as pd
import mlflow
import logging
import os
import sys
import json
from datetime import datetime
from config.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DriftDetector:
    """
    Class for detecting drift between reference and current data.
    
    This class provides methods to detect both feature and prediction drift
    using Evidently's statistical tests.
    """
    
    def __init__(self, drift_threshold=0.05, mlflow_tracking=False):
        """
        Initialize the drift detector.
        
        Args:
            drift_threshold: Threshold for considering drift significant (default: 0.05)
            mlflow_tracking: Whether to log results to MLflow (default: False)
        """
        self.drift_threshold = drift_threshold
        self.mlflow_tracking = mlflow_tracking
    
    def detect_drift(self, reference_data, current_data, features=None):
        """
        Detect drift between reference and current data.
        
        Args:
            reference_data: Reference dataset (pandas DataFrame)
            current_data: Current dataset (pandas DataFrame)
            features: List of features to check for drift (if None, all common columns are used)
            
        Returns:
            Tuple of (drift_detected, drift_score, drifted_features)
        """
        # Determine numerical and categorical features
        if features is None:
            # Use common columns
            common_cols = list(set(reference_data.columns) & set(current_data.columns))
            features = common_cols
        
        # Split features by type
        numerical_features = []
        categorical_features = []
        
        for feature in features:
            if feature in reference_data.columns and feature in current_data.columns:
                if reference_data[feature].dtype.kind in 'fc':  # float or complex
                    numerical_features.append(feature)
                else:
                    categorical_features.append(feature)
        
        # Run drift detection
        results = detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            prediction_column=None,
            mlflow_tracking=self.mlflow_tracking,
            drift_threshold=self.drift_threshold
        )
        
        return (
            results.get('drift_detected', False),
            results.get('drift_score', 0.0),
            results.get('drifted_features', [])
        )
    
    def detect_prediction_drift(self, reference_data, current_data, prediction_column):
        """
        Detect drift in model predictions.
        
        Args:
            reference_data: Reference dataset with predictions
            current_data: Current dataset with predictions
            prediction_column: Name of the prediction column
            
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        # Ensure prediction column exists
        if prediction_column not in reference_data.columns or prediction_column not in current_data.columns:
            logger.error(f"Prediction column '{prediction_column}' not found in data")
            return False, 0.0
        
        # Run drift detection with only prediction column
        results = detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_features=[],
            categorical_features=[],
            prediction_column=prediction_column,
            mlflow_tracking=self.mlflow_tracking,
            drift_threshold=self.drift_threshold
        )
        
        return (
            results.get('drift_detected', False),
            results.get('drift_score', 0.0)
        )

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    prediction_column: str = None,
    mlflow_tracking: bool = False,
    drift_threshold: float = 0.05  # Lower default threshold for detecting drift
) -> dict:
    """
    Detect drift between reference and current data using Evidently.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset to compare against reference
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        prediction_column: Name of the prediction column (optional)
        mlflow_tracking: Whether to log results to MLflow
        drift_threshold: Threshold for considering drift significant
        
    Returns:
        Dictionary containing drift detection results
    """
    try:
        # Set up column mapping
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = numerical_features
        column_mapping.categorical_features = categorical_features
        
        if prediction_column:
            column_mapping.prediction = prediction_column
        
        # Create test suite
        tests = []
        
        # Log the parameters we're testing with
        logger.info(f"Setting up drift detection with threshold: {drift_threshold}")
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Prediction column: {prediction_column}")
        
        # Add column drift tests for each feature
        for feature in numerical_features + categorical_features:
            logger.info(f"Adding column drift test for feature: {feature}")
            tests.append(
                TestColumnDrift(
                    column_name=feature,
                    stattest_threshold=drift_threshold
                )
            )
        
        # Add prediction drift test if applicable
        if prediction_column:
            tests.append(
                TestColumnDrift(
                    column_name=prediction_column,
                    stattest_threshold=drift_threshold
                )
            )
        
        # Add overall drift test
        tests.append(
            TestShareOfDriftedColumns(
                lte=drift_threshold  # Use lte parameter instead of drift_share_threshold
            )
        )
        
        # Create and run test suite
        test_suite = TestSuite(tests=tests)
        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Get results
        results = test_suite.as_dict()
        
        # Debug log the full results structure
        logger.info(f"Test suite completed with {len(results.get('tests', []))} tests")
        logger.info(f"Test suite results summary: {results.get('summary', {})}")
        
        # Dump the entire results structure for debugging
        if logger.getEffectiveLevel() <= logging.DEBUG:
            try:
                logger.debug(f"Full test results: {json.dumps(results, indent=2)}")
            except Exception as e:
                logger.debug(f"Could not dump full results: {str(e)}")
        
        # Calculate drift metrics
        drift_metrics = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'test_results': {}
        }
        
        # Process test results
        for i, test in enumerate(results.get('tests', [])):
            test_name = test.get('name', f'Unknown Test {i}')
            test_status = test.get('status', 'ERROR')
            
            # Log test status and name
            logger.info(f"Test {i+1}: '{test_name}' status: {test_status}")
            
            # Extract detailed test parameters
            test_params = test.get('parameters', {})
            if test_params:
                logger.info(f"Test '{test_name}' parameters: {test_params}")
            else:
                logger.info(f"Test '{test_name}' has no parameters")
                
            # Look for test data and results section
            test_data = test.get('results', {})
            if test_data:
                logger.info(f"Test '{test_name}' results section available")
            
            drift_metrics['test_results'][test_name] = {
                'status': test_status,
                'details': test_params
            }
            
            # Check if drift is detected
            if test_status == 'FAIL':
                drift_metrics['drift_detected'] = True
                
                # Extract feature name from test name
                feature_name = None
                if 'Column Drift Test' in test_name or 'Drift per Column' in test_name:
                    # Try different patterns to extract feature name
                    if ' for ' in test_name:
                        feature_name = test_name.split(' for ')[1].strip()
                    else:
                        feature_name = test_name.replace('Column Drift Test', '').strip()
                        feature_name = feature_name.replace('Drift per Column', '').strip()
                        # Look in parameters if the feature name is empty
                        if not feature_name and 'column_name' in test_params:
                            feature_name = test_params['column_name']
                            
                    logger.info(f"Extracted feature name: '{feature_name}' from test name: '{test_name}'")
                    
                    if feature_name and feature_name not in drift_metrics['drifted_features']:
                        drift_metrics['drifted_features'].append(feature_name)
                        logger.info(f"Added feature '{feature_name}' to drifted features list")
                
                # Calculate drift score based on test parameters
                # Look for various score-related fields in test parameters
                score = None
                
                # First try to get the score directly
                if 'score' in test_params:
                    score = test_params['score']
                    logger.info(f"Found direct score value: {score}")
                    
                    # If score is a p-value (closer to 0 means more drift), convert it
                    if 'stattest' in test_params and 'p_value' in test_params['stattest']:
                        score = 1.0 - score
                        logger.info(f"Converted p-value to drift score: {score}")
                
                # If no direct score, try p_value
                elif 'p_value' in test_params:
                    score = 1.0 - test_params['p_value']  # Convert p-value to drift score
                    logger.info(f"Calculated drift score from p_value: {score}")
                
                # For the share of drifted columns, look for share parameter
                elif 'share' in test_params:
                    score = test_params['share']
                    logger.info(f"Using drift share as score: {score}")
                
                # For features section, take max drift
                elif 'features' in test_params:
                    feature_scores = []
                    for feat, feat_details in test_params['features'].items():
                        if 'score' in feat_details:
                            feat_score = feat_details['score']
                            # Convert p-value scores
                            if 'stattest' in feat_details and 'p_value' in feat_details['stattest']:
                                feat_score = 1.0 - feat_score
                            feature_scores.append(feat_score)
                    
                    if feature_scores:
                        score = max(feature_scores)
                        logger.info(f"Calculated max feature score: {score} from {len(feature_scores)} features")
                
                # If score was calculated, update the drift score if higher
                if score is not None:
                    drift_metrics['drift_score'] = max(drift_metrics['drift_score'], float(score))
                    logger.info(f"Updated drift score to: {drift_metrics['drift_score']}")
        
        # Final drift detection result
        logger.info(f"Final drift detection results: " + 
                   f"drift_detected={drift_metrics['drift_detected']}, " +
                   f"drift_score={drift_metrics['drift_score']}, " +
                   f"drifted_features={drift_metrics['drifted_features']}")
        
        # Log to MLflow if requested
        if mlflow_tracking:
            try:
                # Log drift metrics
                mlflow.log_metric("drift_detected", int(drift_metrics['drift_detected']))
                mlflow.log_metric("drift_score", drift_metrics['drift_score'])
                mlflow.log_metric("n_drifted_features", len(drift_metrics['drifted_features']))
                
                # Log drifted features as a parameter
                mlflow.log_param("drifted_features", ",".join(drift_metrics['drifted_features']))
                
                # Log test results as a JSON file
                with open("drift_test_results.json", "w") as f:
                    json.dump(drift_metrics['test_results'], f, indent=2)
                mlflow.log_artifact("drift_test_results.json")
                
                logger.info("Logged drift metrics to MLflow")
            except Exception as e:
                logger.error(f"Error logging to MLflow: {str(e)}")
        
        return drift_metrics
    
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'error': str(e)
        }

def detect_drift_old(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                numerical_features: list = None, categorical_features: list = None,
                target_column: str = None, prediction_column: str = None,
                save_results: bool = True, mlflow_tracking: bool = False) -> dict:
    """
    Legacy function for drift detection, maintained for backward compatibility.
    """
    try:
        # Set up column mapping
        column_mapping = ColumnMapping()
        
        if numerical_features:
            column_mapping.numerical_features = numerical_features
        if categorical_features:
            column_mapping.categorical_features = categorical_features
        if target_column:
            column_mapping.target = target_column
        if prediction_column:
            column_mapping.prediction = prediction_column
            
        # Create drift test suite
        tests = [DataDriftTestPreset()]
        if prediction_column:
            tests.append(DataQualityTestPreset())
            
        drift_test_suite = TestSuite(tests=tests)
        
        # Run drift detection
        drift_test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Get results
        results = drift_test_suite.as_dict()
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = settings.DRIFT_ARTIFACTS_DIR / f"drift_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Drift detection results saved to {results_file}")
            
            # Log to MLflow if enabled
            if mlflow_tracking:
                try:
                    # Log drift metrics
                    mlflow.log_metric("drift_detected", int(not results['summary']['all_passed']))
                    mlflow.log_metric("failed_tests", sum(1 for test in results['tests'] if test['status'] == 'FAIL'))
                    
                    # Log individual test results
                    for test in results['tests']:
                        mlflow.log_metric(f"test_{test['name']}_status", 
                                       1 if test['status'] == 'PASS' else 0)
                    
                    # Log drift artifacts
                    mlflow.log_artifact(str(results_file))
                    
                    logger.info("Drift detection results logged to MLflow")
                except Exception as e:
                    logger.error(f"Error logging to MLflow: {str(e)}")
        
        # Log results
        logger.info(f"Drift detection completed. All tests passed: {results['summary']['all_passed']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise 