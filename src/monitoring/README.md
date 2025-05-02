# Monitoring Module

This module contains tools for monitoring ML models in production, with a focus on drift detection.

## Drift Detection

The `drift_detection.py` file implements the core drift detection functionality:

- `DriftDetector` class for both feature and prediction drift detection
- Integration with Evidently for statistical testing
- MLflow tracking for drift metrics and history
- Support for custom drift thresholds

### Usage Example

```python
from monitoring.drift_detection import DriftDetector
import pandas as pd

# Load data
reference_data = pd.read_parquet("drift_reference/reference_train_data.parquet")
current_data = pd.read_csv("path/to/new/data.csv")

# Create detector
detector = DriftDetector(drift_threshold=0.05, mlflow_tracking=True)

# Detect drift
drift_detected, drift_score, drifted_features = detector.detect_drift(
    reference_data=reference_data,
    current_data=current_data
)

# Print results
print(f"Drift detected: {drift_detected}")
print(f"Drift score: {drift_score}")
print(f"Drifted features: {drifted_features}")
```

### Implementation Details

The drift detection implementation uses:

1. **Column-level tests**: Statistical tests for each feature to detect distribution changes
2. **Overall drift measures**: Percentage of features showing drift
3. **Optimized detection**: Automatic handling of numerical vs categorical features

For a comprehensive guide on using drift detection, see the [Drift Detection Guide](../../docs/drift_detection_guide.md).

## Related Files

- `drift_api.py`: FastAPI implementation for drift detection as a service
- `save_reference_data.py`: Script to create and save reference data
- `check_production_drift.py`: Script to check production data for drift
- `test_drift_api_client.py`: Client for testing the drift detection API 