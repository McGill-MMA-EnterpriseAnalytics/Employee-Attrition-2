#!/usr/bin/env python3
"""
Simplified FastAPI endpoint for drift detection.
This file should be placed in the project root for easier execution.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_api")

# Make sure src is in the path
sys.path.append(".")

# Import our drift detector
from src.monitoring.drift_detection import DriftDetector

# Create FastAPI app
app = FastAPI(
    title="Employee Attrition Drift Detection API",
    description="API for detecting data drift in employee attrition data",
    version="1.0.0"
)

# Configure paths
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Request/response models
class DriftRequest(BaseModel):
    data: List[Dict[str, Any]]
    features: Optional[List[str]] = None
    threshold: Optional[float] = 0.05

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    drifted_features: List[str]
    n_drifted_features: int
    timestamp: str

# Helper functions
def load_reference_data():
    """Load reference data from parquet file."""
    reference_path = Path("drift_reference/reference_train_data.parquet")
    if not reference_path.exists():
        logger.error(f"Reference data not found at {reference_path}")
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
    
    return pd.read_parquet(reference_path)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Employee Attrition Drift Detection API"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        reference_data = load_reference_data()
        return {"status": "healthy", "reference_data_shape": reference_data.shape}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/detect-drift", response_model=DriftResponse)
async def detect_drift(request: DriftRequest):
    """Detect drift from JSON data."""
    try:
        logger.info(f"Received drift detection request with {len(request.data)} records")
        
        # Convert JSON data to DataFrame
        data = pd.DataFrame(request.data)
        logger.info(f"Converted to DataFrame with shape: {data.shape}")
        
        # Load reference data
        reference_data = load_reference_data()
        
        # Create detector
        detector = DriftDetector(drift_threshold=request.threshold)
        
        # Run drift detection
        drift_detected, drift_score, drifted_features = detector.detect_drift(
            reference_data=reference_data,
            current_data=data,
            features=request.features
        )
        
        # Prepare response
        response = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "n_drifted_features": len(drifted_features),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Drift detected: {drift_detected}, score: {drift_score}")
        return response
    except Exception as e:
        logger.exception(f"Error in drift detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 