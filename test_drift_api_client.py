#!/usr/bin/env python3
"""
Test client for drift detection API.

This script loads test data from a JSON file and sends it to the API
to test drift detection.
"""

import requests
import json
import sys
from pathlib import Path
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"
TEST_DATA_PATH = "temp/test_data.json"

def main():
    print("Drift Detection API Test Client")
    print("==============================")
    
    # 1. Check API health
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"API status: {health_data.get('status', 'unknown')}")
            if 'reference_data_shape' in health_data:
                print(f"Reference data shape: {health_data['reference_data_shape']}")
        else:
            print(f"Error: HTTP status {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return
        
    # 2. Load test data
    print("\n2. Loading test data...")
    try:
        # Try to load from JSON file
        test_data_file = Path(TEST_DATA_PATH)
        if test_data_file.exists():
            with open(test_data_file, 'r') as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} records from {TEST_DATA_PATH}")
        else:
            # No test data file, create some sample data
            print(f"Test data file {TEST_DATA_PATH} not found, creating sample data")
            
            # Create minimal sample data
            test_data = [
                {"Age": 45, "MonthlyIncome": 150000, "Department": "Sales"},
                {"Age": 55, "MonthlyIncome": 180000, "Department": "Engineering"},
                {"Age": 35, "MonthlyIncome": 120000, "Department": "HR"},
                {"Age": 40, "MonthlyIncome": 140000, "Department": "Sales"},
                {"Age": 50, "MonthlyIncome": 160000, "Department": "Engineering"}
            ]
            print(f"Created {len(test_data)} sample records")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # 3. Detect drift
    print("\n3. Detecting drift...")
    try:
        # Prepare the request
        request_data = {
            "data": test_data,
            "threshold": 0.05
        }
        
        # Send request
        response = requests.post(
            f"{API_URL}/detect-drift",
            json=request_data
        )
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            print("\nDrift Detection Results:")
            print(f"Drift detected: {result.get('drift_detected')}")
            print(f"Drift score: {result.get('drift_score')}")
            print(f"Number of drifted features: {result.get('n_drifted_features')}")
            if 'drifted_features' in result and result['drifted_features']:
                print(f"Drifted features: {', '.join(result['drifted_features'])}")
            else:
                print("No features drifted significantly")
        else:
            print(f"Error: HTTP status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling API: {e}")
        return
        
    print("\nTest completed.")

if __name__ == "__main__":
    main() 