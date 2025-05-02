# Drift Reference Data

This directory stores reference data for drift detection.

Reference data acts as a baseline against which new production data is compared to detect drift. It's typically created from your training dataset.

## Files stored here

- `reference_data_*.parquet`: Timestamped reference data snapshots
- `reference_train_data.parquet`: The most recent reference data (symlink)
- Other metadata files related to reference data

## Usage

Reference data is automatically created when you run:

```bash
python save_reference_data.py
``` 