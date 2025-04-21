#!/usr/bin/env bash
#
# Orchestrate data‐drift monitoring & conditional retraining dispatch.
# Usage: 
#   1. Activate your venv:
#        source .venv/bin/activate 
#   2. Run this script:
#        ./scripts/run_monitor_and_retrain.sh

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_drift.py"

# ─── Step 1: Run data‐drift detection ────────────────────────────────────────────
echo "=== Step 1: Running data drift detection ==="
python "$MONITOR_SCRIPT"
exit_code=$?

# ─── Step 2: Conditional retraining trigger ─────────────────────────────────────
if [ "$exit_code" -eq 2 ]; then
  echo "🚨 Drift detected (exit code 2) → dispatching retraining workflow"

  python - << 
  'PYCODE'
import os
import sys

# Make project code importable
root = os.getcwd()
sys.path.insert(0, os.path.join(root, "src"))
sys.path.insert(0, os.path.join(root, "scripts"))

from retrain_utils import trigger_retraining_workflow
trigger_retraining_workflow()
PYCODE

elif [ "$exit_code" -eq 0 ]; then
  echo "✔️ No retraining needed (exit code 0)"
else
  echo "⚠️ Unexpected exit code: $exit_code – aborting"
  exit "$exit_code"
fi

echo "✅ Orchestration complete."