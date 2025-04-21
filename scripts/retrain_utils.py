import os
import requests
from employee_attrition_mlops.config import (
    GITHUB_ACTIONS_PAT_ENV_VAR,
    GITHUB_OWNER_REPO,
    RETRAIN_WORKFLOW_ID,
)

def trigger_retraining_workflow(ref_branch: str = "develop"):
    token = os.getenv(GITHUB_ACTIONS_PAT_ENV_VAR)
    if not token:
        raise RuntimeError(f"Missing PAT in env var {GITHUB_ACTIONS_PAT_ENV_VAR}")
    url = (
        f"https://api.github.com/repos/{GITHUB_OWNER_REPO}"
        f"/actions/workflows/{RETRAIN_WORKFLOW_ID}/dispatches"
    )
    resp = requests.post(
        url,
        json={"ref": ref_branch},
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {token}",
        },
    )
    if resp.status_code != 204:
        raise RuntimeError(f"Dispatch failed: {resp.status_code} {resp.text}")
    print("✅ Retraining workflow dispatched.")
