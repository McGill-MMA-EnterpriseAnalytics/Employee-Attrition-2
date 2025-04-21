"""
Promote a candidate model from Staging to Production.
Usage: python scripts/promote_model.py
"""
import sys
from mlflow.tracking import MlflowClient

MODEL_NAME = "AttritionProductionModel"

def list_staging(client):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not versions:
        print("🚫 No versions in Staging.")
        sys.exit(1)
    print("📦 Versions currently in Staging:")
    for v in versions:
        mv = client.get_model_version(MODEL_NAME, v.version)
        ts = mv.creation_timestamp
        print(f"  • Version {v.version}  | Run ID {v.run_id}  | Created {ts}")
    return [str(v.version) for v in versions]

def promote(client, version):
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Version {version} promoted to Production")

if __name__ == "__main__":
    client = MlflowClient()
    staging_versions = list_staging(client)
    choice = input("Enter version to promote (or 'q' to quit): ").strip()
    if choice.lower() == "q":
        sys.exit(0)
    if choice not in staging_versions:
        print(f"❌ Version {choice} not found in Staging.")
        sys.exit(1)
    promote(client, choice)
