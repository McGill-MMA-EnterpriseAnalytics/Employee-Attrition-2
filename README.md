
## 4.4 Post‑Retraining Governance

### 1. inspecting Staging Models
After an automated retrain, new versions land in **Staging**:

1. Open MLflow UI → **Models** → `AttritionProductionModel`
2. Switch to the **Staging** tab to review metrics, XAI/fairness reports, and drift artifacts.

### 2. Promote to Production
Once satisfied, run:

```bash
scripts/promote_model.py
