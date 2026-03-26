## Part 16: Logging to MLflow

Everything that went into production should be tracked. Log the relativities, validation results, and model state to MLflow.

In a new cell, type this and run it (Shift+Enter):

```python
import mlflow
from datetime import date

mlflow.set_experiment("/Users/your-username/pricing-module-04")

with mlflow.start_run(run_name="gbm_shap_relativities_v1") as run:
    freq_run_id = run.info.run_id  # capture for linking downstream tables

    # Log model parameters
    mlflow.log_params(freq_params)

    # Log validation results as metrics
    for check_name, result in checks.items():
        mlflow.log_metric(f"validation_{check_name}_passed", int(result.passed))

    # Log reconstruction error specifically
    recon_result = checks["reconstruction"]
    if hasattr(recon_result, "value"):
        mlflow.log_metric("reconstruction_max_error", float(recon_result.value))

    # Log the relativities table as a CSV artefact
    rels_path = "/tmp/gbm_relativities.csv"
    rels.to_csv(rels_path, index=False)
    mlflow.log_artifact(rels_path, "relativities")

    # Log Radar export files
    mlflow.log_artifact("/dbfs/tmp/gbm_relativities_radar.csv", "radar_export")
    mlflow.log_artifact("/dbfs/tmp/gbm_age_band_relativities_radar.csv", "radar_export")

    # Log key relativities as metrics for easy comparison
    area_f_rel = rels[rels["feature"] == "area"].set_index("level").loc["F", "relativity"]
    ncd5_rel   = rels[rels["feature"] == "ncd_years"].set_index("level").loc[5, "relativity"]
    conv_rel   = rels[rels["feature"] == "has_convictions"].set_index("level").loc[1, "relativity"]

    mlflow.log_metric("area_F_relativity",        area_f_rel)
    mlflow.log_metric("ncd5_relativity",           ncd5_rel)
    mlflow.log_metric("conviction_relativity",     conv_rel)
    mlflow.log_metric("n_policies",               len(df))

    # Log the CatBoost model itself
    mlflow.catboost.log_model(freq_model, "catboost_model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Area F relativity logged:    {area_f_rel:.4f}")
    print(f"NCD=5 relativity logged:     {ncd5_rel:.4f}")
    print(f"Conviction relativity logged:{conv_rel:.4f}")
```

You will see the run ID printed and the key metrics. Go to the MLflow UI (click the flask icon in the Databricks left sidebar, then Experiments) to verify the run was logged. You should see:

- The model parameters (loss function, learning rate, etc.)
- The validation pass/fail metrics
- The key relativities as searchable metrics
- The relativities CSV and Radar export files as downloadable artefacts

**Why log relativities to MLflow and not just Unity Catalog?** MLflow gives you the link between the specific model run (the exact training data, the exact hyperparameters) and the relativities derived from it. If someone later asks "which model version did the area F = 1.95 relativity come from?", you can find it in MLflow. Unity Catalog stores the relativity table for querying; MLflow stores the audit trail.