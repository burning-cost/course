## Part 4: Pipeline architecture — what connects to what

The pipeline has ten stages. Every stage reads from a named input and writes to a named output. The names are registered in the configuration dictionary in Stage 1 — change a table name there and it changes everywhere.

```
Stage 1: Configuration
    |
    v
Stage 2: Data ingestion ---------> raw_policies (Delta, version logged)
    |
    v
Stage 3: Feature engineering ----> features (Delta, version logged)
    |
    +---> Stage 4: Walk-forward CV (cv_deviances, mean_cv_deviance)
    |         |
    |         v
    +---> Stage 5: Optuna tuning (best_freq_params, best_sev_params)
    |         |
    |         v
    +---> Stage 6: Final models ---> freq_model (MLflow), sev_model (MLflow)
                                     freq_predictions (Delta)
              |
              +---> Stage 7: SHAP relativities -----> freq_relativities (Delta)
              |
              +---> Stage 8: Conformal intervals ----> conformal_intervals (Delta)
              |
              +---> Stage 8.5: Calibration testing --> calibration verdict
              |
              +---> Stage 9: Rate optimisation ------> rate_action_factors (Delta)
                                                       efficient_frontier (Delta)
                        |
                        v
                   Stage 10: Audit record -----------> pipeline_audit (Delta, append)
```

Three design decisions are worth making explicit.

**Stage 3 feeds Stages 4, 5, and 6 from the same table.** Cross-validation, hyperparameter tuning, and final model training all read from the features Delta table. They do not re-run the transforms. This means the CV metrics reflect the same feature encoding as the final model. If you want to experiment with a new feature, add it to Stage 3, re-run from Stage 3 onwards, and the metrics are comparable.

**Stage 6 is the single source of model outputs.** Stages 7, 8, and 9 all consume the trained models from MLflow. They do not retrain. They do not generate new predictions independently. This means the SHAP relativities, the conformal intervals, and the rate optimisation all work from the same model — which is the model the pricing committee approved.

**Stage 10 appends, never overwrites.** The audit table grows one row per pipeline run. Multiple runs in the same review cycle produce multiple rows, each with its run date and configuration. This is the correct mode because you may run the pipeline several times before presenting to the pricing committee — with different Optuna trial counts, different IBNR buffer settings, or updated data. All of those runs should be preserved. `mode("overwrite")` would silently destroy the history.

### Identifying the pipeline run

Every downstream table carries two identifiers from the run that produced it:

```python
"mlflow_run_id": freq_run_id   # identifies the model artefact in MLflow
"run_date":      RUN_DATE      # date the pipeline was executed
```

The MLflow run ID is the primary key for reproducibility. Given a run ID, you can load the exact model, the exact training parameters, and the exact feature spec that was used:

```python
freq_model = mlflow.catboost.load_model(f"runs:/{freq_run_id}/freq_model")
```

The Delta table version is the primary key for data reproducibility. Given a version number from the audit record, you can read the exact training data:

```python
spark.read.format("delta") \
    .option("versionAsOf", logged_version) \
    .table("main.motor_q2_2026.features") \
    .limit(5) \
    .show()
```

Together, these two identifiers satisfy the FCA's Consumer Duty reproducibility requirement for any pipeline run in the past three years, provided the VACUUM retention policy is set appropriately (Stage 3 handles this).
