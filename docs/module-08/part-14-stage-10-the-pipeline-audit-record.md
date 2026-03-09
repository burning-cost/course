## Part 14: Stage 10 -- The pipeline audit record

The audit record is the last thing the pipeline writes. It is a single row that ties together every output from every upstream stage: the raw data version, the features version, the MLflow run IDs for both models, all validation metrics, and the rate optimisation outcome.

Add a markdown cell:

```python
%md
## Stage 10: Pipeline audit record
```

```python
audit_record = {
    # ---- Identifiers ----
    "run_date":                  RUN_DATE,
    "pipeline_version":          "module_08_v1",

    # ---- Data provenance ----
    "raw_table":                 TABLES["raw"],
    "raw_table_version":         int(raw_version),
    "features_table":            TABLES["features"],
    "features_table_version":    int(feat_version),

    # ---- Model provenance ----
    "freq_model_run_id":         freq_run_id,
    "sev_model_run_id":          sev_run_id,

    # ---- Training data summary ----
    "n_training_rows":           int(len(df_train)),
    "n_test_rows":               int(len(df_test)),
    "test_year":                 int(max_year),
    "training_years":            str(sorted(df_train["accident_year"].unique().tolist())),

    # ---- Model performance ----
    "test_poisson_deviance":     round(test_dev, 5),
    "mean_cv_deviance":          round(mean_cv_deviance, 5),
    "sev_rmse":                  round(sev_rmse, 2),
    "n_cv_folds":                len(cv_deviances),

    # ---- Optuna ----
    "freq_optuna_trials":        N_OPTUNA_TRIALS,
    "freq_best_deviance":        round(freq_study.best_value, 5),
    "sev_optuna_trials":         N_OPTUNA_TRIALS,
    "sev_best_rmse":             round(sev_study.best_value, 2),

    # ---- Conformal ----
    "conformal_alpha":           CONFORMAL_ALPHA,
    "conformal_cal_year":        int(cal_year),
    "conformal_n_cal_claims":    int(len(df_cal_sev)),
    "conformal_min_decile_cov":  round(min_cov, 3),

    # ---- Rate optimisation ----
    "lr_target":                 LR_TARGET,
    "volume_floor":              VOLUME_FLOOR,
    "optimiser_converged":       bool(result.converged),
    "expected_lr":               round(float(result.expected_loss_ratio), 4),
    "expected_volume":           round(float(result.expected_volume_ratio), 4),

    # ---- Configuration ----
    "catalog":                   CATALOG,
    "schema":                    SCHEMA,
    "feature_cols":              json.dumps(FEATURE_COLS),
    "cat_features":              json.dumps(CAT_FEATURES),

    # ---- Notes ----
    "pipeline_notes":            "Module 8 end-to-end pipeline, synthetic motor data",
}

audit_pl = pl.DataFrame([audit_record])

# mode("append") is deliberate -- every run adds a row.
# mode("overwrite") would destroy the history.
spark.createDataFrame(audit_pl.to_pandas()) \
    .write.format("delta") \
    .mode("append") \
    .saveAsTable(TABLES["pipeline_audit"])

print("Pipeline audit record written.")
print(f"Table: {TABLES['pipeline_audit']}")
print("\nAudit record summary:")
for k, v in audit_record.items():
    print(f"  {k:<40} {v}")
```

### What the audit record enables

Given the `run_date` for any historical pipeline run, you can retrieve the full audit record and reproduce every output exactly:

```python
# Six months later: reproduce the training data
logged_raw_version = 0   # from audit_record["raw_table_version"]

historical_raw = spark.read.format("delta") \
    .option("versionAsOf", logged_raw_version) \
    .table(TABLES["raw"]) \
    .toPandas()

# Six months later: load the trained frequency model
logged_freq_run_id = "abc123..."  # from audit_record["freq_model_run_id"]
historical_freq_model = mlflow.catboost.load_model(f"runs:/{logged_freq_run_id}/freq_model")
```

This is the FCA Consumer Duty audit trail in practice. Consumer Duty requires that you can demonstrate, for any pricing decision in the last three years: what model was used, what data it was trained on, what validation metrics it achieved, what rate action it informed. The audit record table, combined with Delta time travel and MLflow model registry, satisfies all four requirements automatically.