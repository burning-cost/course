## Part 14: Stage 10 — The pipeline audit record

The audit record is the last thing the pipeline writes. It is a single row that ties together every output from every upstream stage: raw data version, features version, MLflow run IDs for both models, all validation metrics, calibration verdict, conformal coverage, rate optimisation outcome, and the configuration that produced it.

It is not documentation written by a human. It is a structured record written by the pipeline itself, in the same notebook run that produced all the outputs it describes. That is what makes it useful for a section 166 request: there is no interpretation, no reconstruction from memory.

Add a markdown cell:

```python
%md
## Stage 10: Pipeline audit record
```

```python
import json

audit_record = {
    # ---- Identifiers ----
    "run_date":              RUN_DATE,
    "pipeline_version":      "module_08_v1",

    # ---- Data provenance ----
    "raw_table":             TABLES["raw"],
    "raw_table_version":     int(raw_version),
    "features_table":        TABLES["features"],
    "features_table_version": int(feat_version),

    # ---- Model provenance ----
    "freq_model_run_id":     freq_run_id,
    "sev_model_run_id":      sev_run_id,

    # ---- Training configuration ----
    "feature_cols":          json.dumps(FEATURE_COLS),
    "cat_features":          json.dumps(CAT_FEATURES),
    "train_years":           str(sorted(df_train["accident_year"].unique().tolist())),
    "test_year":             int(max_year),
    "n_training_rows":       int(len(df_train)),
    "n_test_rows":           int(len(df_test)),

    # ---- CV and tuning ----
    "n_cv_folds":            len(cv_deviances),
    "mean_cv_deviance":      round(mean_cv_deviance, 5),
    "fold_deviances":        json.dumps([round(d, 5) for d in cv_deviances]),
    "freq_optuna_trials":    N_OPTUNA_TRIALS,
    "freq_best_deviance":    round(freq_study.best_value, 5),
    "sev_optuna_trials":     N_OPTUNA_TRIALS,
    "sev_best_rmse":         round(sev_study.best_value, 2),

    # ---- Model performance ----
    "test_poisson_deviance": round(test_dev, 5),
    "generalisation_gap":    round(test_dev - mean_cv_deviance, 5),
    "sev_rmse":              round(sev_rmse, 2),

    # ---- Calibration ----
    "cal_balance_ratio":     round(cal_balance_ratio, 4),
    "cal_balance_ok":        bool(cal_balance_ok),
    "cal_auto_p":            round(cal_auto_p, 4),
    "cal_auto_ok":           bool(cal_auto_ok),
    "cal_murphy_verdict":    cal_murphy_verdict,
    "cal_dsc_pct":           cal_dsc_pct,
    "cal_mcb_pct":           cal_mcb_pct,

    # ---- Conformal ----
    "conformal_alpha":       CONFORMAL_ALPHA,
    "conformal_cal_year":    int(cal_year),
    "conformal_n_cal":       int(len(df_cal_sev)),
    "conformal_min_cov":     round(min_cov, 3),

    # ---- Rate optimisation ----
    "lr_target":             LR_TARGET,
    "volume_floor":          VOLUME_FLOOR,
    "optimiser_converged":   bool(result.converged),
    "expected_lr":           round(float(result.expected_loss_ratio), 4),
    "expected_volume":       round(float(result.expected_volume_ratio), 4),
    "enbp_violations":       int(result.enbp_violations),

    # ---- Infrastructure ----
    "catalog":               CATALOG,
    "schema":                SCHEMA,
    "pipeline_notes":        "Module 8 capstone — synthetic UK motor data",
}

# mode("append") is required. Every run adds one row.
# mode("overwrite") would destroy the history.
(
    spark.createDataFrame([audit_record])
    .write.format("delta")
    .mode("append")
    .saveAsTable(TABLES["pipeline_audit"])
)

print(f"Audit record written to {TABLES['pipeline_audit']}")
print(f"\nKey fields:")
print(f"  Raw data version:    {audit_record['raw_table_version']}")
print(f"  Features version:    {audit_record['features_table_version']}")
print(f"  Freq model run:      {audit_record['freq_model_run_id']}")
print(f"  Sev model run:       {audit_record['sev_model_run_id']}")
print(f"  Test deviance:       {audit_record['test_poisson_deviance']}")
print(f"  Murphy verdict:      {audit_record['cal_murphy_verdict']}")
print(f"  Conformal min cov:   {audit_record['conformal_min_cov']}")
print(f"  Optimiser converged: {audit_record['optimiser_converged']}")
print(f"  Expected LR:         {audit_record['expected_lr']}")
```

### Reproducing any historical run

Given the `run_date` for any historical pipeline run, you can reproduce every output exactly:

```python
# Read the audit record for a specific run date
audit_history = (
    spark.table(TABLES["pipeline_audit"])
    .filter(col("run_date") == "2026-03-25")
    .toPandas()
)
row = audit_history.iloc[0]

# Reproduce the training data
historical_raw = (
    spark.read.format("delta")
    .option("versionAsOf", row["raw_table_version"])
    .table(row["raw_table"])
    .toPandas()
)

# Load the trained frequency model
historical_freq = mlflow.catboost.load_model(
    f"runs:/{row['freq_model_run_id']}/freq_model"
)

# Load the SHAP relativities
client = mlflow.MlflowClient()
client.download_artifacts(row["freq_model_run_id"], "shap_relativities.json",
                          dst_path="/tmp/")
```

This is the FCA Consumer Duty audit trail in practice. Section 166 asks you to demonstrate: what model was used, what data it was trained on, what validation metrics it achieved, and what rate action it informed. The audit table, combined with Delta time travel and MLflow model registry, answers all four without any reconstruction from memory.

### Final pipeline summary

```python
print("=" * 65)
print("MODULE 8: END-TO-END PIPELINE — COMPLETE")
print("=" * 65)
print()
print(f"Stages completed:    10")
print(f"Training rows:       {len(df_train):,}")
print(f"Test rows:           {len(df_test):,}")
print(f"CV folds:            {len(cv_deviances)}")
print(f"Mean CV deviance:    {mean_cv_deviance:.5f}")
print(f"Test deviance:       {test_dev:.5f}")
print(f"Severity RMSE:       £{sev_rmse:,.0f}")
print(f"Murphy verdict:      {cal_murphy_verdict}")
print(f"Conformal min cov:   {min_cov:.3f}")
print(f"Optimiser converged: {result.converged}")
print(f"Expected LR:         {result.expected_loss_ratio:.4f}")
print(f"Expected volume:     {result.expected_volume_ratio:.4f}")
print()
print("Delta tables written:")
for k, v in TABLES.items():
    print(f"  {k:<25} {v}")
print()
print(f"MLflow experiment:   motor-pipeline-m08")
print(f"  Frequency model:   {freq_run_id}")
print(f"  Severity model:    {sev_run_id}")
print()
print("Next: present efficient frontier to pricing committee.")
print("      Schedule monthly conformal recalibration trigger.")
print("      Connect to Module 9 demand model for retention-aware pricing.")
```
