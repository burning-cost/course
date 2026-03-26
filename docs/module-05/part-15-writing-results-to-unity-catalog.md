## Part 15: Writing results to Unity Catalog

Intervals and coverage diagnostics go to Delta tables. This gives you version history, time travel, and a permanent audit trail linked to the MLflow run ID.

In a new cell:

```python
%md
## Part 15: Writing to Delta tables
```

```python
import pandas as pd

# Recalibrate on full calibration set before writing production results
cp.calibrate(X_cal, y_cal)

# Regenerate intervals with the full calibration
intervals_90 = cp.predict_interval(X_test, alpha=0.10)
intervals_95 = cp.predict_interval(X_test, alpha=0.05)
intervals_80 = cp.predict_interval(X_test, alpha=0.20)  # 80% interval — used for practical floor below

upper_90 = intervals_90["upper"].to_numpy()
upper_95 = intervals_95["upper"].to_numpy()
point    = intervals_90["point"].to_numpy()
lower    = intervals_90["lower"].to_numpy()

rel_width        = (upper_90 - lower) / np.clip(point, 1e-6, None)
width_threshold  = np.quantile(rel_width, 0.90)
flag_for_review  = rel_width > width_threshold
floor_conformal  = upper_95
floor_practical  = np.maximum(1.5 * point, intervals_80["upper"].to_numpy())

# Build the output DataFrame
intervals_to_write = intervals_90.to_pandas().copy()
intervals_to_write["model_run_date"]       = str(date.today())
intervals_to_write["mlflow_run_id"]        = conf_run_id
intervals_to_write["alpha"]                = 0.10
intervals_to_write["nonconformity_score"]  = "pearson_weighted"
intervals_to_write["tweedie_power"]        = 1.5
intervals_to_write["flag_for_review"]      = flag_for_review.tolist()
intervals_to_write["relative_width"]       = rel_width.tolist()
intervals_to_write["floor_conformal_95"]   = floor_conformal.tolist()
intervals_to_write["floor_practical"]      = floor_practical.tolist()

print("Writing intervals to pricing.motor.conformal_intervals...")
(
    spark.createDataFrame(intervals_to_write)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pricing.motor.conformal_intervals")
)
print(f"Written {len(intervals_to_write):,} rows.")
```

**What this does:** creates a Delta table with one row per policy in the test set. Each row has the point estimate, lower and upper bounds, the underwriting referral flag, relative width, and both minimum premium floors. The `mlflow_run_id` links this table to the MLflow experiment entry where the model and coverage metrics are stored.

Now write the coverage log in append mode:

```python
# Coverage diagnostics: append so we can track over time
diag_final    = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
diag_to_write = diag_final.to_pandas().copy()
diag_to_write["model_run_date"] = str(date.today())
diag_to_write["mlflow_run_id"]  = conf_run_id
diag_to_write["test_years"]     = str(test_years.to_list())
diag_to_write["calibration_n"]  = len(X_cal)

print("Appending to pricing.motor.conformal_coverage_log...")
(
    spark.createDataFrame(diag_to_write)
    .write.format("delta")
    .mode("append")
    .saveAsTable("pricing.motor.conformal_coverage_log")
)
print("Done.")
```

The coverage log is append-mode. Every calibration run adds a new set of diagnostic rows with a timestamp. Query the history:

```python
spark.sql("""
    SELECT model_run_date, decile, coverage, calibration_n, mlflow_run_id
    FROM pricing.motor.conformal_coverage_log
    ORDER BY model_run_date DESC, decile ASC
""").show(30)
```

A declining top-decile coverage trend over successive runs signals that the calibration is going stale. Set an alert: if the top decile falls below 85% for a 90% interval, trigger recalibration before the next reserve cycle.
