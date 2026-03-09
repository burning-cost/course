## Part 10: Writing results to Delta tables

### Why you must persist monitoring results

A monitoring run that produces results and then discards them is useless for governance. The PRA's expectation in SS1/23 is that monitoring outcomes are recorded and available for review. In practice, this means you need a queryable history of every monitoring run: what metrics were computed, what the values were, and what the traffic lights showed.

Delta tables are the right storage layer for this. Delta gives you:

- **Version history**: every write is a new version, and you can time-travel to any previous state
- **Schema enforcement**: the table schema is validated on every write, preventing malformed results from corrupting the log
- **SQL access**: any analyst can query the monitoring log in a notebook or from the Databricks SQL editor without running the full monitoring pipeline

We write three tables: a summary monitoring log (one row per run), a PSI/CSI detail table (one row per feature per run), and an A/E detail table (one row per segment per run).

### Setting up the tables

Create the monitoring schema if it does not already exist:

```python
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")
```

### Writing the monitoring summary

The summary table has one row per monitoring run. This is the high-level log:

```python
from pyspark.sql import Row
from pyspark.sql import functions as F
from datetime import datetime

# Build the summary row from the report summary dict
summary_row = {
    "run_date":             RUN_DATE,
    "model_name":           MODEL_NAME,
    "model_version":        MODEL_VERSION,
    "reference_date":       REFERENCE_DATE,
    "current_date":         CURRENT_DATE,
    "overall_traffic_light": summary["overall_traffic_light"],
    "psi_score":            float(summary["metrics"]["psi_score"]["value"]),
    "psi_traffic_light":    summary["metrics"]["psi_score"]["traffic_light"],
    "ae_ratio":             float(summary["metrics"]["ae_ratio"]["value"]),
    "ae_ci_lower":          float(summary["metrics"]["ae_ratio"]["ci_lower"]),
    "ae_ci_upper":          float(summary["metrics"]["ae_ratio"]["ci_upper"]),
    "ae_traffic_light":     summary["metrics"]["ae_ratio"]["traffic_light"],
    "gini_ref":             float(summary["metrics"]["gini"]["gini_ref"]),
    "gini_cur":             float(summary["metrics"]["gini"]["gini_cur"]),
    "gini_p_value":         float(summary["metrics"]["gini"]["p_value"]),
    "gini_traffic_light":   summary["metrics"]["gini"]["traffic_light"],
    "reference_n":          int(df_reference.shape[0]),
    "current_n":            int(df_current.shape[0]),
    "actual_claims":        int(actual_cur.sum()),
    "expected_claims":      float(expected_cur.sum()),
}

# Create a Spark DataFrame with one row
summary_df = spark.createDataFrame([Row(**summary_row)])

# Write to Delta, appending to the existing log
(
    summary_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TABLES["monitoring_log"])
)

print(f"Monitoring summary written to {TABLES['monitoring_log']}")
```

`mode("append")` adds the new row without overwriting existing records. `mergeSchema: true` allows the table to gain new columns if you add metrics in future monitoring runs without needing to manually alter the schema.

### Writing the CSI detail table

One row per feature per run. This lets you query the CSI history for a specific feature:

```python
# Build CSI rows
csi_rows = []
for feature, result in csi_scores.items():
    csi_rows.append({
        "run_date":       RUN_DATE,
        "model_name":     MODEL_NAME,
        "current_date":   CURRENT_DATE,
        "feature":        feature,
        "csi":            float(result.csi),
        "traffic_light":  result.traffic_light,
        "n_bins":         N_BINS,
    })

csi_df = spark.createDataFrame([Row(**r) for r in csi_rows])

(
    csi_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TABLES["csi_results"])
)

print(f"CSI detail written to {TABLES['csi_results']}  ({len(csi_rows)} rows)")
```

### Writing the A/E segment table

```python
# Portfolio-level result
ae_rows = [{
    "run_date":      RUN_DATE,
    "model_name":    MODEL_NAME,
    "current_date":  CURRENT_DATE,
    "segment":       "portfolio",
    "segment_value": "all",
    "ae_ratio":      float(ae_portfolio.ratio),
    "ci_lower":      float(ae_portfolio.ci_lower),
    "ci_upper":      float(ae_portfolio.ci_upper),
    "actual":        float(actual_cur.sum()),
    "expected":      float(expected_cur.sum()),
    "traffic_light": ae_portfolio.traffic_light,
}]

ae_df = spark.createDataFrame([Row(**r) for r in ae_rows])

(
    ae_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TABLES["ae_results"])
)

print(f"A/E results written to {TABLES['ae_results']}")
```

### Querying the monitoring trend

Once you have several months of data, query the trend directly from the notebook:

```python
# Show the last 12 months of portfolio A/E
trend_query = f"""
SELECT
    current_date,
    ae_ratio,
    ae_ci_lower,
    ae_ci_upper,
    overall_traffic_light,
    psi_score,
    gini_cur
FROM {TABLES["monitoring_log"]}
WHERE model_name = '{MODEL_NAME}'
ORDER BY current_date DESC
LIMIT 12
"""

trend_df = spark.sql(trend_query)
trend_df.show()
```

Or in Polars if you prefer:

```python
trend_pl = pl.from_pandas(trend_df.toPandas())
print(trend_pl)
```

### Using Delta time travel to audit historical reports

Delta stores every version of the table. To see what the monitoring log looked like after the run from a specific date, use the `VERSION AS OF` syntax:

```sql
SELECT *
FROM main.motor_monitoring.monitoring_log
VERSION AS OF 3
WHERE model_name = 'motor_frequency_catboost'
```

Or to query as of a specific timestamp:

```sql
SELECT *
FROM main.motor_monitoring.monitoring_log
TIMESTAMP AS OF '2024-03-01'
```

This is your audit trail. If a regulator asks "what did your monitoring show in March 2024?", you can reproduce the exact state of the table as it existed on that date. This is the governance argument for using Delta rather than a flat CSV log.

### VACUUM and retention

By default, Delta retains the transaction log for 30 days and keeps physically deleted data files for 7 days before VACUUM removes them. For a monitoring log, you want at least 7 years of history (broadly consistent with FCA record-keeping requirements). You need to set **both** retention properties: `deletedFileRetentionDuration` controls how long data files are retained after deletion, and `logRetentionDuration` controls how long the transaction log is kept — and it is the log that enables time travel. Without setting `logRetentionDuration`, time travel fails after 30 days even if the data files are still present.

```python
spark.sql(f"""
    ALTER TABLE {TABLES["monitoring_log"]}
    SET TBLPROPERTIES (
        'delta.deletedFileRetentionDuration' = 'interval 7 years',
        'delta.logRetentionDuration'         = 'interval 7 years'
    )
""")

print("Retention policy set to 7 years.")
```

This ensures you can time-travel to any point in the past 7 years, which covers any reasonable regulatory review window.
