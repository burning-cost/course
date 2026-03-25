## Part 10: Writing results to Delta tables

### Why you must persist monitoring results

A monitoring run that produces results and then discards them is useless for governance. The PRA's expectation in SS1/23 is that monitoring outcomes are recorded and available for review. In practice, this means you need a queryable history of every monitoring run: what metrics were computed, what the values were, and what the traffic lights showed.

Delta tables are the right storage layer for this. Delta gives you:

- **Version history**: every write is a new version, and you can time-travel to any previous state
- **Schema enforcement**: the table schema is validated on every write, preventing malformed results from corrupting the log
- **SQL access**: any analyst can query the monitoring log in a notebook or from the Databricks SQL editor without running the full monitoring pipeline

We write three tables: a summary monitoring log (one row per run), a CSI detail table (one row per feature per run), and an A/E detail table (one row per segment per run).

### Setting up the tables

Create the monitoring schema if it does not already exist:

```python
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")
```

### Writing the monitoring summary

The summary table has one row per monitoring run. We draw values from `report.results_`:

```python
from pyspark.sql import Row

results = report.results_

summary_row = {
    "run_date":        RUN_DATE,
    "model_name":      MODEL_NAME,
    "model_version":   MODEL_VERSION,
    "reference_date":  REFERENCE_DATE,
    "current_date":    CURRENT_DATE,
    "recommendation":  report.recommendation,
    "ae_ratio":        float(results["ae_ratio"]["value"]),
    "ae_ci_lower":     float(results["ae_ratio"]["lower_ci"]),
    "ae_ci_upper":     float(results["ae_ratio"]["upper_ci"]),
    "ae_band":         results["ae_ratio"]["band"],
    "gini_ref":        float(results["gini"]["reference"]),
    "gini_cur":        float(results["gini"]["current"]),
    "gini_p_value":    float(results["gini"]["p_value"]),
    "gini_band":       results["gini"]["band"],
    "score_psi":       float(psi_score),          # computed in Part 4
    "reference_n":     int(len(df_reference)),
    "current_n":       int(len(df_current)),
    "actual_claims":   int(actual_cur.sum()),
    "expected_claims": float((pred_cur * exposure_cur).sum()),
}

summary_df = spark.createDataFrame([Row(**summary_row)])

(summary_df
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["monitoring_log"]))

print(f"Monitoring summary written to {TABLES['monitoring_log']}")
```

`mode("append")` adds the new row without overwriting existing records. `mergeSchema: true` allows the table to gain new columns if you add metrics in future monitoring runs without needing to manually alter the schema.

### Writing the CSI detail table

`csi()` returns a Polars DataFrame with columns `feature`, `csi`, and `band`. Build rows from it directly:

```python
csi_rows = []
for row in csi_df.iter_rows(named=True):
    csi_rows.append({
        "run_date":     RUN_DATE,
        "model_name":   MODEL_NAME,
        "current_date": CURRENT_DATE,
        "feature":      row["feature"],
        "csi":          float(row["csi"]),
        "band":         row["band"],
        "n_bins":       N_BINS,
    })

csi_spark = spark.createDataFrame([Row(**r) for r in csi_rows])

(csi_spark
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["csi_results"]))

print(f"CSI detail written to {TABLES['csi_results']}  ({len(csi_rows)} rows)")
```

### Writing the A/E segment table

```python
ae_rows = [{
    "run_date":      RUN_DATE,
    "model_name":    MODEL_NAME,
    "current_date":  CURRENT_DATE,
    "segment":       "portfolio",
    "segment_value": "all",
    "ae_ratio":      float(results["ae_ratio"]["value"]),
    "ci_lower":      float(results["ae_ratio"]["lower_ci"]),
    "ci_upper":      float(results["ae_ratio"]["upper_ci"]),
    "actual":        float(actual_cur.sum()),
    "expected":      float((pred_cur * exposure_cur).sum()),
    "band":          results["ae_ratio"]["band"],
}]

ae_spark = spark.createDataFrame([Row(**r) for r in ae_rows])

(ae_spark
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["ae_results"]))

print(f"A/E results written to {TABLES['ae_results']}")
```

### Querying the monitoring trend

Once you have several months of data, query the trend directly from the notebook:

```python
trend_query = f"""
SELECT
    current_date,
    ae_ratio,
    ae_ci_lower,
    ae_ci_upper,
    recommendation,
    gini_cur,
    gini_p_value
FROM {TABLES["monitoring_log"]}
WHERE model_name = '{MODEL_NAME}'
ORDER BY current_date DESC
LIMIT 12
"""

trend_df = spark.sql(trend_query)
trend_df.show()
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

By default, Delta retains the transaction log for 30 days and keeps physically deleted data files for 7 days before VACUUM removes them. For a monitoring log, you want at least 7 years of history (broadly consistent with FCA record-keeping requirements). Set both retention properties:

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

Without setting `logRetentionDuration`, time travel fails after 30 days even if the data files are still present.
