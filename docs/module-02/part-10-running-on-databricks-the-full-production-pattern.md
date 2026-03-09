## Part 10: Running on Databricks - the full production pattern

### Loading data from Delta tables

Rather than CSV exports, the production pattern loads data from Delta tables registered in Unity Catalog. This gives you time travel (query the data as it was at model-fit time) and full lineage tracking.

```python
# On Databricks, spark is already available - no import needed
# spark = SparkSession.builder.getOrCreate()  # Only needed outside Databricks

# Load current policy data
df_spark = spark.table("main.pricing.motor_policies")

# Convert to Polars for manipulation
df = pl.from_pandas(df_spark.toPandas())
```

To use the data as it was on a specific date:

```python
df_spark = spark.sql(
    "SELECT * FROM main.pricing.motor_policies TIMESTAMP AS OF '2024-03-15T00:00:00'"
)
```

Record the table version number when fitting - this is how you prove to a regulator what data you used:

```python
table_version = spark.sql(
    "DESCRIBE HISTORY main.pricing.motor_policies LIMIT 1"
).first()["version"]

print(f"Training data version: {table_version}")
```

### Logging to MLflow

MLflow is Databricks' experiment tracking system. It stores parameters, metrics, and artefacts from each model run, giving you a queryable history of every model you have fitted.

```python
import mlflow
from datetime import date

mlflow.set_experiment("/pricing/motor-glm")

with mlflow.start_run(run_name="freq_glm_v2") as run:
    # Log parameters - everything that defines the model
    mlflow.log_params({
        "model_type": "Poisson_GLM",
        "formula": freq_formula,
        "n_policies": len(df),
        "training_data_version": table_version,
        "training_date": str(date.today()),
        "base_levels": str({"area": "A", "ncd_years": "0", "conviction_flag": "0"}),
    })

    # Log metrics - numbers that describe model performance
    mlflow.log_metrics({
        "deviance": glm_freq.deviance,
        "null_deviance": glm_freq.null_deviance,
        "pseudo_r2": 1 - (glm_freq.deviance / glm_freq.null_deviance),
        "aic": glm_freq.aic,
        "n_params": len(glm_freq.params),
        "converged": int(glm_freq.converged),
        "n_iterations": glm_freq.nit,
    })

    # Log the relativities as a CSV artefact
    rels_path = "/tmp/freq_relativities.csv"
    freq_rels.write_csv(rels_path)
    mlflow.log_artifact(rels_path, artifact_path="factor_tables")

    # Log the model summary as a text artefact
    summary_path = "/tmp/glm_freq_summary.txt"
    with open(summary_path, "w") as f:
        f.write(str(glm_freq.summary()))
    mlflow.log_artifact(summary_path, artifact_path="diagnostics")

    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")
```

### Writing factor tables to Unity Catalog

```python
from datetime import date

rels_with_meta = freq_rels.with_columns([
    pl.lit(str(date.today())).alias("model_run_date"),
    pl.lit("freq_glm_v2").alias("model_name"),
    pl.lit(run_id).alias("mlflow_run_id"),
    pl.lit(table_version).alias("training_data_version"),
    pl.lit(len(df)).alias("n_policies_trained"),
])

spark.createDataFrame(rels_with_meta.to_pandas()).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.glm_relativities")

print(f"Written {len(rels_with_meta)} rows to main.pricing.glm_relativities")
```

Using `mode("append")` means every model run adds to the history. You can query how any factor's relativity has changed across model cycles:

```python
area_f_history = spark.sql("""
    SELECT model_run_date, model_name, relativity
    FROM main.pricing.glm_relativities
    WHERE feature = 'area' AND level = 'F'
    ORDER BY model_run_date
""")
display(area_f_history)
```

This is the audit trail that both PS 21/5 and Consumer Duty require.