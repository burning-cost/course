## Part 17: Writing relativities to Unity Catalog

For production use, write the relativities to a versioned Delta table. In a new cell, type this and run it (Shift+Enter):

```python
from datetime import date

# Add metadata columns before writing
rels_with_meta = rels.copy()
rels_with_meta["model_run_date"] = str(date.today())
rels_with_meta["model_name"]     = "freq_catboost_module04_v1"
rels_with_meta["n_policies"]     = len(df)
rels_with_meta["mlflow_run_id"]  = run.info.run_id

# Write to Unity Catalog via Spark
spark.createDataFrame(rels_with_meta).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.pricing.gbm_relativities")

print("Relativities written to main.pricing.gbm_relativities")

# Write validation log (append, so history is preserved)
validation_records = [
    {
        "check":          name,
        "passed":         result.passed,
        "message":        result.message,
        "model_run_date": str(date.today()),
        "model_name":     "freq_catboost_module04_v1",
        "mlflow_run_id":  run.info.run_id,
    }
    for name, result in checks.items()
]

spark.createDataFrame(validation_records).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.gbm_relativity_validation_log")

print("Validation log written to main.pricing.gbm_relativity_validation_log")
```

You will see confirmation that the tables were written. You can query them with:

```python
display(spark.table("main.pricing.gbm_relativities"))
```

The table has the full relativity output including confidence intervals, exposure weights, and the metadata columns you added. This is the permanent record.