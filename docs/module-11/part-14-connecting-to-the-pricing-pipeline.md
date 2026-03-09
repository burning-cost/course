## Part 14: Connecting to the pricing pipeline

### The monitoring loop

Monitoring is not a standalone activity. It sits inside a loop:

1. Pipeline (Module 8) produces a trained model and frequency predictions
2. Model is deployed into the rating engine
3. Live policies are written; claims emerge
4. Monitoring (this module) detects drift and triggers recalibration or retraining
5. Retraining produces a new model - back to step 1

This loop only works if monitoring has access to the same data structures the pipeline produces, and if the pipeline can consume recalibration factors that monitoring produces. This part shows how to connect the two.

### Reading predictions from the pipeline's Delta tables

The Module 8 pipeline writes frequency predictions to a Delta table. The monitoring notebook reads those predictions rather than re-running the model from scratch. This has two advantages:

1. The monitoring notebook measures what the deployed model actually predicted, not what a freshly-loaded model would predict today (which might differ if the model has been updated)
2. It is faster - no need to reload and run the model on every monitoring execution

Assuming the pipeline writes predictions to `main.motor_q2_2026.freq_predictions`:

```python
# Load predictions from the pipeline output table
pipeline_schema = "motor_q2_2026"   # From Module 8 configuration
predictions_table = f"{CATALOG}.{pipeline_schema}.freq_predictions"

pred_df = spark.sql(f"""
    SELECT
        policy_id,
        policy_start_date,
        predicted_frequency,
        exposure,
        claim_count
    FROM {predictions_table}
    WHERE prediction_date = (
        SELECT MAX(prediction_date) FROM {predictions_table}
    )
""")

pred_pl = pl.from_pandas(pred_df.toPandas())

print(f"Loaded {pred_pl.shape[0]:,} predictions from {predictions_table}")
print(pred_pl.head())
```

This loads the most recent prediction run. If you need a specific prediction date (for backfilling monitoring for a past month), add a `WHERE prediction_date = '2024-03-15'` clause.

Split into reference and current as before:

```python
pred_reference = pred_pl.filter(
    pl.col("policy_start_date") <= pl.lit(REFERENCE_DATE).str.to_date()
)
pred_current = pred_pl.filter(
    (pl.col("policy_start_date") > pl.lit(REFERENCE_DATE).str.to_date()) &
    (pl.col("policy_start_date") <= pl.lit(CURRENT_DATE).str.to_date())
)

pred_ref = pred_reference["predicted_frequency"].to_numpy()
pred_cur = pred_current["predicted_frequency"].to_numpy()
actual_ref = pred_reference["claim_count"].to_numpy().astype(float)
actual_cur = pred_current["claim_count"].to_numpy().astype(float)
exposure_ref = pred_reference["exposure"].to_numpy()
exposure_cur = pred_current["exposure"].to_numpy()
```

### Reading the recalibration factor in the pipeline

The pipeline needs to apply any active recalibration factor when generating predictions for new quotes. Add this step to the Module 8 pipeline's Stage 6 (final model predictions):

```python
# In the pricing pipeline notebook (Module 8)

def get_active_recalibration_factor(
    model_name: str,
    catalog: str,
    schema: str,
    as_of_date: str,
) -> float:
    """
    Read the most recent recalibration factor effective on or before as_of_date.
    Returns 1.0 if no recalibration has been applied.
    """
    recal_table = f"{catalog}.{schema}.recalibration_history"

    try:
        result = spark.sql(f"""
            SELECT recalibration_factor
            FROM {recal_table}
            WHERE model_name = '{model_name}'
              AND effective_from <= '{as_of_date}'
            ORDER BY effective_from DESC
            LIMIT 1
        """).collect()

        if result:
            factor = result[0]["recalibration_factor"]
            print(f"Recalibration factor applied: {factor:.4f} "
                  f"(model: {model_name})")
            return float(factor)
        else:
            print("No recalibration factor found. Using 1.0.")
            return 1.0

    except Exception as e:
        # Table does not exist yet (first pipeline run)
        print(f"Recalibration table not found: {e}. Using 1.0.")
        return 1.0


# Apply in the pipeline after model prediction
recal_factor = get_active_recalibration_factor(
    model_name=MODEL_NAME,
    catalog=CATALOG,
    schema=SCHEMA,
    as_of_date=RUN_DATE,
)

# Scale predictions
freq_predictions_recalibrated = freq_predictions * recal_factor
```

This means the pipeline always applies the latest recalibration factor. When monitoring updates the factor (Part 13), the next pipeline run picks it up automatically. No pipeline code change required.

### Tracking which model version produced which predictions

The monitoring notebook should always record the model version it tested. If you have run multiple model versions (v1, v2, v3) and want to compare their monitoring histories, you need the version column in the monitoring log.

The pipeline should stamp predictions with the model version:

```python
# In the pipeline, when writing predictions:
# predictions_df = predictions_df.with_columns([
#     pl.lit(MODEL_VERSION).alias("model_version"),
#     pl.lit(RUN_DATE).alias("prediction_date"),
# ])
```

In the monitoring notebook, filter predictions by the model version you are assessing:

```python
MODEL_VERSION_TO_MONITOR = "1"   # Set in configuration cell

pred_df_versioned = spark.sql(f"""
    SELECT *
    FROM {predictions_table}
    WHERE model_version = '{MODEL_VERSION_TO_MONITOR}'
      AND prediction_date = (
          SELECT MAX(prediction_date)
          FROM {predictions_table}
          WHERE model_version = '{MODEL_VERSION_TO_MONITOR}'
      )
""")
```

### What the connection looks like end to end

The data flow is:

```text
insurance_datasets (load_motor)
    -> Pipeline notebook (Module 8)
        -> freq_predictions Delta table  (model_version, prediction_date, predicted_frequency)
            -> Monitoring notebook (this module)
                -> monitoring_log Delta table  (PSI, A/E, Gini, traffic lights)
                -> recalibration_history Delta table  (recal_factor, effective_from)
                    -> Pipeline notebook (next run, reads recal_factor)
```

The Delta tables are the integration layer. Neither notebook calls the other directly. This loose coupling means:

- You can run monitoring independently of the pipeline (and vice versa)
- You can backfill monitoring for historical periods without re-running the pipeline
- A failure in one notebook does not cascade to the other

This is the architecture we recommend for production. It is also the architecture that satisfies regulatory audit requirements: every step writes to a versioned table, and the provenance chain from raw data to monitoring conclusion is fully traceable.
