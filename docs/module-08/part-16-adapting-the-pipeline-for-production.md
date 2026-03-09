## Part 16: Adapting the pipeline for production

The notebook runs end-to-end on synthetic data. Adapting it to a real motor book requires changes in four places only:

**Stage 2 (Data ingestion).** Replace the synthetic data generation with a read from your policy administration system. The only requirement is that the input DataFrame has the columns that TRANSFORMS expects: `age`, `vehicle_age`, `vehicle_group`, `region`, `credit_score`, `exposure`, `claim_count`, `claim_amount`, `accident_year`.

```python
# Replace Stage 2 synthetic generation with:
raw_pl = pl.from_pandas(
    spark.table("your_source_system.motor.policies_claims")
    .filter(col("accident_year").between(2021, 2024))
    .toPandas()
)
```

**Stage 3 (Feature engineering).** Update TRANSFORMS to match your feature set. Every transform must be a pure function: it takes a Polars DataFrame and returns a Polars DataFrame. Do not add logic to `apply_transforms()` itself -- add a new function to TRANSFORMS.

**Stage 5 (Optuna).** Set `N_OPTUNA_TRIALS = 40` for production runs. The 20-trial default in this notebook runs in 5-10 minutes. For a production review cycle where compute cost is not a constraint, 40-60 trials gives meaningfully better parameter selection.

**Stage 4 (IBNR buffer).** Adjust the buffer months to match your line of business. For UK private motor property damage: 6 months. For motor bodily injury: 12-18 months. For employers' liability or solicitors' PI: 24 months minimum.

Everything else -- the MLflow logging structure, the conformal calibration, the audit record -- runs identically on your data.