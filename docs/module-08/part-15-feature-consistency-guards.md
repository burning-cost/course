## Part 15: Feature consistency guards

The `apply_transforms()` function prevents training-scoring divergence for feature engineering logic. The `FeatureSpec` class (introduced in Stage 3) prevents a different failure: scoring data that has the right column names but the wrong dtypes or value ranges.

This part explains when and how to use the FeatureSpec in a production scoring context.

### At training time

Stage 3 already records the spec and logs it to MLflow. This was:

```python
feature_spec = FeatureSpec()
feature_spec.record(features_pl.select(FEATURE_COLS), cat_features=CAT_FEATURES)
feature_spec.to_json("/tmp/feature_spec.json")
# logged to MLflow in Stage 6: mlflow.log_artifact("/tmp/feature_spec.json", ...)
```

The spec records, for each feature:
- `categorical`: the set of valid levels seen at training time
- `numeric`: the min and max values seen at training time

### At scoring time

When a new batch of policies arrives for scoring, validate the feature matrix before passing it to `freq_model.predict()`:

```python
# Load the spec from the MLflow run that produced the deployed model
client = mlflow.MlflowClient()
client.download_artifacts(
    run_id=freq_run_id,
    path="feature_spec/feature_spec.json",
    dst_path="/tmp/scoring_spec/",
)
spec = FeatureSpec.from_json("/tmp/scoring_spec/feature_spec.json")

# Apply transforms to the incoming scoring data
new_batch_pl = pl.read_parquet("/dbfs/incoming/renewals_2026_q3.parquet")
new_features  = apply_transforms(new_batch_pl)

errors = spec.validate(new_features.select(FEATURE_COLS))
if errors:
    for e in errors:
        print(f"FEATURE SPEC VIOLATION: {e}")
    raise ValueError(
        f"Feature spec validation failed ({len(errors)} errors). "
        f"Check apply_transforms() in the scoring code against the "
        f"training pipeline. See violations above."
    )
else:
    pred = freq_model.predict(Pool(
        new_features.select(FEATURE_COLS).to_pandas(),
        baseline=np.log(np.clip(new_features["exposure"].to_numpy(), 1e-6, None)),
        cat_features=CAT_FEATURES,
    ))
    print(f"Scored {len(new_batch_pl):,} policies successfully.")
```

The spec catches the NCB encoding incident before it can cause three months of mispriced renewals. If `ncb_deficit` arrives at scoring time as a string where the training spec recorded it as `numeric`, the validator raises an error immediately with a message that names the column and the type mismatch.

### Delta table retention

All seven tables written by this pipeline carry a 365-day retention policy. This is set individually at each write. If you want to set it as a default for the entire schema, use:

```python
spark.sql(f"""
    ALTER SCHEMA {CATALOG}.{SCHEMA}
    SET DBPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
```

This sets the retention for any future tables created in the schema but does not retroactively update existing tables. For an existing table, the `ALTER TABLE` approach used in each stage is required.

**Checking retention on all tables:**

```python
for table_name, table_path in TABLES.items():
    try:
        props = (
            spark.sql(f"SHOW TBLPROPERTIES {table_path}")
            .filter("key = 'delta.deletedFileRetentionDuration'")
            .collect()
        )
        retention = props[0]["value"] if props else "default (30 days)"
        print(f"{table_name:<25} retention: {retention}")
    except Exception as e:
        print(f"{table_name:<25} could not check: {e}")
```

Any table showing "default (30 days)" needs the `ALTER TABLE SET TBLPROPERTIES` command applied before the next pipeline run. The audit record will reference versions of that table that may be vacuumed within 30 days, making the audit trail incomplete.
