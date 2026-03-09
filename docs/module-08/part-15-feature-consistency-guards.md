## Part 15: Feature consistency guards

The audit record documents what happened. The feature consistency guard prevents failures from happening. This section shows how to build one.

```python
class FeatureSpec:
    """
    Records the expected dtype and range of each feature at training time.
    Validates the scoring data against the spec at inference time.

    Usage:
      spec = FeatureSpec()
      spec.record(X_train_polars, cat_features=CAT_FEATURES)
      spec.to_json("/dbfs/models/feature_spec.json")

      # At scoring time:
      spec_loaded = FeatureSpec.from_json("/dbfs/models/feature_spec.json")
      errors = spec_loaded.validate(X_score_polars)
      if errors:
          raise ValueError(f"Feature spec validation failed: {errors}")
    """

    def __init__(self):
        self.spec = {}

    def record(self, df: pl.DataFrame, cat_features: list) -> None:
        """Record the spec from a training feature DataFrame."""
        for col in df.columns:
            series = df[col]
            if col in cat_features or series.dtype == pl.Utf8:
                self.spec[col] = {
                    "dtype":       "categorical",
                    "unique_vals": sorted(series.drop_nulls().unique().to_list()),
                }
            else:
                self.spec[col] = {
                    "dtype": "numeric",
                    "min":   float(series.min()),
                    "max":   float(series.max()),
                }

    def validate(self, df: pl.DataFrame) -> list:
        """
        Return a list of validation errors.
        An empty list means the feature DataFrame is consistent with the spec.
        """
        errors = []
        for col, col_spec in self.spec.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
                continue
            series = df[col]
            if col_spec["dtype"] == "categorical":
                if series.dtype not in (pl.Utf8, pl.Categorical):
                    errors.append(
                        f"{col}: expected categorical dtype, got {series.dtype}. "
                        f"Call .cast(pl.Utf8) before scoring."
                    )
                else:
                    unseen = set(series.drop_nulls().unique().to_list()) - \
                             set(col_spec["unique_vals"])
                    if unseen:
                        errors.append(
                            f"{col}: unseen categories at scoring time: {unseen}. "
                            f"CatBoost will handle these, but verify they are genuine "
                            f"new values and not encoding errors."
                        )
            else:
                if series.dtype == pl.Utf8 or series.dtype == pl.Categorical:
                    errors.append(
                        f"{col}: expected numeric, got {series.dtype}."
                    )
        return errors

    def to_json(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FeatureSpec":
        import json
        obj = cls()
        with open(path) as f:
            obj.spec = json.load(f)
        return obj

# Record the spec at training time
feature_spec = FeatureSpec()
feature_spec.record(
    features_pl.select(FEATURE_COLS),
    cat_features=CAT_FEATURES,
)

# Log it as an MLflow artefact alongside the model
with mlflow.start_run(run_id=freq_run_id):
    feature_spec.to_json("/tmp/feature_spec.json")
    mlflow.log_artifact("/tmp/feature_spec.json", artifact_path="feature_spec")

print("FeatureSpec recorded and logged to MLflow.")
print("\nSpec summary:")
for col, col_spec in feature_spec.spec.items():
    if col_spec["dtype"] == "categorical":
        print(f"  {col:<20} categorical  {col_spec['unique_vals']}")
    else:
        print(f"  {col:<20} numeric      [{col_spec['min']:.2f}, {col_spec['max']:.2f}]")
```

### Delta table VACUUM policy

Before the audit section completes, set the retention policy on all tables. The default Databricks VACUUM retention is 30 days. Consumer Duty requires three years of reproducibility. Set all pipeline tables to 365 days minimum:

```python
for table_name, table_path in TABLES.items():
    try:
        spark.sql(f"""
            ALTER TABLE {table_path}
            SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
        """)
        print(f"Retention set: {table_path}")
    except Exception as e:
        print(f"Could not set retention on {table_path}: {e}")
```