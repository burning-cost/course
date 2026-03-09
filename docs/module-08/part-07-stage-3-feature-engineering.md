## Part 7: Stage 3 -- Feature engineering

This is the most structurally important stage in the pipeline. Read it carefully.

The failure mode described in Part 1 -- the NCB encoding incident -- happens because feature engineering is defined in two places: the training notebook and the scoring code. When they diverge, the model produces wrong predictions silently.

The solution is to define all feature transforms once, in a single list, and call the same `apply_transforms()` function at training time and scoring time. This is not a style preference. It is the discipline that prevents the most common production failure in ML-based pricing.

Add a markdown cell:

```python
%md
## Stage 3: Feature engineering -- the shared transform layer
```

### Defining the transforms

```python
# -----------------------------------------------------------------------
# Constants used in transforms.
# Module-level, not inside functions, so they can be logged to MLflow
# and serialised alongside the model.
# -----------------------------------------------------------------------

# load_motor() columns available:
#   policy_id, age, vehicle_age, vehicle_group, region,
#   credit_score, exposure, claim_count, claim_amount, accident_year

AGE_BAND_BREAKS = [0, 25, 36, 51, 66, 999]   # right-exclusive
AGE_BAND_LABELS = ["17-25", "26-35", "36-50", "51-65", "66+"]
AGE_MID = {
    "17-25": 21,
    "26-35": 30,
    "36-50": 43,
    "51-65": 58,
    "66+":   72,
}

# -----------------------------------------------------------------------
# Transform functions: pure functions, one job each.
# Each takes a Polars DataFrame and returns a Polars DataFrame.
# The output DataFrame contains all original columns plus the new ones.
# -----------------------------------------------------------------------

def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Band driver age and map to numeric midpoint.
    age_mid is a continuous approximation of the band's centre.
    The GBM can model non-linear age effects correctly with this encoding.
    """
    return df.with_columns(
        pl.when(pl.col("age") < 25).then(pl.lit("17-25"))
          .when(pl.col("age") < 36).then(pl.lit("26-35"))
          .when(pl.col("age") < 51).then(pl.lit("36-50"))
          .when(pl.col("age") < 66).then(pl.lit("51-65"))
          .otherwise(pl.lit("66+"))
          .alias("age_band")
    ).with_columns(
        pl.col("age_band")
          .replace(AGE_MID)
          .cast(pl.Float64)
          .alias("age_mid")
    )

def add_young_high_vg(df: pl.DataFrame) -> pl.DataFrame:
    """
    Superadditive interaction: young driver × high vehicle group.
    Young drivers in high-group vehicles are disproportionately risky.
    """
    return df.with_columns(
        (
            (pl.col("age") < 25) & (pl.col("vehicle_group") > 35)
        ).cast(pl.Int32).alias("young_high_vg")
    )

def add_log_exposure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-compute log(exposure) so it is always available in the feature table.
    This is the offset term for the Poisson model.
    We store it in the features table so downstream steps do not need to
    recompute it from the raw exposure column.
    """
    return df.with_columns(
        pl.col("exposure").log().alias("log_exposure")
    )

# -----------------------------------------------------------------------
# The master transform list and feature column specification.
# These are the ONLY definitions you should change if you add a new feature.
# -----------------------------------------------------------------------
TRANSFORMS   = [encode_age, add_young_high_vg, add_log_exposure]

# FEATURE_COLS: the columns passed to CatBoost as input features.
# This does NOT include log_exposure, claim_count, or claim_amount.
FEATURE_COLS = ["age_mid", "vehicle_age", "vehicle_group", "region", "credit_score", "young_high_vg"]

# CAT_FEATURES: the subset of FEATURE_COLS that CatBoost should treat as categorical.
# CatBoost handles categories natively -- do not one-hot encode them.
CAT_FEATURES = ["region"]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply all transforms in order.
    Call this function identically at training time and scoring time.
    Never apply individual transforms manually outside this function.
    """
    for fn in TRANSFORMS:
        df = fn(df)
    return df

# Test the transforms on a small sample
sample = raw_pl.head(5)
sample_transformed = apply_transforms(sample)
print("Transform test passed.")
print("Columns after transforms:")
print([c for c in sample_transformed.columns])
print("\nFirst row (selected columns):")
print(sample_transformed.select(["age", "age_band", "age_mid",
                                  "vehicle_group", "young_high_vg",
                                  "exposure", "log_exposure"]))
```

**What you should see:** The `age_mid` for a policy with `age=23` should be 21. For `age=43`, it should be 43. The `young_high_vg` flag should be 1 for policies with `age < 25` and `vehicle_group > 35`, and 0 otherwise. The `log_exposure` should be the natural log of the exposure value.

### Applying the transforms and writing to Delta

```python
# Apply to the full dataset
features_pl = apply_transforms(raw_pl)

print(f"Features shape: {features_pl.shape}")
print(f"Feature columns present: {[c for c in FEATURE_COLS if c in features_pl.columns]}")
print(f"All feature cols present: {all(c in features_pl.columns for c in FEATURE_COLS)}")

# Write to Delta
spark.createDataFrame(features_pl.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["features"])

spark.sql(f"""
    ALTER TABLE {TABLES['features']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

# Log the features table version
feat_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['features']} LIMIT 1"
).collect()[0]["version"]

print(f"\nFeatures table: {TABLES['features']}")
print(f"Delta version:  {feat_version}")
```
