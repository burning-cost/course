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
NCB_MAX     = 5
VEHICLE_ORD = {"A":1,"B":2,"C":3,"D":4,"E":5}   # not used in this data, shown as pattern

AGE_MID = {
    "17-25": 21,
    "26-35": 30,
    "36-50": 43,
    "51-65": 58,
    "66+":   72,
}

MILEAGE_ORD = {
    "<5k":    1,
    "5k-10k": 2,
    "10k-15k":3,
    "15k+":   4,
}

# -----------------------------------------------------------------------
# Transform functions: pure functions, one job each.
# Each takes a Polars DataFrame and returns a Polars DataFrame.
# The output DataFrame contains all original columns plus the new ones.
# -----------------------------------------------------------------------

def encode_ncb(df: pl.DataFrame) -> pl.DataFrame:
    """
    NCB deficit = NCB_MAX - ncb_years.
    Puts new drivers (ncb=0) at deficit=5 (high) and
    fully discounted drivers (ncb=5) at deficit=0 (low).
    The GBM can then find a monotone increasing effect for the deficit.
    """
    return df.with_columns(
        (NCB_MAX - pl.col("ncb_years")).alias("ncb_deficit")
    )

def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map age band to a numeric midpoint.
    age_mid is a continuous approximation of the band's centre.
    The GBM can model non-linear age effects correctly with this encoding.
    """
    return df.with_columns(
        pl.col("age_band")
          .replace(AGE_MID)
          .cast(pl.Float64)
          .alias("age_mid")
    )

def encode_mileage(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map mileage band to an ordinal integer.
    Preserves the ordering (<5k < 5k-10k < 10k-15k < 15k+).
    """
    return df.with_columns(
        pl.col("annual_mileage")
          .replace(MILEAGE_ORD)
          .cast(pl.Int32)
          .alias("mileage_ord")
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
TRANSFORMS   = [encode_ncb, encode_age, encode_mileage, add_log_exposure]

# FEATURE_COLS: the columns passed to CatBoost as input features.
# This does NOT include log_exposure or claim_count or incurred_loss.
FEATURE_COLS = ["ncb_deficit", "vehicle_group", "age_mid", "mileage_ord", "region"]

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
print(sample_transformed.select(["ncb_years","ncb_deficit","age_band","age_mid",
                                  "annual_mileage","mileage_ord","exposure","log_exposure"]))
```

**What you should see:** The `ncb_deficit` column for a policy with `ncb_years=5` should be 0. For `ncb_years=0`, it should be 5. The `age_mid` for `age_band="36-50"` should be 43. The `mileage_ord` for `"10k-15k"` should be 3. The `log_exposure` should be the natural log of the exposure value.

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