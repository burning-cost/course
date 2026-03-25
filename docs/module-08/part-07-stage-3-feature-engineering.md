## Part 7: Stage 3 — Feature engineering

This is the most structurally important stage. Read the NCB encoding incident in Part 1 again. The correct pattern — one function, called identically at training time and scoring time — prevents that failure. The incorrect pattern — feature logic defined anywhere other than this shared layer — creates the conditions for it.

Add a markdown cell:

```python
%md
## Stage 3: Feature engineering — the shared transform layer
```

### Defining the transforms

```python
import polars as pl

# -----------------------------------------------------------------------
# Constants used in feature engineering.
# Module-level, not inside functions, so they can be logged to MLflow.
# -----------------------------------------------------------------------

NCB_MAX = 5   # maximum NCB years in the tariff structure

VEHICLE_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

AGE_MID = {
    "17-25": 21.0,
    "26-35": 30.0,
    "36-50": 43.0,
    "51-65": 58.0,
    "66+":   72.0,
}

MILEAGE_ORD = {"<5k": 1, "5k-10k": 2, "10k-15k": 3, "15k+": 4}

# -----------------------------------------------------------------------
# Transform functions.
# Each takes a Polars DataFrame and returns a Polars DataFrame.
# All original columns are preserved; new columns are added.
# -----------------------------------------------------------------------

def encode_ncb(df: pl.DataFrame) -> pl.DataFrame:
    """
    NCB deficit = NCB_MAX - ncb_years.
    Deficit 0 = full no-claims discount. Deficit 5 = no discount.
    CatBoost treats this as continuous — the monotone relationship with
    frequency (higher deficit = more claims) is directly learnable.
    """
    return df.with_columns(
        (NCB_MAX - pl.col("ncb")).alias("ncb_deficit")
    )

def encode_vehicle(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vehicle group A-E mapped to ordinal 1-5.
    CatBoost can learn a monotone or non-monotone response to this.
    Passing as ordinal integer (not categorical) lets CatBoost model
    the gradient across the group spectrum.
    """
    return df.with_columns(
        pl.col("vehicle_group")
          .replace(VEHICLE_ORD)
          .cast(pl.Int32)
          .alias("vehicle_ord")
    )

def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Age band mapped to numeric midpoint.
    The GBM can model the non-linear age-frequency curve directly.
    """
    return df.with_columns(
        pl.col("age_band")
          .replace(AGE_MID)
          .cast(pl.Float64)
          .alias("age_mid")
    )

def encode_mileage(df: pl.DataFrame) -> pl.DataFrame:
    """
    Annual mileage band mapped to ordinal 1-4.
    """
    return df.with_columns(
        pl.col("annual_mileage")
          .replace(MILEAGE_ORD)
          .cast(pl.Int32)
          .alias("mileage_ord")
    )

def add_log_exposure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-compute log(exposure) for use as the Poisson offset.
    Stored in the features table so downstream stages do not need
    to recompute it from the raw exposure column.
    """
    return df.with_columns(
        pl.col("exposure").log().alias("log_exposure")
    )

# -----------------------------------------------------------------------
# The master transform list and feature specification.
# Change a feature: add or modify a transform here. Nowhere else.
# -----------------------------------------------------------------------

TRANSFORMS = [
    encode_ncb,
    encode_vehicle,
    encode_age,
    encode_mileage,
    add_log_exposure,
]

# FEATURE_COLS: columns passed to CatBoost as model input.
# Does not include log_exposure (passed as baseline), claim_count, or incurred_loss.
FEATURE_COLS = ["ncb_deficit", "vehicle_ord", "age_mid", "mileage_ord", "region"]

# CAT_FEATURES: subset of FEATURE_COLS that CatBoost should treat as categorical.
# CatBoost handles categories natively — do not one-hot encode them.
CAT_FEATURES = ["region"]


def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply all transforms in TRANSFORMS order.

    This is the only function that should be called at training time
    and scoring time. Never call individual transforms manually outside
    this function — that is how the encoding divergence happens.
    """
    for fn in TRANSFORMS:
        df = fn(df)
    return df


# Test on a small sample
sample = raw_pl.head(5)
sample_t = apply_transforms(sample)
print("Transform test:")
print(sample_t.select(["age_band", "age_mid", "ncb", "ncb_deficit",
                        "vehicle_group", "vehicle_ord", "annual_mileage", "mileage_ord",
                        "exposure", "log_exposure"]))
```

**What you should see:** A row with `age_band="17-25"` should have `age_mid=21.0`. `ncb=5` should give `ncb_deficit=0`. `vehicle_group="C"` should give `vehicle_ord=3`. `mileage_ord` should match the MILEAGE_ORD mapping.

### Applying the transforms and writing to Delta

```python
features_pl = apply_transforms(raw_pl)

# Verify all feature columns are present
missing = [c for c in FEATURE_COLS if c not in features_pl.columns]
if missing:
    raise ValueError(f"Missing feature columns after transforms: {missing}")

print(f"Features shape: {features_pl.shape}")
print(f"Feature columns: {FEATURE_COLS}")
print(f"All feature cols present: True")

(
    spark.createDataFrame(features_pl.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["features"])
)

spark.sql(f"""
    ALTER TABLE {TABLES['features']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

feat_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['features']} LIMIT 1"
).collect()[0]["version"]

print(f"\nFeatures table: {TABLES['features']}")
print(f"Delta version:  {feat_version}")
```

### The FeatureSpec guard

Beyond the shared transform layer, a `FeatureSpec` records the expected dtype and range of each feature at training time and validates incoming data against it at scoring time. This is the second line of defence against the NCB encoding incident: if a scoring pipeline passes `ncb_deficit` as a float when the spec records it as Int32, the validator raises an error immediately.

```python
import json

class FeatureSpec:
    """
    Records dtype and range of each feature at training time.
    Validates scoring data against the spec at inference time.
    """

    def __init__(self):
        self.spec: dict = {}

    def record(self, df: pl.DataFrame, cat_features: list[str]) -> None:
        for col in df.columns:
            s = df[col]
            if col in cat_features or s.dtype == pl.Utf8:
                self.spec[col] = {
                    "dtype":       "categorical",
                    "unique_vals": sorted(s.drop_nulls().unique().to_list()),
                }
            else:
                self.spec[col] = {
                    "dtype": "numeric",
                    "min":   float(s.min()),
                    "max":   float(s.max()),
                }

    def validate(self, df: pl.DataFrame) -> list[str]:
        """Return a list of validation errors. Empty list = OK."""
        errors = []
        for col, spec in self.spec.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
                continue
            s = df[col]
            if spec["dtype"] == "categorical":
                if s.dtype not in (pl.Utf8, pl.Categorical):
                    errors.append(
                        f"{col}: expected categorical, got {s.dtype}. "
                        f"Likely encoding divergence — check apply_transforms()."
                    )
                else:
                    unseen = set(s.drop_nulls().unique().to_list()) - set(spec["unique_vals"])
                    if unseen:
                        errors.append(
                            f"{col}: unseen categories {unseen}. "
                            f"Verify these are genuine new values, not encoding errors."
                        )
            else:
                if s.dtype in (pl.Utf8, pl.Categorical):
                    errors.append(
                        f"{col}: expected numeric, got {s.dtype}."
                    )
        return errors

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FeatureSpec":
        obj = cls()
        with open(path) as f:
            obj.spec = json.load(f)
        return obj


# Record the spec from the full feature set
feature_spec = FeatureSpec()
feature_spec.record(features_pl.select(FEATURE_COLS), cat_features=CAT_FEATURES)

# Save to /tmp for MLflow logging (happens after freq model is trained in Stage 6)
feature_spec.to_json("/tmp/feature_spec.json")

print("FeatureSpec recorded.")
for col, col_spec in feature_spec.spec.items():
    if col_spec["dtype"] == "categorical":
        print(f"  {col:<15} categorical  {col_spec['unique_vals']}")
    else:
        print(f"  {col:<15} numeric      [{col_spec['min']:.2f}, {col_spec['max']:.2f}]")
```
