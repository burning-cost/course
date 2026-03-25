## Part 5: Generating the synthetic motor portfolio

We use the same synthetic dataset from Modules 2-5. Load it from `insurance-datasets` with the same seed so the portfolio is identical throughout the course.

```python
from insurance_datasets import load_motor
import polars as pl
import numpy as np

# Same 100,000-policy portfolio as all previous modules
# polars=True returns a Polars DataFrame directly
df_raw = load_motor(n_policies=100_000, seed=42, polars=True)
```

This module needs two planted interactions that we demonstrate the neural interaction detector can find:

1. `driver_age < 25` AND `vehicle_group > 35`: a supermultiplicative bump for young drivers in high-group vehicles. This is the interaction the Module 3 GLM missed and the GBM found.
2. `ncd_years == 0` AND `conviction_points > 0`: zero NCD combined with conviction points adds extra risk beyond what the main effects capture.

```python
# Add the two interactions as explicit features — we know they exist
# The NID will score these two interactions highly, validating the detection
df = df_raw.with_columns(
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("has_convictions"),
    (
        (pl.col("driver_age") < 25) & (pl.col("vehicle_group") > 35)
    ).cast(pl.Int32).alias("young_high_vg"),
    (
        (pl.col("ncd_years") == 0) & (pl.col("conviction_points") > 0)
    ).cast(pl.Int32).alias("zero_ncd_convicted"),
)

claim_count = df["claim_count"].to_numpy()
exposure = df["exposure"].to_numpy()

print(f"Policies:     {len(df):,}")
print(f"Claim count:  {claim_count.sum():,}")
print(f"Mean freq:    {(claim_count.sum() / exposure.sum()):.4f}")

# Check the two interaction groups
young_high = df.filter(pl.col("young_high_vg") == 1)
zero_ncd_conv = df.filter(pl.col("zero_ncd_convicted") == 1)

print(f"\nyoung_high_vg group:      {len(young_high):,} policies, "
      f"freq={young_high['claim_count'].sum() / young_high['exposure'].sum():.4f}")
print(f"zero_ncd_convicted group: {len(zero_ncd_conv):,} policies, "
      f"freq={zero_ncd_conv['claim_count'].sum() / zero_ncd_conv['exposure'].sum():.4f}")
```

**What to notice:** The `young_high_vg` group will show materially higher claim frequency than the product of the young driver and high vehicle group main effects would predict. The `zero_ncd_convicted` group similarly shows a frequency bump beyond the sum of the NCD=0 and conviction effects. These are exactly the kind of interactions a GLM with main effects alone cannot capture — and the ones the neural interaction detector should score highest.

### Create the feature DataFrame

```python
# Discretise continuous features as we would in a real pricing model
driver_age_arr = df["driver_age"].to_numpy()
vehicle_group_arr = df["vehicle_group"].to_numpy()
annual_mileage_arr = df["annual_mileage"].to_numpy()

age_band = np.select(
    [driver_age_arr < 22, driver_age_arr < 26, driver_age_arr < 35, driver_age_arr < 50, driver_age_arr < 70],
    ["17-21", "22-25", "26-34", "35-49", "50-69"],
    default="70+"
)

vg_band = np.select(
    [vehicle_group_arr <= 10, vehicle_group_arr <= 20, vehicle_group_arr <= 30, vehicle_group_arr <= 40],
    ["1-10", "11-20", "21-30", "31-40"],
    default="41-50"
)

mileage_band = np.select(
    [annual_mileage_arr < 8_000, annual_mileage_arr < 15_000, annual_mileage_arr < 25_000],
    ["low", "medium", "high"],
    default="very_high"
)

X = pl.DataFrame({
    "area":              df["area"].to_numpy(),
    "vehicle_group":     vg_band,
    "ncd_years":         df["ncd_years"].to_numpy().astype(np.int32),
    "age_band":          age_band,
    "has_convictions":   df["has_convictions"].to_numpy().astype(np.int32),
    "annual_mileage":    mileage_band,
}).with_columns([
    pl.col("area").cast(pl.Categorical),
    pl.col("vehicle_group").cast(pl.Categorical),
    pl.col("age_band").cast(pl.Categorical),
    pl.col("annual_mileage").cast(pl.Categorical),
])

y = claim_count.astype(np.float64)
exposure_arr = exposure.astype(np.float64)

print("Feature DataFrame shape:", X.shape)
print("Columns:", X.columns)
print("\nArea distribution:")
print(X["area"].value_counts().sort("area"))
```

**Why we discretise:** GLMs in personal lines pricing work with banded continuous variables. A 50-level vehicle group becomes a 5-band version, reducing parameter count and improving credibility. The interaction detection pipeline works on whatever feature representation you give it. Using the banded version here keeps the tutorial consistent with what you would do in production.

Note on the feature set: `X` contains `has_convictions` (binary integer: 1 if conviction_points > 0, stored as `pl.Int32`) rather than the raw `conviction_points` integer. The second planted interaction is between `ncd_years` (continuous integer, 0-5) and `has_convictions` (binary integer 0/1). Since neither column has a Polars Categorical or String dtype, the library treats both as continuous and encodes the interaction as a single product column `_ix_ncd_years_has_convictions` — one new parameter. This is the correct encoding: effectively a slope on `ncd_years` that switches on when `has_convictions == 1`.
