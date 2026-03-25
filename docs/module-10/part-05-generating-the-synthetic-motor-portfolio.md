## Part 5: Generating the synthetic motor portfolio

We use the standard synthetic dataset from Modules 2–5, then extend it by resampling claim counts with two known interaction effects added. This gives us ground truth to validate the detector against: we know exactly which pairs have genuine interactions in the data.

```python
from insurance_datasets import load_motor, TRUE_FREQ_PARAMS
import polars as pl
import numpy as np

# Base portfolio — same 100,000 policies as all previous modules
df_raw = load_motor(n_policies=100_000, seed=42, polars=True)
```

The `load_motor` DGP is purely additive main effects. We add interactions by resampling claim counts with two additional log-multipliers. Starting from the true additive DGP, adding the interaction terms, and drawing new Poisson realisations is the honest way to plant known interactions in synthetic data.

```python
rng = np.random.default_rng(seed=0)   # separate seed from the DGP seed

driver_age_raw    = df_raw["driver_age"].to_numpy()
vehicle_group_raw = df_raw["vehicle_group"].to_numpy()
ncd_years_raw     = df_raw["ncd_years"].to_numpy()
conviction_pts    = df_raw["conviction_points"].to_numpy()
exposure_raw      = df_raw["exposure"].to_numpy()
area_raw          = df_raw["area"].to_numpy()

# Interaction 1: young driver (age < 25) in high vehicle group (> 35)
# Adds 0.30 log-units to claim frequency — supermultiplicative risk
ix1_mask = (driver_age_raw < 25) & (vehicle_group_raw > 35)

# Interaction 2: ncd_years >= 3 AND has a conviction
# Drivers with substantial NCD who still hold active convictions are extra risky.
# The product ncd_years * has_convictions is largest for this group, so the GLM's
# product interaction term represents this correctly: a positive coefficient means
# the more NCD years a convicted driver holds, the worse the risk adjustment.
has_conv_mask = conviction_pts > 0
ix2_mask = (ncd_years_raw >= 3) & has_conv_mask

# Reconstruct the base DGP log-frequency (matches TRUE_FREQ_PARAMS from insurance-datasets)
# The age effect blends linearly from the young effect down to zero between ages 25–30
age_effect = np.zeros(len(df_raw))
age_effect[driver_age_raw < 25] = TRUE_FREQ_PARAMS["driver_age_young"]
age_effect[driver_age_raw >= 70] = TRUE_FREQ_PARAMS["driver_age_old"]
blend_mask = (driver_age_raw >= 25) & (driver_age_raw < 30)
age_effect[blend_mask] = TRUE_FREQ_PARAMS["driver_age_young"] * (30 - driver_age_raw[blend_mask]) / 5.0

area_effect = np.zeros(len(df_raw))
for band, key in [("B", "area_B"), ("C", "area_C"), ("D", "area_D"),
                  ("E", "area_E"), ("F", "area_F")]:
    area_effect[area_raw == band] = TRUE_FREQ_PARAMS[key]

log_lambda = (
    TRUE_FREQ_PARAMS["intercept"]
    + TRUE_FREQ_PARAMS["vehicle_group"] * vehicle_group_raw
    + age_effect
    + TRUE_FREQ_PARAMS["ncd_years"] * ncd_years_raw
    + TRUE_FREQ_PARAMS["has_convictions"] * has_conv_mask.astype(float)
    + area_effect
    + np.log(np.clip(exposure_raw, 1e-6, None))
    + 0.30 * ix1_mask.astype(float)   # Interaction 1: +0.30 log-units
    + 0.20 * ix2_mask.astype(float)   # Interaction 2: +0.20 log-units
)

claim_count_with_ix = rng.poisson(np.exp(log_lambda))

# Build the working DataFrame with the interaction-augmented claim counts
df = df_raw.with_columns([
    pl.Series("claim_count", claim_count_with_ix),
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("has_convictions"),
    ((pl.col("driver_age") < 25) & (pl.col("vehicle_group") > 35)
     ).cast(pl.Int32).alias("young_high_vg"),
    ((pl.col("ncd_years") >= 3) & (pl.col("conviction_points") > 0)
     ).cast(pl.Int32).alias("ncd3_convicted"),
])

claim_count = df["claim_count"].to_numpy()
exposure    = df["exposure"].to_numpy()

print(f"Policies:     {len(df):,}")
print(f"Claim count:  {claim_count.sum():,}")
print(f"Mean freq:    {(claim_count.sum() / exposure.sum()):.4f}")

young_high = df.filter(pl.col("young_high_vg") == 1)
ncd3_conv  = df.filter(pl.col("ncd3_convicted") == 1)

print(f"\nyoung_high_vg group:   {len(young_high):,} policies, "
      f"freq={young_high['claim_count'].sum() / young_high['exposure'].sum():.4f}")
print(f"ncd3_convicted group:  {len(ncd3_conv):,} policies, "
      f"freq={ncd3_conv['claim_count'].sum() / ncd3_conv['exposure'].sum():.4f}")
```

**What to notice:** The `young_high_vg` group shows materially higher claim frequency than the product of the young driver and high vehicle group main effects would predict. The `ncd3_convicted` group similarly shows a frequency bump beyond the sum of the NCD and conviction effects. These are the interactions the GLM with main effects alone cannot capture, and the ones the neural interaction detector should score highest.

### Create the feature DataFrame

```python
driver_age_arr     = df["driver_age"].to_numpy()
vehicle_group_arr  = df["vehicle_group"].to_numpy()
annual_mileage_arr = df["annual_mileage"].to_numpy()

age_band = np.select(
    [driver_age_arr < 22, driver_age_arr < 26, driver_age_arr < 35,
     driver_age_arr < 50, driver_age_arr < 70],
    ["17-21", "22-25", "26-34", "35-49", "50-69"],
    default="70+"
)

vg_band = np.select(
    [vehicle_group_arr <= 10, vehicle_group_arr <= 20,
     vehicle_group_arr <= 30, vehicle_group_arr <= 40],
    ["1-10", "11-20", "21-30", "31-40"],
    default="41-50"
)

mileage_band = np.select(
    [annual_mileage_arr < 8_000, annual_mileage_arr < 15_000,
     annual_mileage_arr < 25_000],
    ["low", "medium", "high"],
    default="very_high"
)

X = pl.DataFrame({
    "area":            df["area"].to_numpy(),
    "vehicle_group":   vg_band,
    "ncd_years":       df["ncd_years"].to_numpy().astype(np.int32),
    "age_band":        age_band,
    "has_convictions": df["has_convictions"].to_numpy().astype(np.int32),
    "annual_mileage":  mileage_band,
}).with_columns([
    pl.col("area").cast(pl.Categorical),
    pl.col("vehicle_group").cast(pl.Categorical),
    pl.col("age_band").cast(pl.Categorical),
    pl.col("annual_mileage").cast(pl.Categorical),
])

y            = claim_count.astype(np.float64)
exposure_arr = exposure.astype(np.float64)

print("Feature DataFrame shape:", X.shape)
print("Columns:", X.columns)
print("\nArea distribution:")
print(X["area"].value_counts().sort("area"))
```

**Why we discretise:** GLMs in personal lines pricing work with banded continuous variables. The interaction detection pipeline works on whatever feature representation you give it. Using the banded version here keeps the tutorial consistent with production practice.

Note on the feature set: `X` contains `has_convictions` (binary integer: 1 if conviction_points > 0, stored as `pl.Int32`) and `ncd_years` (continuous integer 0–5, stored as `pl.Int32`). Since neither column has a Polars Categorical or String dtype, the library treats both as continuous and encodes their interaction as a single product column `_ix_ncd_years_has_convictions`. The product `ncd_years * has_convictions` is largest when `ncd_years >= 3` and `has_convictions == 1` — the cell where the second planted interaction fires. A positive GLM coefficient on this term captures what we planted: extra risk that grows with NCD years for convicted drivers.
