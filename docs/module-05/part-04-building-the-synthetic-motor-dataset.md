## Part 4: Building the synthetic motor dataset

We use the same synthetic UK motor portfolio as Modules 2, 3, and 4. The dataset is loaded from `insurance-datasets` with the same seed and size, so the portfolio is identical across all modules.

Add a markdown cell to keep the notebook organised. In Databricks, cells starting with `%md` render as formatted text rather than code:

```python
%md
## Part 4: Data preparation
```

Now in a new cell, paste this and run it:

```python
from insurance_datasets import load_motor
import polars as pl
import numpy as np

# Same 100,000-policy portfolio as Modules 2-4
df = pl.from_pandas(load_motor(n_policies=100_000, seed=42))

# Feature engineering: superadditive interaction (same as Module 3)
# load_motor() provides: age, vehicle_age, vehicle_group, region, credit_score,
# exposure, claim_count, claim_amount
df = df.with_columns(
    (
        (pl.col("age") < 25) & (pl.col("vehicle_group") > 35)
    ).cast(pl.Int32).alias("young_high_vg"),
)

# Assign synthetic accident years — load_motor() does not include accident_year,
# so we create a plausible temporal column for the three-way split below.
rng_year = np.random.default_rng(seed=42)
accident_year = rng_year.choice([2019, 2020, 2021, 2022, 2023], size=len(df),
                                 p=[0.15, 0.17, 0.20, 0.23, 0.25])
df = df.with_columns(pl.Series("accident_year", accident_year.astype(np.int32)))

# Sort chronologically — essential for the temporal split below
df = df.sort("accident_year")

# Pure premium: loss cost per year of exposure
# This is what we model with Tweedie when we want a combined frequency-severity signal
df = df.with_columns(
    (pl.col("claim_amount") / pl.col("exposure")).alias("pure_premium")
)

print(f"Dataset: {len(df):,} rows")
print(f"Accident years: {df['accident_year'].min()} - {df['accident_year'].max()}")
print(f"Overall claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
print(f"Mean pure premium: £{df['pure_premium'].mean():.2f}")
print(f"Zero-claim rows: {(df['claim_count'] == 0).mean():.1%}")
df.head(5)
```

**What you should see:**

```text
Dataset: 100,000 rows
Accident years: 2019 - 2023
Overall claim frequency: 0.077
Mean pure premium: £xxx.xx
Zero-claim rows: ~92-94%
```

The exact numbers will match across modules because you use the same seed. If you see a `KeyError` or `NameError`, check that the cell above (the imports) ran successfully first.

**What this does:** Loads 100,000 motor policies with realistic UK characteristics and a Poisson-Gamma claims DGP. The `load_motor()` function provides columns `policy_id`, `age`, `vehicle_age`, `vehicle_group`, `region`, `credit_score`, `exposure`, `claim_count`, and `claim_amount`. Because `accident_year` is not part of the dataset, we attach a synthetic one drawn from a realistic year distribution — this is only needed for the temporal split and does not affect the modelling. The sort by `accident_year` is then essential: without it, the temporal split is meaningless. The `pure_premium` column is the ratio of claim amount to exposure, which is the response variable for the Tweedie model.

**Why the same dataset across all modules:** Conformal prediction intervals calibrated in this module will be applied to the same portfolio that the GBM was trained on in Module 3 and the SHAP relativities were extracted from in Module 4. Consistency matters: if you calibrate on a different data generating process to the one you train on, the coverage guarantees break down. Using the same library dataset eliminates this risk.
