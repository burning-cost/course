## Part 3: Recreating the training data

We use the same synthetic UK motor portfolio from Module 3. Load it from `insurance-datasets` — the base data is identical to Modules 2 and 3.

The superadditive interaction between young drivers (under 25) and high vehicle groups (above 35) is the signal the GBM finds that the GLM misses. We add it here as a feature column for validation purposes — we know it exists because we can see it in the data, and the SHAP analysis will confirm the GBM has learned it.

In a new cell, type this and run it (Shift+Enter):

```python
%md
## Part 3: Data preparation
```

Running a cell that starts with `%md` renders it as formatted markdown text in the notebook. This keeps your notebook organised with visible section headings.

Now in a new cell, type this and run it (Shift+Enter):

```python
from insurance_datasets import load_motor
import polars as pl
import numpy as np

# Same portfolio used in Modules 2 and 3
df = pl.from_pandas(load_motor(n_policies=100_000, seed=42))

# Feature engineering consistent with Module 3
df = df.with_columns(
    (
        (pl.col("driver_age") < 25) & (pl.col("vehicle_group") > 35)
    ).cast(pl.Int32).alias("young_high_vg"),
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("has_convictions"),
)

print(f"Rows:            {len(df):,}")
print(f"Total claims:    {df['claim_count'].sum():,}")
print(f"Total exposure:  {df['exposure'].sum():,.0f} policy-years")
print(f"Claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
df.head(3)
```

The output looks like:

```text
Rows:            100,000
Total claims:    ~7,500
Total exposure:  ~97,000 policy-years
Claim frequency: 0.077
```

A claim frequency around 7-8% on a motor book is realistic for a standard UK personal lines portfolio.

Now add the engineered features. In a new cell, type this and run it (Shift+Enter):

```python
# Final feature list for the frequency model
FREQ_FEATURES = ["area", "ncd_years", "has_convictions", "vehicle_group", "driver_age", "annual_mileage"]
CAT_FEATURES  = ["area", "has_convictions"]
CONT_FEATURES = ["ncd_years", "vehicle_group", "driver_age", "annual_mileage"]

print("Feature lists set:")
print(f"  FREQ_FEATURES: {FREQ_FEATURES}")
print(f"  CAT_FEATURES:  {CAT_FEATURES}")
print(f"  CONT_FEATURES: {CONT_FEATURES}")
```

You will see the feature list printed. No errors means the feature engineering worked.

**Why `conviction_points` becomes `has_convictions`:** The raw conviction points (0, 3, 6, 9) contain ordinal information, but the most important split is clean vs. any conviction. A binary flag is simpler and more interpretable for committee presentation. In practice you would test both encodings.

**Why annual_mileage is included here but not in Module 2:** The GLM in Module 2 uses a minimal factor set to demonstrate the GLM workflow clearly. The GBM in Module 3 and SHAP analysis in Module 4 benefit from richer features because the GBM can find non-linear effects and interactions across more variables. `annual_mileage` is in the base dataset from `load_motor()` and is a genuine risk factor.
