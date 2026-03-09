## Part 4: Regenerating the motor dataset

We use the same 100,000-policy synthetic UK motor portfolio from Module 2. Load it from the `insurance-datasets` library so the base data is identical to what you used there.

Create a new cell with a markdown header to keep the notebook organised. In Databricks, a cell starting with `%md` renders as formatted text rather than running Python:

```python
%md
## Part 4: Data preparation
```

Now create the next cell and paste in the data loading code:

```python
from insurance_datasets import load_motor
import polars as pl
import numpy as np

# Same portfolio as Module 2 — same seed, same n_policies
df = pl.from_pandas(load_motor(n_policies=100_000, seed=42))

# This module introduces a superadditive interaction between young drivers
# (under 25) and high vehicle groups (above 35). We add this as a feature
# rather than baking it into the DGP — the base dataset is the same across
# all modules, and we demonstrate below that the GLM misses this interaction.
df = df.with_columns(
    (
        (pl.col("driver_age") < 25) & (pl.col("vehicle_group") > 35)
    ).cast(pl.Int32).alias("young_high_vg"),
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("has_convictions"),
)

# Conviction points encoded to discrete levels for the GBM
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

print(f"Total policies: {len(df):,}")
print(f"Total claims:   {df['claim_count'].sum():,}")
print(f"Total exposed:  {df['exposure'].sum():,.0f} policy-years")
print(f"Claim frequency:{df['claim_count'].sum() / df['exposure'].sum():.4f}")
print()
print("Policies by accident year:")
print(
    df.group_by("accident_year")
    .agg(
        pl.len().alias("policies"),
        pl.col("claim_count").sum().alias("claims"),
        pl.col("exposure").sum().alias("exposure"),
    )
    .with_columns((pl.col("claims") / pl.col("exposure")).alias("freq"))
    .sort("accident_year")
)
```

Run this cell. It takes a second or two. The output shows the first five rows of the DataFrame as a table.

What you are looking at: 100,000 motor policies across accident years 2019-2023. Each row is one policy. The `accident_year` column is important — we use it to construct our cross-validation folds later.

The key concept for this module: the `young_high_vg` column flags young drivers (under 25) in high vehicle groups (above 35). These policies are worse than the multiplicative combination of the youth penalty and the vehicle group effect would suggest — this is a superadditive interaction. This is the signal that the GLM underestimates and the GBM should find.

The base dataset from `load_motor()` does not plant this interaction in the DGP. Instead, we define it as a feature and show that the GBM discovers it from the data patterns, while the main-effects GLM cannot. In Module 4, we use SHAP values to confirm the discovery.

Now check some basic counts:

```python
print(f"Young driver + high vehicle group: {df['young_high_vg'].sum():,} policies ({df['young_high_vg'].mean()*100:.1f}%)")
print(f"Any conviction points:              {df['has_convictions'].sum():,} policies ({df['has_convictions'].mean()*100:.1f}%)")
```

You should see roughly 4-6% of policies with the young_high_vg flag and around 6-8% with any conviction points. The frequency for the young_high_vg group will be substantially higher than the product of the young driver and high vehicle group effects alone — this is the pattern the GBM will detect.
