## Part 2: Building the dataset

### What the dataset contains

We use a standard 100,000-policy synthetic UK motor portfolio that is shared across all modules in this course. The data comes from the `insurance-datasets` library, which generates realistic policy characteristics and claims from a known data generating process (DGP). Because the true parameters are known, you can fit GLMs and verify whether the coefficients you recover match the ground truth.

The portfolio has one row per policy with these columns:

- `area` — ABI area band (A through F, A being lowest risk)
- `vehicle_group` — ABI group 1-50
- `ncd_years` — No Claims Discount years, 0-5 (UK scale)
- `driver_age` — 17 to 85
- `conviction_points` — total endorsement points (0 = clean licence)
- `annual_mileage` — estimated annual mileage
- `exposure` — earned policy years (less than 1.0 for cancellations)
- `claim_count` — number of claims in the period
- `incurred` — total incurred cost (0.0 if no claims)

The true DGP uses a Poisson frequency model with a log-linear predictor and a Gamma severity model. The true parameters are documented in `insurance_datasets.MOTOR_TRUE_FREQ_PARAMS` and `insurance_datasets.MOTOR_TRUE_SEV_PARAMS`.

### A new Python concept: the random number generator

The `seed=42` argument to `load_motor()` fixes the random number generator. Using the same seed every time means the data is reproducible — you and a colleague running this same code will get exactly the same 100,000 policies. If you change the seed, you get a different dataset with the same statistical properties.

### Loading the data

Add `insurance-datasets` to the install cell at the top of your notebook (go back and update it now if you have not already):

```python
%pip install polars statsmodels scipy matplotlib insurance-datasets
```

After restarting Python, run this in a new cell:

```python
import polars as pl
import numpy as np
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS, MOTOR_TRUE_SEV_PARAMS

# Load the standard motor portfolio used throughout this course
df = pl.from_pandas(load_motor(n_policies=100_000, seed=42))

# Derive a binary conviction flag (policies with any endorsement points)
df = df.with_columns(
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("conviction_flag")
)

print(f"Portfolio: {len(df):,} policies")
print(f"Exposure: {df['exposure'].sum():,.1f} earned years")
print(f"Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum() / df['exposure'].sum():.3f}/year)")
print(f"Total incurred: £{df['incurred'].sum() / 1e6:.1f}m")
print()
print("True DGP frequency parameters:")
for k, v in MOTOR_TRUE_FREQ_PARAMS.items():
    print(f"  {k}: {v}")
```

**What you should see:** Around 97,000 earned years (policies run shorter than a full year due to cancellations), around 7,000-8,000 claims at roughly 7-8% frequency, and total incurred around £25-35m. The exact numbers depend on the random draws but these ranges are correct.

**What load_motor() did:** It generated 100,000 policies with realistic UK motor characteristics — driver ages weighted towards the 30-60 bracket, NCD distribution reflecting a mature book, ABI vehicle groups centred around 25. It then applied the true DGP to generate claim counts (Poisson) and incurred amounts (Gamma). Exposure varies because about 8% of policies cancel mid-term.

**Why we use a library instead of writing the generation ourselves:** Consistency. Every module in this course starts from the same portfolio. When Module 3 talks about what the GLM misses and the GBM finds, you are comparing models trained on the same data. When Module 5 adds conformal prediction intervals, the calibration dataset is the same portfolio. This is how a real pricing team works — one agreed dataset, multiple models built on top of it.

### A new Python concept: f-strings

The `f"..."` syntax in the print statements is an f-string. The `f` prefix tells Python to interpret `{...}` inside the string as an expression to evaluate. So `f"{df['claim_count'].sum():,}"` prints the claim count sum with thousand separators. The `:,` inside the braces is a format specifier meaning "use commas as thousand separators." The `:.3f` means "three decimal places."

### Sanity-checking the data

Before fitting any model, look at the data. In a new cell:

```python
# First few rows
df.head(5)
```

You should see 18 columns. The `exposure` values should be mostly close to 1.0 (annual policies) with some lower values for cancellations. `claim_count` should be 0 for most rows.

In the next cell:

```python
# Summary statistics
df.describe()
```

This shows count, mean, standard deviation, min, and max for each numeric column. The minimum `claim_count` should be 0, the minimum `exposure` should be above 0.0. If you see negative values in either column, something went wrong.

In the next cell, check a one-way by area:

```python
df.group_by("area").agg(
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
    pl.len().alias("policies"),
).with_columns(
    (pl.col("claims") / pl.col("exposure")).alias("freq")
).sort("area")
```

**What you should see:** Area A should have the lowest claim frequency (it is the base area with no uplift). Area F should have the highest, roughly 1.9x area A. This matches `exp(0.65) ≈ 1.92`, the true area F effect from `MOTOR_TRUE_FREQ_PARAMS`.

**A new Python concept: method chaining.** Polars lets you chain operations with `.`. Each method returns a new DataFrame, so you can chain `.group_by()`, `.agg()`, `.with_columns()`, `.sort()` one after another. This is the same idea as Excel's CTRL+T tables with multiple computed columns, but written in a chain rather than spread across cells.
