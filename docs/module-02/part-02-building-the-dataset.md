## Part 2: Building the dataset

### What we are generating

We need 100,000 synthetic motor policies with these attributes:

- Area band (A through F, roughly corresponding to postcode area bands)
- ABI vehicle group (1-50)
- NCD years (0-5)
- Driver age (17-85)
- Conviction flag (0 or 1)
- Earned exposure (fraction of a policy year, 0.05 to 1.0)

For each policy we generate a claim count (from the Poisson process) and, for claimed policies, an average severity (from the Gamma process). We know the true parameters because we define them ourselves.

### A new Python concept: the random number generator

`np.random.default_rng(seed=42)` creates a random number generator with a fixed starting point (`seed=42`). Using the same seed every time means the synthetic data is reproducible - you and a colleague running this same notebook will get exactly the same 100,000 policies. If you change the seed, you get a different dataset with the same statistical properties.

### A new Python concept: `np.where`

`np.where(condition, value_if_true, value_if_false)` applies a condition to every element of an array and returns a new array. It is the vectorised equivalent of an IF statement in Excel. When you see `np.where(area == "B", 0.10, 0)`, it returns an array where each element is 0.10 if the corresponding policy is in area B, and 0 otherwise.

### Generating the data

Create a new cell and run this. We will explain what happened immediately afterwards.

```python
import polars as pl
import numpy as np

rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors - UK motor conventions
areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])

vehicle_group = rng.integers(1, 51, size=n)  # ABI group 1-50
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_flag = rng.binomial(1, 0.06, size=n)
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency parameters (GLM intercept + log-linear effects)
INTERCEPT = -3.10
TRUE_PARAMS = {
    "area_B": 0.10, "area_C": 0.20, "area_D": 0.35,
    "area_E": 0.50, "area_F": 0.65,
    "vehicle_group": 0.018,   # per ABI group unit above 1
    "ncd_years": -0.13,       # per year of NCD
    "young_driver": 0.55,     # age < 25
    "old_driver": 0.28,       # age > 70
    "conviction": 0.42,
}

# Build the log expected claim rate for each policy
log_mu = (
    INTERCEPT
    + np.where(area == "B", TRUE_PARAMS["area_B"], 0)
    + np.where(area == "C", TRUE_PARAMS["area_C"], 0)
    + np.where(area == "D", TRUE_PARAMS["area_D"], 0)
    + np.where(area == "E", TRUE_PARAMS["area_E"], 0)
    + np.where(area == "F", TRUE_PARAMS["area_F"], 0)
    + TRUE_PARAMS["vehicle_group"] * (vehicle_group - 1)
    + TRUE_PARAMS["ncd_years"] * ncd_years
    + np.where(driver_age < 25, TRUE_PARAMS["young_driver"], 0)
    + np.where(driver_age > 70, TRUE_PARAMS["old_driver"], 0)
    + TRUE_PARAMS["conviction"] * conviction_flag
    + np.log(exposure)
)

freq_rate = np.exp(log_mu - np.log(exposure))  # annualised frequency
claim_count = rng.poisson(freq_rate * exposure)

# Severity DGP: Gamma with mean around £3,500, vehicle group effect only.
# NCD reflects driver behaviour and correlates with claim frequency,
# not individual claim size. Including NCD in the severity model would
# capture frequency effects through the back door.
sev_log_mu = (
    np.log(3500)
    + 0.012 * (vehicle_group - 1)
)
true_mean_sev = np.exp(sev_log_mu)
shape_param = 4.0  # coefficient of variation = 1/sqrt(4) = 0.5

has_claim = claim_count > 0
avg_severity = np.where(
    has_claim,
    rng.gamma(shape_param, true_mean_sev / shape_param),
    0.0
)

df = pl.DataFrame({
    "policy_id": np.arange(1, n + 1),
    "area": area,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "driver_age": driver_age,
    "conviction_flag": conviction_flag,
    "exposure": exposure,
    "claim_count": claim_count,
    "avg_severity": avg_severity,
    "incurred": avg_severity * claim_count,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Exposure: {df['exposure'].sum():,.0f} earned years")
print(f"Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum() / df['exposure'].sum():.3f}/year)")
print(f"Total incurred: £{df['incurred'].sum() / 1e6:.1f}m")
```

**What you should see:** Four printed lines showing the portfolio summary. Roughly 88,000 earned years, around 10,000-12,000 claims, total incurred around £40-50m. The exact numbers depend on the random draws but these ranges are correct.

**What the code did:** It generated 100,000 arrays of random numbers (one element per policy), calculated a log-expected-frequency for each policy using the true parameter values, drew actual claim counts from a Poisson distribution with those expected frequencies, and then drew severity amounts from a Gamma distribution for the policies that had claims. The whole thing is stored in a Polars DataFrame called `df`.

### A new Python concept: f-strings

The `f"..."` syntax in the print statements is an f-string. The `f` prefix tells Python to interpret `{...}` inside the string as an expression to evaluate. So `f"{df['claim_count'].sum():,}"` prints the claim count sum with thousand separators. The `:,` inside the braces is a format specifier meaning "use commas as thousand separators." The `:.3f` means "three decimal places."

### Sanity-checking the data

Before fitting any model, look at the data. In a new cell:

```python
# First few rows
df.head(5)
```

You should see a table with 10 columns and 5 rows. Look at the exposure values - they should all be between 0.05 and 1.0. The claim_count should be 0 for most rows (this is a motor book, not a high-frequency line). The avg_severity should be 0.0 for rows with no claims.

In the next cell:

```python
# Summary statistics
df.describe()
```

This shows count, mean, standard deviation, min, and max for each numeric column. The minimum claim_count should be 0, the minimum exposure should be around 0.05. If you see negative values in either column, something went wrong in the generation step.

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

**What you should see:** Area A should have the lowest claim frequency (it is the base area with no uplift). Area F should have the highest, roughly 1.9x area A. This matches `exp(0.65) ≈ 1.92`, the true area F effect we defined in TRUE_PARAMS.

**A new Python concept: method chaining.** Polars lets you chain operations with `.`. Each method returns a new DataFrame, so you can chain `.group_by()`, `.agg()`, `.with_columns()`, `.sort()` one after another. This is the same idea as Excel's CTRL+T tables with multiple computed columns, but written in a chain rather than spread across cells.