## Part 4: Building the synthetic motor dataset

We use the same synthetic UK motor portfolio as Modules 3 and 4. If you saved it to a Delta table, you can read it back. If not, we regenerate it here. The dataset has 100,000 policies with a superadditive interaction between young drivers and high vehicle groups - the same interaction the GBM found in Module 3.

Add a markdown cell to keep the notebook organised. In Databricks, cells starting with `%md` render as formatted text rather than code:

```python
%md
## Part 4: Data preparation
```

Now in a new cell, paste this and run it:

```python
rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors
areas             = ["A", "B", "C", "D", "E", "F"]
area              = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group     = rng.integers(1, 51, size=n)
ncd_years         = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age        = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
annual_mileage    = rng.integers(3_000, 35_000, size=n)
exposure          = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency (log link, multiplicative effects)
INTERCEPT         = -3.10
area_effect       = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.70}
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

log_mu = (
    INTERCEPT
    + np.array([area_effect[a] for a in area])
    + (-0.15) * ncd_years
    + 0.010   * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.55, np.where(driver_age > 70, 0.20, 0.0))
    + np.array([conviction_effect[c] for c in conviction_points])
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)  # interaction
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

# Severity: Gamma-distributed, log link
sev_log_mu = (
    7.80
    + np.array([area_effect[a] * 0.3 for a in area])
    + 0.015 * (vehicle_group - 25)
    + np.array([conviction_effect[c] * 0.2 for c in conviction_points])
)
incurred = np.where(
    claim_count > 0,
    rng.gamma(shape=3.0, scale=np.exp(sev_log_mu) / 3.0, size=n) * claim_count,
    0.0,
)

# Assign accident years (chronological - essential for the temporal split)
accident_year = rng.choice(
    [2019, 2020, 2021, 2022, 2023, 2024],
    size=n,
    p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20],
)

df = pl.DataFrame({
    "accident_year":      accident_year.astype(np.int32),
    "area":               area,
    "vehicle_group":      vehicle_group.astype(np.int32),
    "ncd_years":          ncd_years.astype(np.int32),
    "driver_age":         driver_age.astype(np.int32),
    "conviction_points":  conviction_points.astype(np.int32),
    "annual_mileage":     annual_mileage.astype(np.int32),
    "exposure":           exposure,
    "claim_count":        claim_count.astype(np.int32),
    "incurred":           incurred,
}).with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
).sort("accident_year")

print(f"Dataset: {len(df):,} rows")
print(f"Accident years: {df['accident_year'].min()} - {df['accident_year'].max()}")
print(f"Overall claim frequency: {claim_count.mean():.4f}")
print(f"Mean pure premium: £{incurred.mean():.2f}")
print(f"Zero-claim rows: {(claim_count == 0).mean():.1%}")
df.head(5)
```

**What this does:** generates 100,000 synthetic motor policies with realistic rating factor distributions, a true Poisson frequency model, and Gamma severity. The `incurred / exposure` computation produces the pure premium (loss cost per year of exposure) that we model with Tweedie. The sort by `accident_year` is essential - without it, the temporal split below is meaningless.

**What you should see:**

```
Dataset: 100,000 rows
Accident years: 2019 - 2024
Overall claim frequency: 0.0536
Mean pure premium: £157.xx
Zero-claim rows: 94.x%
```

The exact numbers will match this if you use `seed=42`. If you see a `KeyError` or `NameError`, check that the cell above (the imports) ran successfully first.