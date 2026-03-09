## Part 4: Regenerating the motor dataset

We use the same 100,000-policy synthetic motor portfolio from Module 2. If you have it saved as a Delta table in your workspace, you can read it back. If not, we regenerate it here.

Create a new cell with a markdown header to keep the notebook organised. In Databricks, a cell starting with `%md` renders as formatted text rather than running Python:

```python
%md
## Part 4: Data preparation
```

Now create the next cell and paste in the data generation code. This is the same DGP (data generating process) as Module 2, so the GLM and GBM are trained on the same underlying truth:

```python
rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors
areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group = rng.integers(1, 51, size=n)
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency: log link, multiplicative effects
INTERCEPT = -3.10
area_effect = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.70}
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

log_mu = (
    INTERCEPT
    + np.array([area_effect[a] for a in area])
    + (-0.15) * ncd_years
    + 0.010  * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.55,
      np.where(driver_age > 70, 0.20, 0.0))
    + np.array([conviction_effect[c] for c in conviction_points])
    # Superadditive interaction: young driver x high vehicle group
    # This is what the GLM misses and the GBM finds
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)
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

df = pl.DataFrame({
    "accident_year":      rng.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n,
                                      p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20]),
    "area":               area,
    "vehicle_group":      vehicle_group.astype(np.int32),
    "ncd_years":          ncd_years.astype(np.int32),
    "driver_age":         driver_age.astype(np.int32),
    "conviction_points":  conviction_points.astype(np.int32),
    "exposure_years":     exposure,
    "claim_count":        claim_count.astype(np.int32),
    "incurred":           incurred,
})

df.head(5)
```

Run this cell. It takes a second or two. The output shows the first five rows of the DataFrame as a table.

What you are looking at: 100,000 motor policies across accident years 2019-2024. Each row is one policy. The `accident_year` column is important - we use it to construct our cross-validation folds later. The true DGP contains a superadditive interaction between young drivers (under 25) and high vehicle groups (above 35): these policies are worse than the multiplicative combination of the youth penalty and the vehicle group effect would suggest. This is the signal that the GLM underestimates and the GBM should find.

Now check some basic counts:

```python
print(f"Total policies: {len(df):,}")
print(f"Total claims:   {df['claim_count'].sum():,}")
print(f"Total exposed:  {df['exposure_years'].sum():,.0f} policy-years")
print(f"Claim frequency:{df['claim_count'].sum() / df['exposure_years'].sum():.4f}")
print()
print("Policies by accident year:")
print(
    df.group_by("accident_year")
    .agg(
        pl.len().alias("policies"),
        pl.col("claim_count").sum().alias("claims"),
        pl.col("exposure_years").sum().alias("exposure"),
    )
    .with_columns((pl.col("claims") / pl.col("exposure")).alias("freq"))
    .sort("accident_year")
)
```

You should see roughly 12,000-20,000 policies per accident year, with a portfolio claim frequency around 5-6%. The frequency should be fairly stable across years - the synthetic DGP has no trend. This is the starting point.