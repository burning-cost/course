## Part 3: Recreating the training data

We use the same synthetic UK motor portfolio from Module 3. If you saved it to a Delta table in your workspace, you can read it back. If not, we regenerate it here.

This data generation code is identical to Module 3. The true DGP parameters include a superadditive interaction between young drivers (under 25) and high vehicle groups (above 35) - this is the signal the GBM finds that the GLM misses.

In a new cell, type this and run it (Shift+Enter):

```python
%md
## Part 3: Data preparation
```

Running a cell that starts with `%md` renders it as formatted markdown text in the notebook. This keeps your notebook organised with visible section headings.

Now in a new cell, type this and run it (Shift+Enter):

```python
rng = np.random.default_rng(seed=42)
n = 100_000

areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group = rng.integers(1, 51, size=n)
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

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
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

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
    "area":              area,
    "vehicle_group":     vehicle_group.astype(np.int32),
    "ncd_years":         ncd_years.astype(np.int32),
    "driver_age":        driver_age.astype(np.int32),
    "conviction_points": conviction_points.astype(np.int32),
    "exposure":          exposure,
    "claim_count":       claim_count.astype(np.int32),
    "incurred":          incurred,
})

print(f"Rows:            {len(df):,}")
print(f"Total claims:    {df['claim_count'].sum():,}")
print(f"Total exposure:  {df['exposure'].sum():,.0f} policy-years")
print(f"Claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
df.head(3)
```

The output looks like:

```
Rows:            100,000
Total claims:    4,821
Total exposure:  80,021 policy-years
Claim frequency: 0.0602
```

The exact numbers vary slightly. A claim frequency around 6% on a motor book is realistic for a standard UK personal lines portfolio.

Now add the engineered features. In a new cell, type this and run it (Shift+Enter):

```python
# Feature engineering
df = df.with_columns([
    (pl.col("conviction_points") > 0).cast(pl.Int8).alias("has_convictions"),
    pl.col("annual_mileage").alias("annual_mileage")
    if "annual_mileage" in df.columns
    else pl.lit(None).cast(pl.Int32).alias("annual_mileage"),
])

# Final feature list for the frequency model
FREQ_FEATURES = ["area", "ncd_years", "has_convictions", "vehicle_group", "driver_age"]
CAT_FEATURES  = ["area", "has_convictions"]
CONT_FEATURES = ["ncd_years", "vehicle_group", "driver_age"]

print("Feature lists set:")
print(f"  FREQ_FEATURES: {FREQ_FEATURES}")
print(f"  CAT_FEATURES:  {CAT_FEATURES}")
print(f"  CONT_FEATURES: {CONT_FEATURES}")
```

You will see the feature list printed. No errors means the feature engineering worked.

**Why `conviction_points` becomes `has_convictions`:** The raw conviction points (0, 3, 6, 9) contain ordinal information, but the most important split is clean vs. any conviction. A binary flag is simpler and more interpretable for committee presentation. In practice you would test both encodings.