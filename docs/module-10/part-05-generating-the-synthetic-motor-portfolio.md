## Part 5: Generating the synthetic motor portfolio

We use the same synthetic dataset from Module 2. The data generation code is reproduced here so this module is self-contained. If you still have the Module 2 notebook running, you can skip this step and use the data from there — but it is safer to regenerate.

```python
rng = np.random.default_rng(seed=42)
n   = 100_000

# Rating factors
areas             = ["A", "B", "C", "D", "E", "F"]
area              = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group     = rng.integers(1, 51, size=n)
ncd_years         = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age        = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
annual_mileage    = rng.integers(3_000, 35_000, size=n)
exposure          = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency (with a planted interaction: young driver × high vehicle group)
area_effect       = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.70}
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

log_mu = (
    -3.10
    + np.array([area_effect[a] for a in area])
    + (-0.15) * ncd_years
    + 0.010   * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.55, np.where(driver_age > 70, 0.20, 0.0))
    + np.array([conviction_effect[c] for c in conviction_points])
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)  # planted interaction
    + np.where((ncd_years == 0) & (conviction_points > 0), 0.20, 0.0)  # second planted interaction
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

print(f"Policies:     {n:,}")
print(f"Claim count:  {claim_count.sum():,}")
print(f"Mean freq:    {(claim_count / exposure).mean():.4f}")
```

**What to notice:** the data generation includes two planted interactions:
1. `driver_age < 25` AND `vehicle_group > 35`: a supermultiplicative 0.30 log-unit bump for young drivers in high-group vehicles. This is the interaction the pipeline should detect first.
2. `ncd_years == 0` AND `conviction_points > 0`: zero NCD combined with conviction points adds an extra 0.20 log-unit penalty.

These are exactly the kind of interactions a GLM with main effects alone cannot capture.

### Create the feature DataFrame

```python
# Discretise continuous features as we would in a real pricing model
age_band = np.select(
    [driver_age < 22, driver_age < 26, driver_age < 35, driver_age < 50, driver_age < 70],
    ["17-21", "22-25", "26-34", "35-49", "50-69"],
    default="70+"
)

vg_band = np.select(
    [vehicle_group <= 10, vehicle_group <= 20, vehicle_group <= 30, vehicle_group <= 40],
    ["1-10", "11-20", "21-30", "31-40"],
    default="41-50"
)

mileage_band = np.select(
    [annual_mileage < 8_000, annual_mileage < 15_000, annual_mileage < 25_000],
    ["low", "medium", "high"],
    default="very_high"
)

X = pl.DataFrame({
    "area":              area,
    "vehicle_group":     vg_band,
    "ncd_years":         ncd_years.astype(np.int32),
    "age_band":          age_band,
    "conviction_points": conviction_points.astype(np.int32),
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

**Why we discretise:** GLMs in personal lines pricing work with banded continuous variables. A 70-band vehicle group becomes a 5-band version, reducing parameter count and improving credibility. The interaction detection pipeline works on whatever feature representation you give it. Using the banded version here keeps the tutorial consistent with what you would do in production.