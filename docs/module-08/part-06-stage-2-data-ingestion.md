## Part 6: Stage 2 -- Data ingestion

In production, this stage reads from your policy administration system. In this tutorial, we generate synthetic UK motor data so the notebook is self-contained.

The synthetic data generates 200,000 policies across accident years 2021-2024. The data generating process has the same structure as the Module 3 and Module 5 datasets: a Poisson claim count with a log-linear frequency model, log-normal severity with a superadditive interaction between young drivers and high vehicle groups, and realistic exposure distributions.

Add a markdown cell:

```python
%md
## Stage 2: Data ingestion -- generate synthetic motor portfolio
```

Then add this code cell:

```python
import polars as pl
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=2026)

# -----------------------------------------------------------------------
# Portfolio dimensions
# -----------------------------------------------------------------------
N_POLICIES   = 200_000
ACCIDENT_YEARS = [2021, 2022, 2023, 2024]

# -----------------------------------------------------------------------
# Rating factors
# -----------------------------------------------------------------------
ncb_years     = rng.choice([0,1,2,3,4,5], N_POLICIES,
                           p=[0.08, 0.07, 0.10, 0.15, 0.20, 0.40])
vehicle_group = rng.integers(1, 51, N_POLICIES)
age_band      = rng.choice(["17-25","26-35","36-50","51-65","66+"],
                           N_POLICIES, p=[0.11, 0.22, 0.30, 0.25, 0.12])
annual_mileage = rng.choice(["<5k","5k-10k","10k-15k","15k+"],
                             N_POLICIES, p=[0.18, 0.30, 0.35, 0.17])
region        = rng.choice(["North","Midlands","London","SouthEast","SouthWest"],
                           N_POLICIES, p=[0.20, 0.20, 0.22, 0.25, 0.13])
accident_year = rng.choice(ACCIDENT_YEARS, N_POLICIES)

# Exposure: most policies are near-annual, some are mid-term
exposure      = np.clip(rng.beta(8, 2, N_POLICIES), 0.05, 1.0)

# -----------------------------------------------------------------------
# True claim frequency (log-linear, multiplicative)
# -----------------------------------------------------------------------
# We define the true DGP explicitly so we can assess model performance.
# This is the ground truth - the model will recover an approximation of it.

INTERCEPT = -2.95

age_mid = {"17-25": 21, "26-35": 30, "36-50": 43, "51-65": 58, "66+": 72}
age_effect = np.array([
    -0.85 + 0.03 * age_mid[a] - 0.0002 * age_mid[a]**2
    for a in age_band
])

# NCB: strong monotone effect - each year of NCD reduces frequency
ncb_effect = -0.18 * ncb_years

# Vehicle group: positive effect with superadditive interaction for young drivers
vg_effect = 0.012 * vehicle_group

# Young driver x high vehicle group superadditive interaction
is_young  = np.array([1 if a == "17-25" else 0 for a in age_band])
high_vg   = (vehicle_group > 35).astype(float)
interaction_effect = 0.45 * is_young * high_vg

# Region
region_effects = {
    "North": 0.0, "Midlands": 0.08, "London": 0.40,
    "SouthEast": 0.22, "SouthWest": 0.05
}
region_effect = np.array([region_effects[r] for r in region])

# Mileage: more miles, more claims
mileage_effects = {"<5k": -0.10, "5k-10k": 0.0, "10k-15k": 0.12, "15k+": 0.28}
mileage_effect  = np.array([mileage_effects[m] for m in annual_mileage])

# True log-frequency per policy-year (claim count model)
log_freq = (INTERCEPT + age_effect + ncb_effect + vg_effect +
            interaction_effect + region_effect + mileage_effect)

# Expected claim count = frequency * exposure
true_freq       = np.exp(log_freq)
expected_claims = true_freq * exposure
claim_count     = rng.poisson(expected_claims)

# -----------------------------------------------------------------------
# Severity: log-normal with heavier tail for young high-vehicle drivers
# -----------------------------------------------------------------------
sev_log_mean   = 7.1 + 0.008 * vehicle_group + 0.25 * is_young * high_vg
sev_log_sd     = 0.90 + 0.10 * is_young

# Simulate severity per policy (only meaningful where claim_count > 0)
sev_per_claim  = np.exp(rng.normal(sev_log_mean, sev_log_sd, N_POLICIES))
incurred_loss  = claim_count * sev_per_claim  # zero for no-claim policies

print(f"Policies generated:  {N_POLICIES:,}")
print(f"Total claims:        {claim_count.sum():,}")
print(f"Overall claim rate:  {claim_count.sum() / exposure.sum():.4f} per policy-year")
print(f"Total incurred:      £{incurred_loss.sum():,.0f}")
print(f"Mean severity (claims only): £{incurred_loss[claim_count>0].mean():,.0f}")
```

**What you should see:** Around 200,000 policies, 10,000-12,000 total claims (roughly 5.5% claim frequency), and a mean severity in the range of £2,000-£4,000. The exact numbers will depend on the random seed.

Now assemble the Polars DataFrame:

```python
raw_pl = pl.DataFrame({
    "policy_id":      [f"POL{i:07d}" for i in range(N_POLICIES)],
    "accident_year":  accident_year.tolist(),
    "ncb_years":      ncb_years.tolist(),
    "vehicle_group":  vehicle_group.tolist(),
    "age_band":       age_band.tolist(),
    "annual_mileage": annual_mileage.tolist(),
    "region":         region.tolist(),
    "exposure":       exposure.tolist(),
    "claim_count":    claim_count.tolist(),
    "incurred_loss":  incurred_loss.tolist(),
})

# Sanity checks
assert raw_pl.shape[0] == N_POLICIES,         "Row count mismatch"
assert raw_pl["exposure"].min() > 0,           "Zero or negative exposure"
assert raw_pl["claim_count"].min() >= 0,       "Negative claim count"
assert raw_pl["incurred_loss"].min() >= 0,     "Negative incurred loss"
assert raw_pl["ncb_years"].is_between(0, 5).all(), "NCB out of range"

print("Data shape:", raw_pl.shape)
print("\nAccident year distribution:")
print(raw_pl.group_by("accident_year").agg(
    pl.len().alias("n_policies"),
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
).sort("accident_year").with_columns(
    (pl.col("claims") / pl.col("exposure")).round(4).alias("freq")
))
```

### Writing to Delta Lake

Delta Lake is Databricks' table format. It adds three capabilities over standard Parquet files that matter for pricing: ACID transactions (concurrent reads and writes are consistent), time travel (every version of the data is preserved), and DML operations (you can UPDATE, DELETE, and MERGE specific rows without rewriting the entire table).

Write the raw data to Delta:

```python
# Convert Polars to pandas for Spark (Spark cannot read Polars directly)
# We only convert at the library boundary -- all other processing stays in Polars
raw_spark = spark.createDataFrame(raw_pl.to_pandas())

raw_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["raw"])

# Log the Delta table version
raw_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['raw']} LIMIT 1"
).collect()[0]["version"]

print(f"Raw data written to: {TABLES['raw']}")
print(f"Delta version:       {raw_version}")
print(f"Row count:           {raw_spark.count():,}")
```

**What does `DESCRIBE HISTORY` return?** Every write to a Delta table increments its version number. The first write is version 0. A subsequent overwrite is version 1. An append is version 2. The history table records every version, its timestamp, and the operation type. You can read the data at any historical version with `.option("versionAsOf", N)`.

**What does `raw_version` tell us?** This is the version number of the raw data table as it exists right now, at the point this pipeline ran. We log it to the audit record so that anyone reviewing this pipeline six months later can read the exact data that was used:

```python
spark.read.format("delta") \
    .option("versionAsOf", raw_version) \
    .table(TABLES["raw"]) \
    .toPandas()
```

This is Delta time travel. It works as long as the table's VACUUM retention policy preserves the version files. The default Databricks retention is 30 days -- not long enough for a Consumer Duty audit trail. Set at least 365 days on any table that forms part of the pricing basis:

```python
spark.sql(f"""
    ALTER TABLE {TABLES['raw']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
```