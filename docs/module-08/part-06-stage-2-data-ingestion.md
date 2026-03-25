## Part 6: Stage 2 — Data ingestion

In production this stage reads from your policy administration system. Here we generate synthetic UK motor data with a realistic data-generating process: four accident years, premium inflation, separate frequency and severity components, and named categorical features.

The synthetic data has 200,000 policies rather than the 100,000 used in earlier modules. Walk-forward CV splits by accident year; with 200,000 policies across four years, each fold's training set has enough claims to fit stable models and enough validation policies to produce reliable metrics.

Add a markdown cell:

```python
%md
## Stage 2: Data ingestion
```

Then:

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(2026)

YEARS      = [2022, 2023, 2024, 2025]
N_PER_YEAR = N_POLICIES // len(YEARS)   # 50,000 per year

cohorts = []
for year in YEARS:
    inflation = 1.07 ** (year - 2022)   # 7% annual claims inflation
    n = N_PER_YEAR

    # Risk factors
    age_band      = rng.choice(["17-25","26-35","36-50","51-65","66+"], n,
                                p=[0.10, 0.20, 0.35, 0.25, 0.10])
    ncb           = rng.choice([0, 1, 2, 3, 4, 5], n,
                                p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    vehicle_group = rng.choice(["A","B","C","D","E"], n,
                                p=[0.20, 0.25, 0.25, 0.20, 0.10])
    region        = rng.choice(["London","SouthEast","Midlands","North","Scotland","Wales"], n,
                                p=[0.18, 0.20, 0.22, 0.25, 0.10, 0.05])
    annual_mileage = rng.choice(["<5k","5k-10k","10k-15k","15k+"], n,
                                 p=[0.15, 0.35, 0.35, 0.15])

    # Frequency DGP: age as dominant factor
    age_freq = {"17-25": 0.12, "26-35": 0.07, "36-50": 0.05, "51-65": 0.04, "66+": 0.06}
    freq = np.array([age_freq[a] for a in age_band])
    freq *= np.array([{"A":0.85,"B":0.95,"C":1.00,"D":1.10,"E":1.25}[v] for v in vehicle_group])
    freq *= np.array([{"London":1.15,"SouthEast":1.05,"Midlands":1.00,
                       "North":0.95,"Scotland":0.90,"Wales":0.92}[r] for r in region])
    freq *= np.array([{"<5k":0.75,"5k-10k":0.90,"10k-15k":1.05,"15k+":1.30}[m]
                       for m in annual_mileage])
    claim_count = rng.poisson(freq)

    # Severity DGP: vehicle group and region drive claim size
    sev_base = 2_800 * inflation
    mean_sev = (
        sev_base
        * np.array([{"A":0.75,"B":0.90,"C":1.00,"D":1.15,"E":1.40}[v] for v in vehicle_group])
        * np.array([{"London":1.20,"SouthEast":1.10,"Midlands":1.00,
                     "North":0.95,"Scotland":0.88,"Wales":0.92}[r] for r in region])
    )
    # Gamma(2, mean/2): mean = 2 * mean/2 = mean, variance = 2 * (mean/2)^2
    incurred_loss = np.where(claim_count > 0,
                             rng.gamma(2.0, mean_sev / 2.0, n), 0.0)

    exposure       = rng.uniform(0.3, 1.0, n)
    earned_premium = sev_base * freq / LR_TARGET * inflation * rng.uniform(0.94, 1.06, n)

    cohorts.append(pl.DataFrame({
        "policy_id":      [f"{year}-{i:06d}" for i in range(n)],
        "accident_year":  [year] * n,
        "age_band":       age_band.tolist(),
        "ncb":            ncb.tolist(),
        "vehicle_group":  vehicle_group.tolist(),
        "region":         region.tolist(),
        "annual_mileage": annual_mileage.tolist(),
        "exposure":       exposure.tolist(),
        "earned_premium": earned_premium.tolist(),
        "claim_count":    claim_count.tolist(),
        "incurred_loss":  incurred_loss.tolist(),
    }))

raw_pl = pl.concat(cohorts)

print(f"Total policies:    {len(raw_pl):,}")
print(f"Total claims:      {raw_pl['claim_count'].sum():,}")
print(f"Portfolio freq:    {raw_pl['claim_count'].sum() / raw_pl['exposure'].sum():.4f}")
print(f"Mean severity:     £{raw_pl.filter(pl.col('incurred_loss') > 0)['incurred_loss'].mean():,.0f}")
print(f"Portfolio LR:      {raw_pl['incurred_loss'].sum() / raw_pl['earned_premium'].sum():.3f}")
print()
print(raw_pl.group_by("accident_year").agg(
    pl.len().alias("policies"),
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().round(0).alias("exposure"),
    (pl.col("claim_count").sum() / pl.col("exposure").sum()).round(4).alias("freq"),
).sort("accident_year"))
```

**What you should see:** 200,000 policies with a portfolio frequency of approximately 0.06-0.07. Frequency is stable across years (same underlying DGP). Mean severity increases each year at roughly 7% — the inflation loading. Portfolio LR should be close to 0.72, the target built into the earned premium formula.

### Writing to Delta Lake

```python
(
    spark.createDataFrame(raw_pl.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["raw"])
)

# Set retention policy before logging the version
spark.sql(f"""
    ALTER TABLE {TABLES['raw']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

raw_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['raw']} LIMIT 1"
).collect()[0]["version"]

print(f"Written to:    {TABLES['raw']}")
print(f"Delta version: {raw_version}")
print(f"Row count:     {raw_pl.shape[0]:,}")
```

**Why 365-day retention?** Delta's default VACUUM policy removes version files after 30 days. The FCA's Consumer Duty requires that you can demonstrate your pricing basis for any policy in the last three years. 30-day retention means you lose the ability to reproduce any pipeline run more than a month old. Set retention to 365 days minimum on every table that forms part of the pricing basis. We do this at ingestion time, before the version number is logged to the audit record, so the version is guaranteed to survive.

**Why convert Polars to pandas for Spark?** Databricks Spark cannot read Polars DataFrames directly — only pandas or Arrow. The conversion is confined to the Delta write call. All other processing in the pipeline stays in Polars. Polars is consistently faster for in-memory tabular operations and avoids pandas' indexing gotchas that have caused bugs in earlier iterations of this pipeline.
