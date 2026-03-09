## Part 6: Stage 2 -- Data ingestion

In production, this stage reads from your policy administration system. In this tutorial, we load synthetic UK motor data from the `insurance-datasets` library — the same dataset used across all course modules.

We use 200,000 policies here (versus 100,000 in earlier modules) to simulate a larger book and give the pipeline more data to work with for walk-forward cross-validation. The portfolio spans accident years 2019-2023 with realistic exposure distributions.

Add a markdown cell:

```python
%md
## Stage 2: Data ingestion -- load synthetic motor portfolio
```

Then add this code cell:

```python
from insurance_datasets import load_motor
import polars as pl
import numpy as np
import pandas as pd

# Load 200,000-policy portfolio for the pipeline
# Same DGP as Modules 2-5 — column names and distributions are identical.
# load_motor() columns: policy_id, age, vehicle_age, vehicle_group, region,
#                       credit_score, exposure, claim_count, claim_amount
raw_pl = pl.from_pandas(load_motor(n_policies=200_000, seed=42))

# Add synthetic accident years — load_motor() does not include accident_year.
# We assign years 2019-2023 with realistic weighting to enable temporal splits.
rng_year = np.random.default_rng(seed=42)
accident_year = rng_year.choice([2019, 2020, 2021, 2022, 2023], size=len(raw_pl),
                                 p=[0.14, 0.16, 0.18, 0.22, 0.30])
raw_pl = raw_pl.with_columns(
    pl.Series("accident_year", accident_year.astype(np.int32))
)

# Sanity checks on the loaded data
assert raw_pl.shape[0] == 200_000,       "Row count mismatch"
assert raw_pl["exposure"].min() > 0,     "Zero or negative exposure"
assert raw_pl["claim_count"].min() >= 0, "Negative claim count"
assert raw_pl["claim_amount"].min() >= 0, "Negative claim amount"

print(f"Policies loaded:    {raw_pl.shape[0]:,}")
print(f"Total claims:       {raw_pl['claim_count'].sum():,}")
print(f"Overall claim rate: {raw_pl['claim_count'].sum() / raw_pl['exposure'].sum():.4f} per policy-year")
print(f"Total claim amount: £{raw_pl['claim_amount'].sum():,.0f}")

print("\nData shape:", raw_pl.shape)
print("\nAccident year distribution:")
print(raw_pl.group_by("accident_year").agg(
    pl.len().alias("n_policies"),
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
).sort("accident_year").with_columns(
    (pl.col("claims") / pl.col("exposure")).round(4).alias("freq")
))
```

**What you should see:** 200,000 policies, around 15,000 total claims (roughly 7-8% claim frequency), and total claim amount in the £50-80m range. The accident year distribution spans 2019-2023.

**Why 200,000 policies here:** Walk-forward cross-validation splits the data by accident year. With 5 years of data and training sets that grow from 1 to 4 years, a 200,000-policy book gives each fold enough claims to fit stable frequency and severity models. With only 100,000 policies, the early folds (1 year of training data, ~40,000 policies) would be too thin for reliable hyperparameter tuning.

**Why the same library across all modules:** The column names, distributions, and DGP parameters are identical to Modules 2-5. A pricing team would not rebuild the synthetic portfolio for every model iteration. Consistent data means the pipeline's outputs are directly comparable to what earlier modules produced.

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
