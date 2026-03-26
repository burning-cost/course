# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Databricks for Pricing Teams — Setup Notebook
# MAGIC
# MAGIC Full setup workflow for a pricing team environment. Runs end-to-end on a
# MAGIC single-node Databricks cluster (DBR 14.x LTS ML recommended).
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Creates a Unity Catalog schema for the course
# MAGIC 2. Generates a synthetic UK motor claims dataset
# MAGIC 3. Writes it as a Delta table with enforced schema and partitioning
# MAGIC 4. Demonstrates Delta time travel — querying a prior table version
# MAGIC 5. Shows basic pricing data profiling (claim frequency by area, severity distribution)
# MAGIC 6. Demonstrates incremental updates with MERGE
# MAGIC 7. Sets up a Databricks Workflow using the SDK (requires a paid workspace)
# MAGIC
# MAGIC Runtime: ~15 minutes on a small cluster (4 cores).
# MAGIC
# MAGIC **Prerequisites:** Unity Catalog enabled on your Databricks workspace.
# MAGIC Free Edition users can follow all steps except section 7 (Workflows).

# COMMAND ----------

# MAGIC %pip install databricks-sdk polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date, timedelta
import random

import numpy as np
import polars as pl
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType, BooleanType
)

print("Libraries loaded.")
print(f"Today: {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unity Catalog schema setup
# MAGIC
# MAGIC We create a catalog and schema for the course. In a real environment these would
# MAGIC be created once by a platform admin — you would be granted access, not creating
# MAGIC the catalog itself. We create it here so the course is self-contained.
# MAGIC
# MAGIC **Note:** Most teams do not get a greenfield environment. If your workspace already
# MAGIC has a catalog structure set up by a platform team, update CATALOG and SCHEMA
# MAGIC below to point to what you have been given. You will need at minimum USE CATALOG,
# MAGIC USE SCHEMA, and CREATE TABLE permissions on the target namespace.
# MAGIC
# MAGIC If you already have a catalog and schema you want to use, skip the catalog
# MAGIC creation cell and update CATALOG and SCHEMA below.

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"
TABLE   = "claims_exposure"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

print(f"Target table: {FULL_TABLE}")

# COMMAND ----------

# Create catalog (requires account admin or metastore admin — skip if using an existing catalog)
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'Insurance pricing models and data'")
    print(f"Catalog '{CATALOG}' ready.")
except Exception as e:
    print(f"Could not create catalog (you may not have admin rights): {e}")
    print("If using an existing catalog, update CATALOG above and continue.")
    print("Ask your platform team to grant you USE CATALOG on the appropriate catalog.")

# COMMAND ----------

# Create schema
spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}
    COMMENT 'Motor personal lines pricing: claims, exposure, models, relativities'
""")
print(f"Schema '{CATALOG}.{SCHEMA}' ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate synthetic UK motor data
# MAGIC
# MAGIC We generate a realistic synthetic motor portfolio with known properties.
# MAGIC The dataset mimics a typical UK personal lines motor book:
# MAGIC - 100,000 policy-periods across accident years 2019-2024
# MAGIC - ABI area bands A-F, ABI vehicle groups 1-50
# MAGIC - NCD years 0-5 (skewed towards NCD 4-5, reflecting a mature book)
# MAGIC - Driver age 17-84
# MAGIC - Claim frequency ~10% pa, claim severity 2,000-15,000
# MAGIC
# MAGIC The true DGP parameters are logged below so you can verify extraction later.

# COMMAND ----------

def generate_motor_portfolio(
    n_policies: int = 100_000,
    accident_years: tuple = (2019, 2020, 2021, 2022, 2023, 2024),
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate a synthetic UK personal lines motor portfolio.

    Each row is one policy-period. Exposure is a random float < 1 to simulate
    mid-term cancellations and new business part-years.

    True frequency DGP (Poisson, log link):
        log(freq) = -3.0
                    + 0.10 * (area_B) + 0.20 * (area_C) + 0.35 * (area_D)
                    + 0.50 * (area_E) + 0.65 * (area_F)
                    + (-0.12) * ncd_years
                    + 0.45 * has_convictions
                    + 0.010 * (vehicle_group - 25)
                    + young_driver_effect(driver_age)

    True severity DGP (Gamma, log link):
        log(sev) = 7.8 + 0.008 * vehicle_group + 0.30 * has_convictions

    NCD distribution: skewed towards higher values (reflecting a mature book
    where most policyholders have accumulated NCD). A uniform 0-5 distribution
    would understate the concentration at NCD 5 typical of a real motor book.
    """
    rng = np.random.default_rng(seed)

    rows_per_year = n_policies // len(accident_years)

    records = []
    for ay in accident_years:
        n = rows_per_year + (n_policies % len(accident_years) if ay == accident_years[-1] else 0)

        area_bands = rng.choice(["A", "B", "C", "D", "E", "F"], size=n,
                                p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        # NCD skewed towards 4-5: reflects a mature UK motor book
        ncd_years  = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                                p=[0.08, 0.07, 0.10, 0.15, 0.20, 0.40])
        vehicle_group = rng.integers(1, 51, size=n)
        driver_age = rng.integers(17, 85, size=n)
        conviction_points = rng.choice([0, 0, 0, 0, 0, 3, 6, 9], size=n)
        annual_mileage = rng.integers(2000, 25001, size=n)
        exposure = rng.uniform(0.25, 1.0, size=n)

        has_convictions = (conviction_points > 0).astype(int)

        # Log-linear frequency parameters
        area_effect = {
            "A": 0.00, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.65
        }
        log_mu = (
            -3.0
            + np.array([area_effect[a] for a in area_bands])
            + (-0.12) * ncd_years
            + 0.45 * has_convictions
            + 0.010 * (vehicle_group - 25)
            # Young driver U-shape: ages 17-24 elevated, 70+ mildly elevated
            + np.where(driver_age < 25, 0.55 * (25 - driver_age) / 8, 0.0)
            + np.where(driver_age > 70, 0.30 * (driver_age - 70) / 14, 0.0)
        )
        mu = np.exp(log_mu) * exposure
        claim_count = rng.poisson(mu).astype(int)

        # Gamma severity
        log_sev = (
            7.8
            + 0.008 * (vehicle_group - 25)
            + 0.30 * has_convictions
        )
        mean_sev = np.exp(log_sev)
        # Gamma with shape=2 (moderate dispersion)
        sev_per_claim = rng.gamma(shape=2.0, scale=mean_sev / 2.0, size=n)
        incurred = np.where(claim_count > 0, claim_count * sev_per_claim, 0.0)

        # Policy dates
        base_date = date(ay, 1, 1)
        day_offsets = rng.integers(0, 365, size=n)
        policy_starts = [base_date + timedelta(days=int(d)) for d in day_offsets]
        policy_ends   = [s + timedelta(days=int(e * 365)) for s, e in zip(policy_starts, exposure)]

        policy_ids = [f"POL{ay}{str(i).zfill(6)}" for i in range(n)]

        for i in range(n):
            # Clip birth year to avoid pre-1900 dates for very old drivers in early accident years
            birth_year = max(ay - int(driver_age[i]), 1900)
            records.append({
                "policy_id":        policy_ids[i],
                "accident_year":    ay,
                "policy_start":     policy_starts[i],
                "policy_end":       policy_ends[i],
                "exposure_years":   round(float(exposure[i]), 4),
                "claim_count":      int(claim_count[i]),
                "incurred":         round(float(incurred[i]), 2),
                "area_band":        area_bands[i],
                "ncd_years":        int(ncd_years[i]),
                "vehicle_group":    int(vehicle_group[i]),
                "driver_age":       int(driver_age[i]),
                "annual_mileage":   float(annual_mileage[i]),
                "conviction_points":int(conviction_points[i]),
                "policyholder_name": f"Policyholder_{policy_ids[i]}",
                "policyholder_dob":  date(birth_year, 6, 15),
            })

    return pl.DataFrame(records)


print("Generating synthetic portfolio (100,000 policy-periods)...")
portfolio = generate_motor_portfolio(n_policies=100_000, seed=42)

print(f"Portfolio: {len(portfolio):,} policy-periods")
print(f"Accident years: {sorted(portfolio['accident_year'].unique().to_list())}")
print(f"Exposure: {portfolio['exposure_years'].sum():.0f} earned years")
print(f"Claims: {portfolio['claim_count'].sum():,} ({portfolio['claim_count'].sum() / portfolio['exposure_years'].sum():.3f} per earned year)")
print(f"Total incurred: {portfolio['incurred'].sum() / 1e6:.1f}m")
print()
print("True DGP parameters:")
print("  Frequency (Poisson, log link):")
print("    Intercept: -3.0 (base rate is exp(-3.0) = 5.0% per unit exposure at area A, NCD=0, no convictions)")
print("    area_B: +0.10, area_C: +0.20, area_D: +0.35, area_E: +0.50, area_F: +0.65")
print("    ncd_years: -0.12 per year (NCD=5 vs NCD=0 = exp(-0.60) = 0.549)")
print("    has_convictions: +0.45 (exp(0.45) = 1.568)")
print("    vehicle_group: +0.010 per group above 25")
print("    young driver (<25): strong positive, elderly (>70): mild positive")
print("  Severity (Gamma, log link):")
print("    Intercept: 7.8 (base severity ~2,440)")
print("    vehicle_group: +0.008 per group above 25")
print("    has_convictions: +0.30 (exp(0.30) = 1.350)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write to Delta with enforced schema
# MAGIC
# MAGIC We define the schema explicitly rather than inferring it. Every production data
# MAGIC load should do this.
# MAGIC
# MAGIC Note: delta.autoOptimize.optimizeWrite was deprecated in DBR 11.3 and has no
# MAGIC effect in DBR 14.x. Do not include it in new table definitions — the runtime
# MAGIC handles write optimisation automatically.

# COMMAND ----------

claims_schema = StructType([
    StructField("policy_id",          StringType(),  nullable=False),
    StructField("accident_year",      IntegerType(), nullable=False),
    StructField("policy_start",       DateType(),    nullable=True),
    StructField("policy_end",         DateType(),    nullable=True),
    StructField("exposure_years",     DoubleType(),  nullable=False),
    StructField("claim_count",        IntegerType(), nullable=False),
    StructField("incurred",           DoubleType(),  nullable=False),
    StructField("area_band",          StringType(),  nullable=True),
    StructField("ncd_years",          IntegerType(), nullable=True),
    StructField("vehicle_group",      IntegerType(), nullable=True),
    StructField("driver_age",         IntegerType(), nullable=True),
    StructField("annual_mileage",     DoubleType(),  nullable=True),
    StructField("conviction_points",  IntegerType(), nullable=True),
    StructField("policyholder_name",  StringType(),  nullable=True),
    StructField("policyholder_dob",   DateType(),    nullable=True),
])

# COMMAND ----------

# Create table with explicit DDL (schema + partitioning + properties)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
        policy_id          STRING  NOT NULL,
        accident_year      INT     NOT NULL,
        policy_start       DATE,
        policy_end         DATE,
        exposure_years     DOUBLE  NOT NULL,
        claim_count        INT     NOT NULL,
        incurred           DOUBLE  NOT NULL,
        area_band          STRING,
        ncd_years          INT,
        vehicle_group      INT,
        driver_age         INT,
        annual_mileage     DOUBLE,
        conviction_points  INT,
        policyholder_name  STRING,
        policyholder_dob   DATE
    )
    USING DELTA
    PARTITIONED BY (accident_year)
    TBLPROPERTIES (
        'delta.logRetentionDuration' = 'interval 7 years',
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Motor personal lines claims and exposure. One row per policy-period. Partitioned by accident_year.'
""")
# delta.logRetentionDuration is set to 7 years - common practice that exceeds
# the FCA's SYSC 9.1.1R minimum (5 years) and aligns with HMRC requirements.
# See the tutorial for context on the interaction with GDPR right to erasure.
print(f"Table {FULL_TABLE} created (or already exists).")

# COMMAND ----------

# Write the 2019-2022 subset first (we will simulate adding 2023-2024 later)
portfolio_initial = portfolio.filter(pl.col("accident_year") <= 2022)

# Bridge: Polars -> pandas -> Spark. PySpark does not yet natively accept Polars DataFrames.
spark_df = spark.createDataFrame(portfolio_initial.to_pandas(), schema=claims_schema)

(
    spark_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("accident_year")
    .saveAsTable(FULL_TABLE)
)

print(f"Initial load complete: {len(portfolio_initial):,} rows (accident years 2019-2022)")
print()

spark.sql(f"DESCRIBE HISTORY {FULL_TABLE}").select(
    "version", "timestamp", "operation", "operationMetrics"
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Delta time travel
# MAGIC
# MAGIC Every write to a Delta table creates a new version. We can query any historical
# MAGIC version — essential for reproducing a model run against the exact data snapshot
# MAGIC it used.
# MAGIC
# MAGIC We will:
# MAGIC 1. Record the current version number
# MAGIC 2. Add 2023-2024 data (simulating a new bordereaux)
# MAGIC 3. Query the old version to confirm time travel works

# COMMAND ----------

# Record the version before the next write
history = spark.sql(f"DESCRIBE HISTORY {FULL_TABLE} LIMIT 1").collect()
version_before_update = history[0]["version"]
timestamp_before_update = history[0]["timestamp"]

print(f"Version before update: {version_before_update}")
print(f"Timestamp: {timestamp_before_update}")

# COMMAND ----------

# Add 2023-2024 data
portfolio_update = portfolio.filter(pl.col("accident_year") >= 2023)

(
    spark.createDataFrame(portfolio_update.to_pandas(), schema=claims_schema)
    .write
    .format("delta")
    .mode("append")
    .saveAsTable(FULL_TABLE)
)

print(f"Update complete: added {len(portfolio_update):,} rows (accident years 2023-2024)")
print(f"Table now has {spark.table(FULL_TABLE).count():,} rows")

# COMMAND ----------

print("Full table history:")
spark.sql(f"DESCRIBE HISTORY {FULL_TABLE}").select(
    "version", "timestamp", "operation",
    F.col("operationMetrics")["numOutputRows"].alias("numOutputRows"),
).show(10, truncate=False)

# COMMAND ----------

# Query current version via Spark, convert to Polars for analysis
df_current = pl.from_pandas(spark.table(FULL_TABLE).toPandas())
print(f"Current version: {len(df_current):,} rows, accident years: "
      f"{sorted(df_current['accident_year'].unique().to_list())}")

# Query the version before the 2023-2024 update
df_historical = pl.from_pandas(
    spark.read
    .format("delta")
    .option("versionAsOf", version_before_update)
    .table(FULL_TABLE)
    .toPandas()
)
print(f"Version {version_before_update}: {len(df_historical):,} rows, accident years: "
      f"{sorted(df_historical['accident_year'].unique().to_list())}")

# COMMAND ----------

print("In a modelling notebook, log the table version used for training:")
print(f"""
    audit = {{
        'training_table': '{FULL_TABLE}',
        'training_table_version': {version_before_update},
        'training_accident_years': '2019-2022',
        'run_date': '{date.today()}',
    }}
    # Write to pricing.governance.model_run_log
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Incremental updates with MERGE
# MAGIC
# MAGIC Month-end bordereaux typically contain updated development on prior accident years
# MAGIC (late-reported claims, reserve movements) as well as new business. MERGE applies
# MAGIC only the changed rows atomically.
# MAGIC
# MAGIC The correct pattern:
# MAGIC 1. Create the update DataFrame
# MAGIC 2. Register it as a temp view — separate statement, because createOrReplaceTempView returns None
# MAGIC 3. Call spark.sql() with the MERGE SQL, referring to the view by name

# COMMAND ----------

rng = np.random.default_rng(999)

# Pick 500 existing 2023 policies to update (simulate late claim development)
existing_ids_spark = (
    spark.table(FULL_TABLE)
    .filter(F.col("accident_year") == 2023)
    .limit(500)
    .select("policy_id", "accident_year", "claim_count", "incurred")
)
existing_ids = pl.from_pandas(existing_ids_spark.toPandas())
factors = pl.Series("factor", rng.uniform(1.02, 1.15, size=len(existing_ids)))
existing_ids = existing_ids.with_columns(
    (pl.col("incurred") * factors).round(2).alias("incurred")
)

# 200 new policies
new_policies = generate_motor_portfolio(n_policies=200, accident_years=(2024,), seed=888)

# Pull matching rows from the full portfolio and apply updated incurred
updated_existing = (
    portfolio
    .filter(pl.col("policy_id").is_in(existing_ids["policy_id"]))
    .join(existing_ids.select(["policy_id", "incurred"]), on="policy_id", how="left", suffix="_new")
    .with_columns(pl.col("incurred_new").alias("incurred"))
    .drop("incurred_new")
)

bordereaux = pl.concat([updated_existing, new_policies])
bordereaux_spark = spark.createDataFrame(bordereaux.to_pandas(), schema=claims_schema)

# Step 1: register temp view (separate statement — createOrReplaceTempView returns None)
bordereaux_spark.createOrReplaceTempView("bordereaux_updates")

# Step 2: MERGE SQL refers to the view by name
spark.sql(f"""
    MERGE INTO {FULL_TABLE} AS target
    USING bordereaux_updates AS source
    ON target.policy_id = source.policy_id
       AND target.accident_year = source.accident_year
    WHEN MATCHED THEN
        UPDATE SET
            target.claim_count    = source.claim_count,
            target.incurred       = source.incurred,
            target.exposure_years = source.exposure_years
    WHEN NOT MATCHED THEN
        INSERT *
""")

print(f"MERGE complete.")
print(f"Table now has {spark.table(FULL_TABLE).count():,} rows")
print()
spark.sql(f"DESCRIBE HISTORY {FULL_TABLE}").select(
    "version", "operation", "operationMetrics"
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data profiling — the basics a pricing team needs
# MAGIC
# MAGIC Before fitting any model, run these checks. They catch the data issues that
# MAGIC produce wrong models rather than error messages.

# COMMAND ----------

df = pl.from_pandas(spark.table(FULL_TABLE).toPandas())

print("=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)
print()

ay_summary = (
    df
    .group_by("accident_year")
    .agg([
        pl.col("policy_id").count().alias("n_policies"),
        pl.col("exposure_years").sum().alias("exposure"),
        pl.col("claim_count").sum().alias("claims"),
        pl.col("incurred").sum().alias("incurred"),
    ])
    .sort("accident_year")
    .with_columns([
        (pl.col("claims") / pl.col("exposure")).alias("claim_freq"),
        (pl.col("incurred") / pl.col("claims").clip(lower_bound=1)).alias("avg_severity"),
    ])
)

print("Policy-periods by accident year:")
print(ay_summary)

# COMMAND ----------

import scipy.stats as stats

area_summary = (
    df
    .group_by("area_band")
    .agg([
        pl.col("exposure_years").sum().alias("exposure"),
        pl.col("claim_count").sum().alias("claims"),
    ])
    .sort("area_band")
    .with_columns(
        (pl.col("claims") / pl.col("exposure")).alias("freq")
    )
)

alpha = 0.05
print()
print("Claim frequency by area band (with 95% Poisson CI):")
for row in area_summary.to_dicts():
    lower = stats.chi2.ppf(alpha / 2, 2 * row["claims"]) / (2 * row["exposure"])
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * (row["claims"] + 1)) / (2 * row["exposure"])
    print(f"  {row['area_band']}: freq={row['freq']:.4f}  95% CI [{lower:.4f}, {upper:.4f}]  "
          f"(n_claims={row['claims']:,}, exposure={row['exposure']:,.0f})")

# COMMAND ----------

claims_only = df.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_sev")
)

print()
print("Severity distribution (claims-only policies):")
print(f"  n policies with claims: {len(claims_only):,}")
print(f"  Mean severity:          {claims_only['avg_sev'].mean():,.0f}")
print(f"  Median severity:        {claims_only['avg_sev'].median():,.0f}")
print(f"  75th percentile:        {claims_only['avg_sev'].quantile(0.75):,.0f}")
print(f"  95th percentile:        {claims_only['avg_sev'].quantile(0.95):,.0f}")
print(f"  99th percentile:        {claims_only['avg_sev'].quantile(0.99):,.0f}")
print(f"  Max severity:           {claims_only['avg_sev'].max():,.0f}")

# COMMAND ----------

print()
print("NCD year distribution:")
ncd_dist = (
    df
    .group_by("ncd_years")
    .agg(pl.count().alias("n"))
    .sort("ncd_years")
)
print(ncd_dist)
print()

n_negative_exposure = (df["exposure_years"] <= 0).sum()
n_negative_incurred = (df["incurred"] < 0).sum()
n_missing_area = df["area_band"].is_null().sum()
n_extreme_sev = (claims_only["avg_sev"] > 50_000).sum()

print("Data quality flags:")
print(f"  Negative or zero exposure:     {n_negative_exposure}")
print(f"  Negative incurred:             {n_negative_incurred}")
print(f"  Missing area band:             {n_missing_area}")
print(f"  Severity > 50,000:             {n_extreme_sev} ({100 * n_extreme_sev / max(len(claims_only), 1):.2f}% of claims)")

if any([n_negative_exposure > 0, n_negative_incurred > 0]):
    print()
    print("WARNING: data quality issues found. Investigate before modelling.")
else:
    print()
    print("All data quality checks passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Databricks Workflows — automated scheduling
# MAGIC
# MAGIC This section creates a Databricks Workflow that runs the data quality check
# MAGIC notebook on a daily schedule. Requires a paid Databricks workspace with the
# MAGIC Jobs API enabled.
# MAGIC
# MAGIC Skip this section if using Free Edition.
# MAGIC
# MAGIC In production this pattern would schedule the full modelling pipeline:
# MAGIC data_prep -> train -> extract_relativities -> export. We show the simpler
# MAGIC single-task version here to demonstrate the SDK without the full pipeline.

# COMMAND ----------

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.jobs import (
        JobSettings, Task, NotebookTask, JobEmailNotifications, CronSchedule,
        JobCluster, ClusterSpec,
    )

    w = WorkspaceClient()
    current_user = w.current_user.me()
    print(f"Workspace: {w.config.host}")
    print(f"Current user: {current_user.user_name}")
    WORKFLOWS_AVAILABLE = True
except Exception as e:
    print(f"Workflows API not available: {e}")
    print("This is expected in Free Edition. Skip to the JSON example below.")
    WORKFLOWS_AVAILABLE = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. Workflow definition as code
# MAGIC
# MAGIC Define the Workflow in Python using the SDK, commit the definition file to
# MAGIC your Git repo alongside the notebooks it runs. A Workflow that exists only
# MAGIC in the UI is not auditable and not reproducible.

# COMMAND ----------

WORKFLOW_DEFINITION = {
    "name": "motor_pricing_daily_dq",
    "schedule": {
        "quartz_cron_expression": "0 0 7 * * ?",
        "timezone_id": "Europe/London",
    },
    "tasks": [
        {
            "task_key": "data_quality_check",
            "description": "Validate claims data quality, write results to Delta",
            "notebook_task": {
                "notebook_path": "/Repos/pricing-team/motor-pricing/notebooks/data_quality_check",
                "base_parameters": {
                    "target_table": FULL_TABLE,
                    "output_table": f"{CATALOG}.{SCHEMA}.dq_monitoring",
                    "alert_threshold_pct": "5.0",
                },
            },
        }
    ],
    "email_notifications": {
        "on_failure": ["pricing-team@yourinsurer.co.uk"],
    },
    "max_concurrent_runs": 1,
}

print("Workflow definition (as would be committed to the repo):")
print(json.dumps(WORKFLOW_DEFINITION, indent=2))

# COMMAND ----------

if WORKFLOWS_AVAILABLE:
    from databricks.sdk.service import jobs as jobs_sdk

    try:
        new_job = w.jobs.create(
            settings=jobs_sdk.JobSettings.from_dict(WORKFLOW_DEFINITION)
        )
        print(f"Workflow created: job_id = {new_job.job_id}")
        print(f"View at: {w.config.host}#job/{new_job.job_id}")
    except Exception as e:
        print(f"Could not create workflow: {e}")
        print("Check that you have Jobs admin or Creator permission.")
else:
    print("Workflows API not available — showing example output only.")
    print()
    print("Cron '0 0 7 * * ?' means every day at 07:00 Europe/London.")
    print("The job would run data quality checks before the pricing team starts work.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write model run audit record
# MAGIC
# MAGIC Every time a model pipeline runs, it should write a structured record to the
# MAGIC governance schema. This is the FCA audit trail in practice.

# COMMAND ----------

current_version = spark.sql(
    f"DESCRIBE HISTORY {FULL_TABLE} LIMIT 1"
).collect()[0]["version"]

audit_record = {
    "run_date":              str(date.today()),
    "run_type":              "module_01_setup_notebook",
    "table_written":         FULL_TABLE,
    "source_table":          FULL_TABLE,
    "source_table_version":  int(current_version),
    "n_rows":                len(df),
    "n_claims":              int(df["claim_count"].sum()),
    "total_exposure":        round(float(df["exposure_years"].sum()), 2),
    "accident_years":        str(sorted(df["accident_year"].unique().to_list())),
    "dq_negative_exposure":  int((df["exposure_years"] <= 0).sum()),
    "dq_negative_incurred":  int((df["incurred"] < 0).sum()),
    "dq_missing_area":       int(df["area_band"].is_null().sum()),
    "notes":                 "Initial load and setup for Module 1 course notebook",
}

audit_schema_ddl = f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.governance.model_run_log (
        run_date             STRING,
        run_type             STRING,
        table_written        STRING,
        source_table         STRING,
        source_table_version BIGINT,
        n_rows               BIGINT,
        n_claims             BIGINT,
        total_exposure       DOUBLE,
        accident_years       STRING,
        dq_negative_exposure INT,
        dq_negative_incurred INT,
        dq_missing_area      INT,
        notes                STRING
    )
    USING DELTA
    COMMENT 'Model and pipeline run log for FCA audit trail'
"""

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.governance")
    spark.sql(audit_schema_ddl)

    audit_df = spark.createDataFrame([audit_record])
    audit_df.write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.governance.model_run_log"
    )
    print(f"Audit record written to {CATALOG}.governance.model_run_log")
    print(f"  Training data version: {current_version}")
    print(f"  n_rows: {audit_record['n_rows']:,}")
    print(f"  n_claims: {audit_record['n_claims']:,}")
except Exception as e:
    print(f"Could not write audit record (governance schema may not exist): {e}")
    print("In a real environment, create the governance schema first with admin access.")
    print(f"Audit record that would have been written:")
    print(json.dumps(audit_record, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What we have built in this notebook:
# MAGIC
# MAGIC | Artefact | Location |
# MAGIC |----------|---------|
# MAGIC | Motor claims Delta table | pricing.motor.claims_exposure |
# MAGIC | Table partitioned by accident_year | --- |
# MAGIC | 7-year version history retention (exceeds FCA minimum (SYSC 9.1.1R: 5 years)) | TBLPROPERTIES |
# MAGIC | Time travel demonstrated | Version 0 vs current |
# MAGIC | MERGE-based incremental update | Demonstrated |
# MAGIC | Data quality checks | Run inline, ready to schedule |
# MAGIC | Workflow definition | WORKFLOW_DEFINITION dict, commit to repo |
# MAGIC | Audit log | pricing.governance.model_run_log |
# MAGIC
# MAGIC The next step is to train a frequency model on this data in Module 2.
# MAGIC The training notebook will read from pricing.motor.claims_exposure,
# MAGIC log the table version it used, and write model outputs back to
# MAGIC pricing.motor.freq_model_relativities.

# COMMAND ----------

print("=" * 60)
print("MODULE 1 SETUP COMPLETE")
print("=" * 60)
print()
print(f"Target table: {FULL_TABLE}")
print(f"Current version: {current_version}")
print(f"Rows: {len(df):,}")
print(f"Claim frequency (all years): {df['claim_count'].sum() / df['exposure_years'].sum():.4f}")
print()
print("Next: Module 2 — GLMs for Frequency and Severity")
print("Training notebook reads from this table and writes relativities back to Unity Catalog.")
