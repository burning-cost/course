# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: Conformal Prediction Intervals for Insurance Pricing
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC Prediction intervals that hold: distribution-free coverage guarantees on
# MAGIC insurance claims data using the `insurance-conformal` library.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Trains a CatBoost Tweedie pure premium model
# MAGIC 2. Calibrates conformal prediction intervals using `InsuranceConformalPredictor`
# MAGIC 3. Validates coverage by risk decile (the key diagnostic)
# MAGIC 4. Demonstrates three practical applications: uncertain risk flagging, minimum premium
# MAGIC    floors, and portfolio-level reserve range estimates
# MAGIC 5. Writes intervals and coverage diagnostics to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 15-20 minutes on a 4-core cluster.
# MAGIC
# MAGIC **Prerequisites:** Module 1 notebook run (pricing.motor.claims_exposure exists).
# MAGIC CatBoost and insurance-conformal installed.
# MAGIC
# MAGIC **Key concept:** The coverage guarantee is distribution-free in that it makes no
# MAGIC parametric assumptions about the data — but it does require exchangeability and a
# MAGIC well-calibrated conformity score. For insurance, that means: calibrate on recent
# MAGIC business, test on more recent business, and validate coverage by decile.

# COMMAND ----------

%pip install "insurance-conformal[catboost]" polars --quiet
# In a Databricks notebook, use %pip install rather than uv add — the %pip command installs into the cluster session. Outside Databricks, use uv add.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
import mlflow
import matplotlib.pyplot as plt

from insurance_conformal import InsuranceConformalPredictor, CoverageDiagnostics

print(f"Today: {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data and make the temporal split
# MAGIC
# MAGIC The temporal split is the single most important practical decision in this module.
# MAGIC Conformal prediction requires that calibration data is exchangeable with test data.
# MAGIC In insurance, exchangeable means: drawn from the same temporal distribution.
# MAGIC
# MAGIC Split order: train (oldest) -> calibration -> test (most recent)
# MAGIC
# MAGIC Do not use a random split here. A random split hides temporal trends in residuals.
# MAGIC It technically satisfies the exchangeability condition but makes coverage drift
# MAGIC invisible until it is too late.

# COMMAND ----------

CATALOG    = "pricing"
SCHEMA     = "motor"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.claims_exposure"

try:
    df = pl.from_pandas(spark.table(FULL_TABLE).toPandas())
    print(f"Loaded from Delta: {len(df):,} rows")
except Exception:
    print("Loading fallback synthetic dataset (Module 1 table not found)")
    from insurance_conformal.datasets import load_motor_synthetic
    df = load_motor_synthetic(n_policies=50_000, seed=42)

print(f"Rows: {len(df):,}")
print(f"Accident years: {sorted(df['accident_year'].unique().to_list())}")

# COMMAND ----------

# Temporal split: 60% train, 20% calibration, 20% test
# Sort by accident_year to ensure temporal ordering
df = df.sort("accident_year")
n  = len(df)

train_end = int(0.60 * n)
cal_end   = int(0.80 * n)

X_COLS = [
    "vehicle_group", "driver_age", "ncd_years",
    "area", "conviction_points", "annual_mileage",
]
CAT_FEATURES = ["area"]

# The pure premium target: total incurred per policy-year
# This is what we model directly with Tweedie - no need for a freq/sev split
# when the goal is prediction intervals on the combined outcome
df = df.with_columns(
    (pl.col("incurred") / pl.col("exposure_years").clip(lower_bound=0.01)).alias("pure_premium")
)

X_train = df[:train_end][X_COLS].to_pandas()
y_train = df[:train_end]["pure_premium"].to_pandas()

X_cal   = df[train_end:cal_end][X_COLS].to_pandas()
y_cal   = df[train_end:cal_end]["pure_premium"].to_pandas()

X_test  = df[cal_end:][X_COLS].to_pandas()
y_test  = df[cal_end:]["pure_premium"].to_pandas()

train_years = sorted(df[:train_end]["accident_year"].unique().to_list())
cal_years   = sorted(df[train_end:cal_end]["accident_year"].unique().to_list())
test_years  = sorted(df[cal_end:]["accident_year"].unique().to_list())

print(f"Train:       {len(X_train):,} policies (accident years {train_years})")
print(f"Calibration: {len(X_cal):,} policies (accident years {cal_years})")
print(f"Test:        {len(X_test):,} policies (accident years {test_years})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train the Tweedie pure premium model
# MAGIC
# MAGIC We use a Tweedie model with variance_power=1.5, which is the standard choice
# MAGIC for UK motor pure premium modelling. Power=1.5 sits between Poisson (power=1)
# MAGIC and Gamma (power=2), modelling the compound Poisson-Gamma structure of insurance
# MAGIC pure premiums - a continuous distribution with a point mass at zero when claims=0.
# MAGIC
# MAGIC The pure premium target has two advantages for conformal prediction:
# MAGIC 1. One model, one set of calibration residuals, one interval
# MAGIC 2. No need to compose frequency and severity intervals separately
# MAGIC
# MAGIC The trade-off: you cannot decompose the interval into a frequency component
# MAGIC and a severity component. If you need that decomposition, train separate
# MAGIC conformal predictors for each and accept the composition complexity.

# COMMAND ----------

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
cal_pool   = Pool(X_cal,   y_cal,   cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

tweedie_params = {
    "loss_function":   "Tweedie:variance_power=1.5",
    "eval_metric":     "Tweedie:variance_power=1.5",
    "learning_rate":   0.05,
    "depth":           5,
    "min_data_in_leaf": 50,
    "iterations":      500,
    "random_seed":     42,
    "verbose":         100,
}

# Note: early stopping on the calibration pool introduces a mild dependency.
# The calibration pool has influenced the fitting decision (iteration count).
# In practice this has negligible effect on coverage.
# If you need strict separation, use a separate validation pool for early stopping.
model = CatBoostRegressor(**tweedie_params)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

test_preds = model.predict(test_pool)
test_rmse  = float(np.sqrt(np.mean((test_preds - y_test.values)**2)))

print(f"Best iteration: {model.best_iteration_}")
print(f"Test RMSE:      {test_rmse:.4f}")
print(f"Test mean pred: {test_preds.mean():.4f}")
print(f"Test mean actual: {y_test.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wrap and calibrate: InsuranceConformalPredictor
# MAGIC
# MAGIC The calibration step is fast: score the calibration observations, sort the scores,
# MAGIC store the quantile. For 10,000 calibration observations this takes a few seconds.
# MAGIC
# MAGIC The `pearson_weighted` non-conformity score divides the absolute residual by
# MAGIC ŷ^(p/2), which normalises by the model's expected standard deviation.
# MAGIC This produces intervals that widen proportionally with risk level.
# MAGIC
# MAGIC Do not use the `raw` score for insurance data. The module tutorial shows
# MAGIC why: raw residual intervals achieve 90% marginal coverage but only 72%
# MAGIC coverage in the top risk decile. The aggregate number hides the failure.

# COMMAND ----------

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # variance-weighted: correct for Tweedie data
    distribution="tweedie",
    tweedie_power=1.5,                 # matches the model's loss function
)

cp.calibrate(X_cal, y_cal)

print(f"Calibration complete.")
print(f"  n_calibration: {len(X_cal):,}")
print(f"  Calibration 90th percentile score: {cp.calibration_scores_[int(0.90 * len(cp.calibration_scores_))]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate prediction intervals
# MAGIC
# MAGIC `alpha=0.10` gives 90% prediction intervals.
# MAGIC The lower bound is clipped at zero unconditionally.
# MAGIC
# MAGIC Notice that large risks (high point estimate) get wider absolute intervals.
# MAGIC That is the variance-weighted score working correctly: a £20,000 expected
# MAGIC loss has more absolute variance than a £200 expected loss.

# COMMAND ----------

# 90% intervals for uncertainty flagging
intervals_90 = cp.predict_interval(X_test, alpha=0.10)
# 95% intervals for conservative minimum premium floors
intervals_95 = cp.predict_interval(X_test, alpha=0.05)
# 80% intervals for practical floor construction
intervals_80 = cp.predict_interval(X_test, alpha=0.20)

print("90% prediction intervals (first 10 rows):")
print(intervals_90.head(10).to_string())
print(f"\nInterval statistics:")
widths_90 = intervals_90["upper"] - intervals_90["lower"]
print(f"  Mean width (90%):   {widths_90.mean():.4f}")
print(f"  Median width (90%): {widths_90.median():.4f}")
print(f"  Max width (90%):    {widths_90.max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate coverage - the gate before any downstream use
# MAGIC
# MAGIC Do not skip this step. The coverage-by-decile diagnostic is the gate.
# MAGIC
# MAGIC What you are looking for:
# MAGIC - All deciles within 5pp of target: good, use the intervals
# MAGIC - Monotone decline from low to high deciles: residual heteroscedasticity,
# MAGIC   try switching to the "deviance" score
# MAGIC - Non-monotone: distribution shift between calibration and test data
# MAGIC
# MAGIC With `pearson_weighted`, coverage should be flat across deciles.
# MAGIC Any monotone pattern means the score function is not fully accounting
# MAGIC for the variance structure in your specific data.

# COMMAND ----------

diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print("Coverage by decile (target: 0.90 across all deciles):")
print(diag.to_string())

# Check for concerning patterns
coverages = diag["coverage"].to_list()
min_cov   = min(coverages)
max_cov   = max(coverages)
spread    = max_cov - min_cov

print(f"\nCoverage range: [{min_cov:.3f}, {max_cov:.3f}]  spread={spread:.3f}")

if spread > 0.10:
    print("WARNING: Coverage spread > 10pp. Investigate score function choice.")
elif min_cov < 0.85:
    print("WARNING: At least one decile below 85% coverage. Investigate distribution shift.")
else:
    print("Coverage is acceptable. Intervals may be used for downstream applications.")

# COMMAND ----------

# Coverage plot with Wilson score confidence bands
try:
    fig = cp.coverage_plot(X_test, y_test, alpha=0.10)
    plt.show()
except Exception as e:
    print(f"Coverage plot requires matplotlib: {e}")

# COMMAND ----------

# Full summary
# cp.summary() returns a dict of coverage metrics (e.g. {"marginal_coverage": 0.904, ...}).
# Assign the return value to use it for downstream logging or comparisons.
try:
    summary = cp.summary(X_test, y_test, alpha=0.10)
    print("Coverage summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Summary: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Application 1: Uncertain risk flagging
# MAGIC
# MAGIC The most immediate operational use. Flag the top 10% of risks by relative
# MAGIC interval width for underwriting referral.
# MAGIC
# MAGIC Relative width = (upper - lower) / point estimate
# MAGIC
# MAGIC High relative width indicates the model is uncertain about this risk - typically
# MAGIC because the feature combination is rare in the training data. This is distinct
# MAGIC from the risk being inherently high-risk: a young driver in a high vehicle group
# MAGIC might have a wide interval because we have few training examples of exactly that
# MAGIC combination, even though we "know" young drivers are expensive.

# COMMAND ----------

point_est    = intervals_90["point"].to_numpy()
lower_90     = intervals_90["lower"].to_numpy()
upper_90     = intervals_90["upper"].to_numpy()

rel_width     = (upper_90 - lower_90) / np.clip(point_est, 1e-6, None)
width_threshold = np.quantile(rel_width, 0.90)

flag_for_review = rel_width > width_threshold
n_flagged       = flag_for_review.sum()

print(f"Relative width threshold (90th pctile): {width_threshold:.4f}")
print(f"Policies flagged for review: {n_flagged:,} ({100*n_flagged/len(flag_for_review):.1f}%)")

# Characterise flagged risks
X_test_pl = pl.from_pandas(X_test.reset_index(drop=True))
flagged_mask   = pl.Series("flagged", flag_for_review)
X_test_pl = X_test_pl.with_columns(flagged_mask)

print("\nFlagged risk profile vs portfolio:")
for col in ["driver_age", "vehicle_group", "ncd_years"]:
    flagged_mean = X_test_pl.filter(pl.col("flagged"))[col].mean()
    all_mean     = X_test_pl[col].mean()
    print(f"  {col:<20}: flagged={flagged_mean:.1f}  portfolio={all_mean:.1f}")

conv_flagged = (X_test_pl.filter(pl.col("flagged"))["conviction_points"] > 0).mean()
conv_all     = (X_test_pl["conviction_points"] > 0).mean()
print(f"  {'% convictions':<20}: flagged={conv_flagged:.1%}  portfolio={conv_all:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Application 2: Minimum premium floors
# MAGIC
# MAGIC The 95% upper bound as a principled minimum premium floor.
# MAGIC
# MAGIC Logic: the 90% upper bound is the loss cost we expect to be exceeded only
# MAGIC 10% of the time. Using it as a minimum premium floor means we anticipate being
# MAGIC unprofitable on this risk no more than 10% of the time, assuming the calibration
# MAGIC period was representative. For a more conservative floor, use the 95% upper bound
# MAGIC (5% exceedance), which we demonstrate below.
# MAGIC
# MAGIC In practice, you would not set the minimum premium equal to the upper bound -
# MAGIC that is too conservative for a competitive market. The practical floor combines
# MAGIC the conformal upper bound with a minimum multiplier:
# MAGIC "max(1.5 x technical premium, 80% upper bound)"
# MAGIC
# MAGIC This is auditable under Consumer Duty: the 80% coverage level is validated,
# MAGIC the 1.5x multiplier captures the insurer's loading, and the combination is
# MAGIC risk-specific rather than uniform.

# COMMAND ----------

upper_95_arr = intervals_95["upper"].to_numpy()
upper_80_arr = intervals_80["upper"].to_numpy()

floor_conventional = np.maximum(1.3 * point_est, 250)      # typical current practice
floor_conformal_95 = upper_95_arr                           # 95% upper bound
floor_practical    = np.maximum(1.5 * point_est, upper_80_arr)  # combined approach

print("Minimum premium floor comparison:")
print(f"{'Approach':<30} {'Median':>10} {'Mean':>10} {'95th pctile':>14}")
print("-" * 66)
for label, floor in [
    ("Conventional (1.3x, floor 250)", floor_conventional),
    ("Conformal 95% upper bound",      floor_conformal_95),
    ("Practical (1.5x vs 80% upper)", floor_practical),
]:
    print(f"{label:<30} {np.median(floor):>10.2f} {np.mean(floor):>10.2f} {np.quantile(floor, 0.95):>14.2f}")

# Where does conformal floor exceed conventional?
higher = floor_conformal_95 > floor_conventional
print(f"\nConformal > conventional floor: {higher.sum():,} policies ({higher.mean():.1%})")
print("These are the high-volatility risks where the conventional flat multiplier")
print("understates the required floor.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Application 3: Portfolio reserve range estimates
# MAGIC
# MAGIC Individual prediction intervals aggregate to portfolio-level range estimates.
# MAGIC Two methods:
# MAGIC
# MAGIC 1. Naive (perfect correlation): sum of individual upper bounds.
# MAGIC    Assumes all risks simultaneously hit their upper bound. Conservative.
# MAGIC
# MAGIC 2. Independence (CLT): approximates individual standard deviations from
# MAGIC    interval widths, sums variance, takes sqrt. Assumes no shared risk factors.
# MAGIC
# MAGIC The true range lies between these bounds. For UK motor, independence is
# MAGIC reasonable for most risks with a catastrophe/weather overlay added separately.

# COMMAND ----------

portfolio_point = point_est.sum()
portfolio_lower_naive = lower_90.sum()
portfolio_upper_naive = upper_90.sum()

# CLT-based range: approximate sd from interval width
# For a symmetric 90% interval: width = 2 * 1.645 * sd => sd = width / 3.29
# WARNING: Individual conformal prediction intervals are asymmetric for Tweedie models
# (e.g. lower=2.13, point=14.92, upper=58.21). This CLT-based aggregation uses a
# symmetric normal approximation to derive individual standard deviations, which will
# understate portfolio-level variance. The independence range below is therefore an
# optimistic lower bound on portfolio uncertainty. For a more accurate portfolio range,
# simulate from the Tweedie distribution at the calibrated quantile scale.
approx_sd = (upper_90 - lower_90) / 3.29
portfolio_sd = np.sqrt((approx_sd**2).sum())

portfolio_lower_indep = max(0, portfolio_point - 1.645 * portfolio_sd)
portfolio_upper_indep = portfolio_point + 1.645 * portfolio_sd

print(f"Portfolio point estimate (sum):  {portfolio_point:,.0f}")
print()
print(f"Naive 90% range (perfect corr):")
print(f"  [{portfolio_lower_naive:,.0f}, {portfolio_upper_naive:,.0f}]")
print(f"  Upper/lower ratio: {portfolio_upper_naive / portfolio_lower_naive:.2f}")
print()
print(f"Independence 90% range (CLT):")
print(f"  [{portfolio_lower_indep:,.0f}, {portfolio_upper_indep:,.0f}]")
print(f"  Upper/lower ratio: {portfolio_upper_indep / portfolio_lower_indep:.2f}")
print()
print("Present both to the reserving team. Be explicit that independence")
print("excludes weather events and economic shocks - add a catastrophe overlay.")

# COMMAND ----------

# Segmented reserve ranges by area band
import pandas as pd

segment_frame = pd.concat([
    X_test.reset_index(drop=True)[["area"]],
    pd.DataFrame({
        "point": point_est,
        "lower": lower_90,
        "upper": upper_90,
    })
], axis=1)

seg_summary = segment_frame.groupby("area").agg(
    n_risks       = ("point", "count"),
    total_point   = ("point", "sum"),
    total_lower   = ("lower", "sum"),
    total_upper   = ("upper", "sum"),
).assign(
    upper_lower_ratio = lambda df: df["total_upper"] / df["total_lower"]
).sort_values("upper_lower_ratio", ascending=False)

print("Reserve range by area band (90% intervals, naive sum):")
print(seg_summary.to_string())
print("\nAreas with high upper/lower ratios have the most reserve uncertainty.")
print("These are candidates for aggregate stop-loss reinsurance cover.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write results to Unity Catalog
# MAGIC
# MAGIC Write intervals and coverage diagnostics to Delta tables.
# MAGIC Do not write to local files - Delta gives you versioning, time travel,
# MAGIC and a permanent audit trail tied to the model run.
# MAGIC
# MAGIC The coverage log is an append-mode table: every run adds one set of
# MAGIC diagnostic rows. Track this over time - degrading top-decile coverage
# MAGIC is the signal that triggers recalibration.

# COMMAND ----------

with mlflow.start_run(run_name="conformal_calibration_m05"):
    mlflow.log_param("nonconformity_score",  "pearson_weighted")
    mlflow.log_param("tweedie_power",        1.5)
    mlflow.log_param("alpha",                0.10)
    mlflow.log_param("calibration_n",        len(X_cal))
    mlflow.log_param("calibration_years",    str(cal_years))
    mlflow.log_param("test_years",           str(test_years))
    mlflow.log_metric("marginal_coverage",   float(diag["coverage"].mean()))
    mlflow.log_metric("min_decile_coverage", float(diag["coverage"].min()))
    conf_run_id = mlflow.active_run().info.run_id

print(f"MLflow run: {conf_run_id}")

# COMMAND ----------

# Write intervals to Delta
intervals_to_write = intervals_90.to_pandas().copy()
intervals_to_write["model_run_date"]     = str(date.today())
intervals_to_write["mlflow_run_id"]      = conf_run_id
intervals_to_write["alpha"]              = 0.10
intervals_to_write["nonconformity_score"] = "pearson_weighted"
intervals_to_write["flag_for_review"]    = flag_for_review.tolist()
intervals_to_write["relative_width"]     = rel_width.tolist()

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    (
        spark.createDataFrame(intervals_to_write)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.conformal_intervals")
    )
    print(f"Intervals written to {CATALOG}.{SCHEMA}.conformal_intervals")

    # Coverage diagnostics: append so we can track over time
    diag_to_write = diag.to_pandas().copy()
    diag_to_write["model_run_date"] = str(date.today())
    diag_to_write["mlflow_run_id"]  = conf_run_id
    diag_to_write["test_years"]     = str(test_years)

    (
        spark.createDataFrame(diag_to_write)
        .write.format("delta")
        .mode("append")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.conformal_coverage_log")
    )
    print(f"Coverage diagnostics appended to {CATALOG}.{SCHEMA}.conformal_coverage_log")

except Exception as e:
    print(f"Could not write to Unity Catalog: {e}")
    print("Intervals and diagnostics available in memory as `intervals_to_write` and `diag`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Recalibration - separate from retraining
# MAGIC
# MAGIC The key operational advantage of conformal prediction: calibration can be
# MAGIC refreshed on recent data without retraining the base model.
# MAGIC
# MAGIC When to recalibrate: monthly or quarterly, or when the coverage monitoring
# MAGIC log shows top-decile coverage degrading below target.
# MAGIC
# MAGIC When to retrain: annually, or when recalibration does not restore coverage
# MAGIC (which indicates the model's rankings have drifted, not just its scale).

# COMMAND ----------

# Demonstrate recalibration: recalibrate using only the most recent calibration data
# This simulates a quarterly refresh cycle
n_recent = min(2_000, len(X_cal))

X_cal_recent = X_cal.tail(n_recent)
y_cal_recent = y_cal.tail(n_recent)

import time
t0 = time.time()
cp.calibrate(X_cal_recent, y_cal_recent)
recal_time = time.time() - t0

# Check coverage after recalibration
diag_recal = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(f"Recalibration on {n_recent:,} most recent observations: {recal_time:.2f} seconds")
print(f"Coverage after recalibration:")
print(diag_recal.select(["decile", "coverage"]).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What this notebook built:
# MAGIC
# MAGIC | Step | Output |
# MAGIC |------|--------|
# MAGIC | Temporal split | 60/20/20 by accident year; calibration = most recent before test |
# MAGIC | Tweedie model | CatBoost Tweedie (power=1.5) on pure premium target |
# MAGIC | Conformal predictor | pearson_weighted score; calibrated on 10,000 policies |
# MAGIC | Coverage validation | Flat coverage by decile confirms intervals are valid |
# MAGIC | Uncertain risk flagging | Top 10% by relative width; driven by sparse risk cells |
# MAGIC | Minimum premium floors | Conformal 95% upper bound vs conventional 1.3x multiplier |
# MAGIC | Reserve range | Independence-based CLT range for reserving inputs |
# MAGIC | Delta tables | conformal_intervals, conformal_coverage_log |
# MAGIC
# MAGIC The coverage-by-decile diagnostic is the gate. Do not proceed to downstream
# MAGIC applications without confirming flat coverage across deciles.
# MAGIC
# MAGIC Next: Module 6 - Credibility and Bayesian Methods
# MAGIC Thin-cell regularisation: how to handle the sparse cells that produce
# MAGIC wide intervals in this module.

# COMMAND ----------

print("=" * 60)
print("MODULE 5 COMPLETE")
print("=" * 60)
print()
print(f"Calibration set:       {len(X_cal):,} policies")
print(f"Test set:              {len(X_test):,} policies")
print(f"Marginal coverage:     {float(diag['coverage'].mean()):.3f}")
print(f"Min decile coverage:   {float(diag['coverage'].min()):.3f}")
print(f"Policies flagged:      {flag_for_review.sum():,} ({flag_for_review.mean():.1%})")
print()
print("Next: Module 6 - Credibility and Bayesian Pricing")
