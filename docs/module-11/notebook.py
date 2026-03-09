# Databricks notebook source
# MAGIC %md
# MAGIC # Module 11: Model Monitoring and Drift Detection
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC A trained model is not a finished product. It is an assumption that the world it was trained on
# MAGIC still resembles the world it is scoring. This notebook shows you how to test that assumption
# MAGIC systematically, using the `insurance-monitoring` library.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates synthetic UK motor policy data with reference and current periods
# MAGIC 2. Trains a CatBoost frequency model on the reference period
# MAGIC 3. Simulates a drifted "current" dataset with shifted feature distributions
# MAGIC 4. Computes Population Stability Index (PSI) on predicted scores
# MAGIC 5. Computes Characteristic Stability Index (CSI) on all model features
# MAGIC 6. Computes Actual vs Expected ratios with Poisson confidence intervals
# MAGIC 7. Runs a Gini drift z-test (DeLong method)
# MAGIC 8. Assembles a MonitoringReport with automated traffic-light signals
# MAGIC 9. Interprets the combined signals
# MAGIC 10. Writes all monitoring results to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 20-30 minutes on a 4-core cluster.
# MAGIC
# MAGIC **Prerequisites:** None. This notebook is self-contained and generates its own synthetic data.
# MAGIC In a real deployment, replace the data generation cell with
# MAGIC `spark.table("your_catalog.motor.policies")`.
# MAGIC
# MAGIC **Key concept:** PSI and CSI tell you what has changed. A/E and Gini tell you whether it matters
# MAGIC for pricing accuracy. Run them together. Act on the combination, not any single metric in isolation.

# COMMAND ----------

%pip install insurance-monitoring catboost polars mlflow scikit-learn --quiet
# Use %pip in Databricks notebooks. Outside Databricks, use: uv add insurance-monitoring catboost polars mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from insurance_monitoring import (
    PSICalculator,
    CSICalculator,
    AERatio,
    GiniDrift,
    MonitoringReport,
)

warnings.filterwarnings("ignore", category=UserWarning)

print(f"Today: {date.today()}")
print(f"Polars:              {pl.__version__}")
print("insurance-monitoring: imported OK")
print("All imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0: Configuration
# MAGIC
# MAGIC All configurable values in one cell. If a value appears in more than one place it
# MAGIC will eventually be inconsistent. In production, drive these from a config file or
# MAGIC Databricks widgets.

# COMMAND ----------

from datetime import date

# -----------------------------------------------------------------------
# Unity Catalog coordinates
# -----------------------------------------------------------------------
CATALOG = "main"
SCHEMA  = "motor_monitoring"

TABLES = {
    "monitoring_log": f"{CATALOG}.{SCHEMA}.monitoring_log",
    "csi_results":    f"{CATALOG}.{SCHEMA}.csi_results",
    "ae_results":     f"{CATALOG}.{SCHEMA}.ae_results",
}

# -----------------------------------------------------------------------
# Monitoring parameters
# -----------------------------------------------------------------------
# REFERENCE_DATE: when the model was trained / last validated
# CURRENT_DATE:   end of the monitoring window we are assessing
REFERENCE_DATE = "2023-12-31"
CURRENT_DATE   = "2024-06-30"

MODEL_NAME    = "motor_frequency_catboost"
MODEL_VERSION = "1"

N_BINS = 10  # PSI/CSI bin count. 10 is the standard; reduce to 5 for sparse features

RUN_DATE = str(date.today())

print(f"Run date:          {RUN_DATE}")
print(f"Reference period:  up to {REFERENCE_DATE}")
print(f"Current period:    {REFERENCE_DATE} to {CURRENT_DATE}")
print(f"Catalog/schema:    {CATALOG}.{SCHEMA}")
print(f"Model:             {MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Generate synthetic motor data
# MAGIC
# MAGIC We generate 80,000 synthetic UK motor policies across two periods. The reference period
# MAGIC represents 2022-2023. The "current" period represents H1 2024 and has deliberate
# MAGIC distribution shifts injected to simulate the drift scenarios you encounter in practice:
# MAGIC - Young driver proportion has grown (acquisition of a comparison-site book)
# MAGIC - High-mileage vehicles have increased (post-pandemic behaviour shift)
# MAGIC - Claim frequency has crept up for young drivers (concept drift)
# MAGIC
# MAGIC In production, replace this with `spark.table("your_catalog.motor.policies")`.

# COMMAND ----------

rng = np.random.default_rng(seed=42)

N_REF = 60_000   # reference period: model training data
N_CUR = 20_000   # current monitoring period

FEATURE_NAMES = [
    "driver_age", "vehicle_age", "vehicle_group",
    "annual_mileage", "ncd_years", "region",
]
CAT_FEATURES = ["region"]


def generate_motor_data(n, period, rng, drift=False):
    """
    Generate synthetic UK motor policy data.

    Parameters
    ----------
    n       : number of policies
    period  : label string, stored in the output
    rng     : numpy random generator (for reproducibility)
    drift   : if True, inject distribution shifts into the current period
    """
    # --- Feature distributions ---
    if drift:
        # Young driver proportion shifted up: more under-25s in current data
        driver_age = np.where(
            rng.random(n) < 0.25,
            rng.integers(17, 25, n),       # 25% young drivers (was ~15%)
            rng.integers(25, 80, n),
        )
        # Higher mileage: 20% shift in annual mileage distribution
        annual_mileage = rng.gamma(shape=4.5, scale=3200, size=n).astype(int).clip(2000, 50000)
    else:
        driver_age     = np.where(
            rng.random(n) < 0.15,
            rng.integers(17, 25, n),
            rng.integers(25, 80, n),
        )
        annual_mileage = rng.gamma(shape=4.0, scale=3000, size=n).astype(int).clip(2000, 50000)

    vehicle_age   = rng.integers(0, 20, n)
    vehicle_group = rng.integers(1, 51, n)    # 1-50 ABI group
    ncd_years     = rng.integers(0, 10, n)
    region        = rng.choice(
        ["North", "Midlands", "London", "South", "Scotland", "Wales"],
        size=n,
        p=[0.20, 0.20, 0.20, 0.20, 0.12, 0.08],
    )
    exposure      = rng.uniform(0.25, 1.0, n)  # fraction of a year

    # --- True log-rate model for claim frequency ---
    log_rate = (
        -3.0
        + np.where(driver_age < 25, 0.80, 0.0)
        + np.where(driver_age > 70, 0.30, 0.0)
        - 0.015 * ncd_years
        + 0.008 * (annual_mileage / 1000)
        + 0.012 * vehicle_age
        + 0.003 * vehicle_group
        + np.where(region == "London", 0.35, 0.0)
        + np.where(region == "Scotland", -0.20, 0.0)
        + rng.normal(0, 0.10, n)   # unexplained noise
    )

    if drift:
        # Concept drift: young drivers claiming more than model expects
        log_rate += np.where(driver_age < 25, 0.25, 0.0)

    freq      = np.exp(log_rate)
    claim_count = rng.poisson(freq * exposure)

    return pl.DataFrame({
        "period":         [period] * n,
        "driver_age":     driver_age.tolist(),
        "vehicle_age":    vehicle_age.tolist(),
        "vehicle_group":  vehicle_group.tolist(),
        "annual_mileage": annual_mileage.tolist(),
        "ncd_years":      ncd_years.tolist(),
        "region":         region.tolist(),
        "exposure":       exposure.tolist(),
        "claim_count":    claim_count.tolist(),
    })


df_reference = generate_motor_data(N_REF, "reference", rng, drift=False)
df_current   = generate_motor_data(N_CUR, "current",   rng, drift=True)

print(f"Reference records: {len(df_reference):,}")
print(f"Current records:   {len(df_current):,}")
print()
print(f"Reference claim rate: {df_reference['claim_count'].sum() / df_reference['exposure'].sum():.4f}")
print(f"Current claim rate:   {df_current['claim_count'].sum() / df_current['exposure'].sum():.4f}")
print()
print("Reference young driver proportion:",
      f"{(df_reference['driver_age'] < 25).sum() / len(df_reference):.2%}")
print("Current young driver proportion:",
      f"{(df_current['driver_age'] < 25).sum() / len(df_current):.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Train the frequency model on reference data
# MAGIC
# MAGIC We use CatBoost with a Poisson loss function, which is the standard choice for claim
# MAGIC frequency in UK motor pricing. The exposure is passed as a log-offset — not as a weight.
# MAGIC
# MAGIC The model is trained on the reference period only. We hold out 20% for evaluation.
# MAGIC In Module 8 this model would be registered in MLflow and loaded by name here; for
# MAGIC this self-contained notebook we train it fresh.

# COMMAND ----------

# Train/validation split within the reference period
ref_pd = df_reference.to_pandas()

split   = int(0.8 * len(ref_pd))
X_train = ref_pd[:split][FEATURE_NAMES]
y_train = ref_pd[:split]["claim_count"]
e_train = ref_pd[:split]["exposure"]

X_val   = ref_pd[split:][FEATURE_NAMES]
y_val   = ref_pd[split:]["claim_count"]
e_val   = ref_pd[split:]["exposure"]

# Pool objects with exposure as baseline (log-offset)
train_pool = Pool(
    X_train, y_train,
    cat_features=CAT_FEATURES,
    weight=None,
    baseline=np.log(e_train.values),
)
val_pool = Pool(
    X_val, y_val,
    cat_features=CAT_FEATURES,
    baseline=np.log(e_val.values),
)

model = CatBoostClassifier(
    loss_function="Poisson",
    eval_metric="Poisson",
    learning_rate=0.05,
    depth=5,
    min_data_in_leaf=50,
    iterations=400,
    random_seed=42,
    verbose=100,
)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)

print(f"Best iteration: {model.best_iteration_}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Generate predictions for both periods
# MAGIC
# MAGIC We generate predicted frequencies (annualised rate, not expected claim count) for
# MAGIC every policy in both the reference and current windows.
# MAGIC
# MAGIC PSI runs on the raw frequency predictions. A/E runs on predicted claim counts
# MAGIC (frequency x exposure). Gini runs on the binary claim indicator (did this policy
# MAGIC have at least one claim?).

# COMMAND ----------

ref_pd = df_reference.to_pandas()
cur_pd = df_current.to_pandas()

# CatBoost Poisson predict() returns the rate (not the count)
# The baseline offset is applied internally during training; at predict time
# we want the rate without the exposure offset
ref_pool_pred = Pool(ref_pd[FEATURE_NAMES], cat_features=CAT_FEATURES)
cur_pool_pred = Pool(cur_pd[FEATURE_NAMES], cat_features=CAT_FEATURES)

pred_ref = model.predict(ref_pool_pred)
pred_cur = model.predict(cur_pool_pred)

exposure_ref = ref_pd["exposure"].values
exposure_cur = cur_pd["exposure"].values

# Expected claim counts = rate * exposure
expected_ref = pred_ref * exposure_ref
expected_cur = pred_cur * exposure_cur

# Binary claim indicator for Gini
y_ref = (ref_pd["claim_count"] > 0).astype(int).values
y_cur = (cur_pd["claim_count"] > 0).astype(int).values

actual_ref = ref_pd["claim_count"].values.astype(float)
actual_cur = cur_pd["claim_count"].values.astype(float)

print(f"Reference predictions: mean={pred_ref.mean():.4f}, "
      f"min={pred_ref.min():.4f}, max={pred_ref.max():.4f}")
print(f"Current predictions:   mean={pred_cur.mean():.4f}, "
      f"min={pred_cur.min():.4f}, max={pred_cur.max():.4f}")
print()
print(f"Reference AUC: {roc_auc_score(y_ref, pred_ref):.4f}")
print(f"Current AUC:   {roc_auc_score(y_cur, pred_cur):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Population Stability Index (PSI) on predicted scores
# MAGIC
# MAGIC PSI measures whether the distribution of predicted frequencies has shifted between
# MAGIC the reference and current periods. We run it on predicted scores first because a
# MAGIC stable score distribution with elevated A/E means something different from an unstable
# MAGIC score distribution with elevated A/E.
# MAGIC
# MAGIC Thresholds (industry standard, from credit scoring practice):
# MAGIC - PSI < 0.10: no significant change
# MAGIC - PSI 0.10-0.20: minor change, worth monitoring
# MAGIC - PSI > 0.20: significant change, investigate

# COMMAND ----------

psi_calc = PSICalculator(n_bins=N_BINS)

psi_result = psi_calc.calculate(
    reference=pred_ref,
    current=pred_cur,
    exposure_ref=exposure_ref,   # weight by exposure for correctness
    exposure_cur=exposure_cur,
)

print(f"PSI (score distribution): {psi_result.psi:.4f}")
print(f"Traffic light:            {psi_result.traffic_light}")
print()
print("Bin-level breakdown:")
for bin_info in psi_result.bins:
    print(f"  [{bin_info.lower:.3f}, {bin_info.upper:.3f}): "
          f"ref={bin_info.ref_pct:.2%}, "
          f"cur={bin_info.cur_pct:.2%}, "
          f"contribution={bin_info.contribution:.4f}")

# Store for report
psi_score = psi_result

# COMMAND ----------

# Visualise the PSI result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(pred_ref, bins=50, alpha=0.6, label="Reference", color="steelblue", density=True)
ax1.hist(pred_cur, bins=50, alpha=0.6, label="Current",   color="tomato",    density=True)
ax1.set_xlabel("Predicted frequency (annualised)")
ax1.set_ylabel("Density")
ax1.set_title(f"Score distribution  (PSI = {psi_result.psi:.3f}, {psi_result.traffic_light})")
ax1.legend()
ax1.axvline(np.median(pred_ref), color="steelblue", linestyle="--", alpha=0.7, label="Ref median")
ax1.axvline(np.median(pred_cur), color="tomato",    linestyle="--", alpha=0.7, label="Cur median")

contributions = [b.contribution for b in psi_result.bins]
colors        = ["green" if c < 0.02 else "orange" if c < 0.05 else "red" for c in contributions]
ax2.bar(range(len(contributions)), contributions, color=colors)
ax2.set_xlabel("PSI bin")
ax2.set_ylabel("Contribution to PSI")
ax2.set_title("PSI contribution by bin")
ax2.axhline(0.02, color="orange", linestyle="--", alpha=0.7, label="Amber threshold")
ax2.axhline(0.05, color="red",    linestyle="--", alpha=0.7, label="Red threshold")
ax2.legend()

plt.tight_layout()
plt.savefig("/tmp/psi_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to /tmp/psi_score_distribution.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5: Characteristic Stability Index (CSI) on features
# MAGIC
# MAGIC CSI is PSI applied to each input feature. It tells you which features have drifted.
# MAGIC A high PSI with low CSI across all features means the distribution shift is not
# MAGIC explained by the individual features (interaction effects, or a feature not in the
# MAGIC model). A high PSI with one high-CSI feature points directly to the cause.
# MAGIC
# MAGIC For categorical features, each distinct value is its own bin. New categories in the
# MAGIC current data that did not exist in reference are a hard signal — investigate the
# MAGIC data pipeline before the model.

# COMMAND ----------

csi_calc = CSICalculator(n_bins=N_BINS)

csi_results = {}
for feature in FEATURE_NAMES:
    ref_values = df_reference[feature].to_numpy()
    cur_values = df_current[feature].to_numpy()

    result = csi_calc.calculate(
        feature_name=feature,
        reference=ref_values,
        current=cur_values,
    )
    csi_results[feature] = result

# Summary table, ranked by CSI
print(f"{'Feature':<25} {'CSI':>8}  {'Status'}")
print("-" * 50)
for feat, result in sorted(csi_results.items(), key=lambda x: x[1].csi, reverse=True):
    flag = " <-- investigate" if result.csi > 0.20 else ""
    print(f"{feat:<25} {result.csi:>8.4f}  {result.traffic_light}{flag}")

# Store for report
csi_scores = csi_results

# COMMAND ----------

# Check for new categories in categorical features
for feature in CAT_FEATURES:
    ref_cats = set(df_reference[feature].unique().to_list())
    cur_cats = set(df_current[feature].unique().to_list())
    new_cats     = cur_cats - ref_cats
    missing_cats = ref_cats - cur_cats
    if new_cats:
        print(f"{feature}: NEW categories in current: {new_cats}")
    if missing_cats:
        print(f"{feature}: MISSING from current (ref only): {missing_cats}")

print("Category check complete.")

# COMMAND ----------

# Visualise features with notable drift
notable_features = [f for f, r in csi_results.items() if r.csi > 0.10]

if not notable_features:
    print("No features with CSI > 0.10. Book mix is stable.")
else:
    n_features = len(notable_features)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, notable_features):
        ref_vals  = df_reference[feature].to_numpy()
        cur_vals  = df_current[feature].to_numpy()
        csi_val   = csi_results[feature].csi

        ax.hist(ref_vals, bins=30, alpha=0.6, label="Reference", color="steelblue", density=True)
        ax.hist(cur_vals, bins=30, alpha=0.6, label="Current",   color="tomato",    density=True)
        ax.set_title(f"{feature}  (CSI={csi_val:.3f})")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig("/tmp/csi_feature_drift.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Features with CSI > 0.10: {notable_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6: Actual vs Expected ratios
# MAGIC
# MAGIC The A/E ratio is the ratio of observed claims to predicted claims. It tests calibration:
# MAGIC is the model's average prediction correct? We compute it at portfolio level first, then
# MAGIC break it down by segment to find where calibration has drifted most.
# MAGIC
# MAGIC The confidence interval uses the Wald approximation (sqrt of observed claims, not
# MAGIC expected). This is slightly conservative when A/E is far from 1.0 but adequate for
# MAGIC routine monitoring. If 1.0 is inside the 95% CI, there is no statistically significant
# MAGIC evidence of a calibration problem.

# COMMAND ----------

ae_calc = AERatio()

ae_result = ae_calc.calculate(
    actual=actual_cur,
    expected=expected_cur,
    exposure=exposure_cur,
)

print(f"Portfolio A/E ratio: {ae_result.ratio:.4f}")
print(f"95% CI:              [{ae_result.ci_lower:.4f}, {ae_result.ci_upper:.4f}]")
print(f"Actual claims:       {actual_cur.sum():.0f}")
print(f"Expected claims:     {expected_cur.sum():.1f}")
print(f"Traffic light:       {ae_result.traffic_light}")
print()
if ae_result.ci_lower > 1.0:
    print("CI excludes 1.0 from below. Model is systematically under-predicting.")
elif ae_result.ci_upper < 1.0:
    print("CI excludes 1.0 from above. Model is systematically over-predicting.")
else:
    print("CI contains 1.0. No statistically significant evidence of calibration drift.")

# Store for report
ae_portfolio = ae_result

# COMMAND ----------

# Segment A/E breakdown by driver age band
df_cur_with_preds = df_current.with_columns([
    pl.Series("expected", expected_cur),
])

age_bands = [(17, 25, "17-24"), (25, 40, "25-39"), (40, 60, "40-59"), (60, 100, "60+")]

print("\nA/E by driver age band:")
print(f"{'Band':<15} {'A/E':>8}  {'CI lower':>10}  {'CI upper':>10}  {'Actual':>8}  {'Expected':>10}")
print("-" * 70)

for low, high, label in age_bands:
    mask = (
        (df_cur_with_preds["driver_age"] >= low) &
        (df_cur_with_preds["driver_age"] < high)
    )
    seg = df_cur_with_preds.filter(mask)
    if len(seg) == 0:
        continue

    result = ae_calc.calculate(
        actual=seg["claim_count"].to_numpy().astype(float),
        expected=seg["expected"].to_numpy(),
        exposure=seg["exposure"].to_numpy(),
    )
    print(f"{label:<15} {result.ratio:>8.4f}  {result.ci_lower:>10.4f}  "
          f"{result.ci_upper:>10.4f}  {seg['claim_count'].sum():>8.0f}  "
          f"{seg['expected'].sum():>10.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: Gini drift z-test (DeLong method)
# MAGIC
# MAGIC The Gini coefficient measures discrimination: does the model rank high-risk policies
# MAGIC above low-risk ones? A falling Gini means the model is less able to distinguish
# MAGIC claimants from non-claimants. This is concept drift.
# MAGIC
# MAGIC This is a different signal from A/E:
# MAGIC - A/E off, Gini OK: calibration problem. Apply a recalibration factor.
# MAGIC - Gini dropping, A/E may or may not be affected: discrimination has weakened. Retraining required.
# MAGIC - Both off: serious. Escalate immediately.
# MAGIC
# MAGIC The DeLong variance estimator properly accounts for the correlation structure when
# MAGIC comparing AUC across two samples.

# COMMAND ----------

gini_calc = GiniDrift()

gini_result = gini_calc.calculate(
    y_ref=y_ref,
    pred_ref=pred_ref,
    y_cur=y_cur,
    pred_cur=pred_cur,
)

print(f"Gini (reference): {gini_result.gini_ref:.4f}")
print(f"Gini (current):   {gini_result.gini_cur:.4f}")
print(f"Difference:       {gini_result.gini_cur - gini_result.gini_ref:+.4f}")
print(f"Z-statistic:      {gini_result.z_stat:.4f}")
print(f"P-value:          {gini_result.p_value:.4f}")
print(f"Traffic light:    {gini_result.traffic_light}")
print()
if gini_result.p_value < 0.05:
    drop = gini_result.gini_ref - gini_result.gini_cur
    if drop > 0.03:
        print("Statistically significant Gini drop >= 0.03. Discrimination has weakened. "
              "Retraining is required.")
    else:
        print("Statistically significant change but drop < 0.03. Monitor closely.")
else:
    print("No statistically significant Gini change. Discrimination is stable.")

# Store for report
gini_drift = gini_result

# COMMAND ----------

# ROC curves for reference and current periods
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr_ref, tpr_ref, _ = roc_curve(y_ref, pred_ref)
axes[0].plot(fpr_ref, tpr_ref, color="steelblue", linewidth=2,
             label=f"Reference (Gini={gini_result.gini_ref:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC - Reference period")
axes[0].legend()

fpr_cur, tpr_cur, _ = roc_curve(y_cur, pred_cur)
axes[1].plot(fpr_cur, tpr_cur, color="tomato", linewidth=2,
             label=f"Current (Gini={gini_result.gini_cur:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC - Current period")
axes[1].legend()

plt.suptitle(
    f"Gini drift: {gini_result.gini_ref:.3f} -> {gini_result.gini_cur:.3f}  "
    f"(p={gini_result.p_value:.3f})",
    fontsize=13
)
plt.tight_layout()
plt.savefig("/tmp/gini_drift.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/gini_drift.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8: Build the MonitoringReport
# MAGIC
# MAGIC The `MonitoringReport` class assembles all four metrics and applies the combined
# MAGIC traffic-light rules:
# MAGIC
# MAGIC | Metric | Green | Amber | Red |
# MAGIC |--------|-------|-------|-----|
# MAGIC | Score PSI | < 0.10 | 0.10-0.20 | > 0.20 |
# MAGIC | A/E ratio | CI contains 1.0 | CI excludes 1.0, ratio in [0.90, 1.10] | CI excludes 1.0, ratio outside [0.90, 1.10] |
# MAGIC | Gini drop | drop < 0.03 AND p > 0.10 | weak signal | p < 0.05 and drop >= 0.03 |
# MAGIC | Any CSI | < 0.10 | 0.10-0.20 | > 0.20 |
# MAGIC
# MAGIC Overall: RED if any metric is RED; AMBER if two or more metrics are AMBER or one is
# MAGIC AMBER and rest are GREEN; GREEN otherwise.

# COMMAND ----------

report = MonitoringReport(
    model_name=MODEL_NAME,
    reference_date=REFERENCE_DATE,
    current_date=CURRENT_DATE,
)

report.add_psi(psi_score)

for feature, result in csi_scores.items():
    report.add_csi(result)

report.add_ae(ae_portfolio)
report.add_gini_drift(gini_drift)

# COMMAND ----------

summary = report.summary()

print("=" * 60)
print("MONITORING REPORT")
print(f"Model:     {summary['model_name']}")
print(f"Reference: {summary['reference_date']}")
print(f"Current:   {summary['current_date']}")
print(f"Run date:  {summary['run_date']}")
print("=" * 60)
print()

overall = summary["overall_traffic_light"]
print(f"OVERALL STATUS: {overall}")
print()

m = summary["metrics"]
print(f"{'Metric':<35} {'Value':>10}  {'Status'}")
print("-" * 60)
print(f"{'Score PSI':<35} {m['psi_score']['value']:>10.4f}  {m['psi_score']['traffic_light']}")
print(f"{'A/E ratio':<35} {m['ae_ratio']['value']:>10.4f}  {m['ae_ratio']['traffic_light']}")
print(f"{'A/E CI lower':<35} {m['ae_ratio']['ci_lower']:>10.4f}")
print(f"{'A/E CI upper':<35} {m['ae_ratio']['ci_upper']:>10.4f}")
print(f"{'Gini (reference)':<35} {m['gini']['gini_ref']:>10.4f}")
print(f"{'Gini (current)':<35} {m['gini']['gini_cur']:>10.4f}  {m['gini']['traffic_light']}")
print(f"{'Gini p-value':<35} {m['gini']['p_value']:>10.4f}")
print()
print("FEATURE CSI:")
print(f"{'Feature':<30} {'CSI':>8}  {'Status'}")
print("-" * 50)
for csi_item in sorted(summary["csi"], key=lambda x: x["csi"], reverse=True):
    print(f"{csi_item['feature']:<30} {csi_item['csi']:>8.4f}  {csi_item['traffic_light']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: Interpret the signals
# MAGIC
# MAGIC This cell prints a structured interpretation of the combined monitoring signals.
# MAGIC In production this would be the body of an automated email to the pricing team.
# MAGIC
# MAGIC The five scenarios to recognise:
# MAGIC 1. All green: log and move on
# MAGIC 2. Elevated PSI, green A/E, stable Gini: mix shift absorbed by model — no action
# MAGIC 3. Elevated PSI, elevated A/E, stable Gini: calibration problem — apply recalibration factor
# MAGIC 4. Green PSI, elevated A/E, falling Gini: concept drift — escalate, begin retraining
# MAGIC 5. Elevated CSI on one feature, all else green: investigate the feature, no immediate model action

# COMMAND ----------

def interpret_monitoring_report(summary):
    """Print a structured interpretation of the monitoring summary."""
    m       = summary["metrics"]
    overall = summary["overall_traffic_light"]
    psi_val = m["psi_score"]["value"]
    psi_tl  = m["psi_score"]["traffic_light"]
    ae_val  = m["ae_ratio"]["value"]
    ae_tl   = m["ae_ratio"]["traffic_light"]
    gini_tl = m["gini"]["traffic_light"]
    gini_p  = m["gini"]["p_value"]
    gini_drop = m["gini"]["gini_ref"] - m["gini"]["gini_cur"]

    red_csi_features = [c["feature"] for c in summary["csi"] if c["traffic_light"] == "RED"]

    print(f"INTERPRETATION ({overall})")
    print("=" * 60)

    # Scenario 1: all green
    if overall == "GREEN":
        print("All metrics within tolerance. No model action required.")
        print("Log this result and file as evidence of ongoing monitoring.")
        return

    # Scenario 4: concept drift (most serious)
    if (psi_tl == "GREEN" and ae_tl in ("AMBER", "RED")
            and gini_tl in ("AMBER", "RED") and gini_p < 0.05 and gini_drop > 0.02):
        print("PATTERN: Green PSI, elevated A/E, falling Gini.")
        print("This is CONCEPT DRIFT. The relationship between features and claims has changed.")
        print("A recalibration factor will fix the average but not the discrimination.")
        print("ACTION: Escalate to head of pricing. Begin retraining. Apply temporary")
        print("recalibration factor (1/A/E) as a holding measure while retraining runs.")
        return

    # Scenario 3: calibration problem
    if (psi_tl in ("AMBER", "RED") and ae_tl in ("AMBER", "RED")
            and gini_tl == "GREEN"):
        print("PATTERN: Elevated PSI, elevated A/E, stable Gini.")
        print(f"Score distribution has shifted (PSI={psi_val:.3f}) and the model is")
        print(f"{'under' if ae_val > 1.0 else 'over'}-predicting overall (A/E={ae_val:.3f}).")
        print("Discrimination is intact (Gini stable). This is a calibration problem")
        print("driven by mix shift.")
        print(f"ACTION: Apply a recalibration factor of 1/{ae_val:.3f} = "
              f"{1/ae_val:.3f} to all predictions while investigating root cause.")
        return

    # Scenario 2: mix shift absorbed
    if psi_tl in ("AMBER", "RED") and ae_tl == "GREEN" and gini_tl == "GREEN":
        print("PATTERN: Elevated PSI, green A/E, stable Gini.")
        print("The book mix has changed but the model is correctly applying risk loads")
        print("to the new mix. No model action required.")
        print("ACTION: Log the CSI results. Update the reference distribution if the")
        print("new mix is structural (not seasonal).")
        return

    # Scenario 5: single feature CSI elevated, all else ok
    if red_csi_features and ae_tl == "GREEN" and gini_tl == "GREEN":
        print(f"PATTERN: High CSI on {red_csi_features}, other metrics green.")
        print("Feature(s) have shifted but are not materially affecting predictions.")
        print("ACTION: Investigate whether the shift is a data quality issue or genuine.")
        print("Flag for review at next model validation. No immediate model action.")
        return

    # Generic fallback
    print(f"Combined amber/red signals across: PSI={psi_tl}, A/E={ae_tl}, Gini={gini_tl}")
    print("Review individual metrics above. No single scenario pattern matched.")
    print("Bring to pricing committee with the full monitoring pack.")


interpret_monitoring_report(summary)

# COMMAND ----------

# Save the report as JSON to DBFS
report_json = json.dumps(summary, indent=2, default=str)
report_path = f"/tmp/monitoring_report_{CURRENT_DATE}.json"
with open(report_path, "w") as f:
    f.write(report_json)
print(f"Report saved to {report_path}")
print()
print("Report JSON (first 600 chars):")
print(report_json[:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 10: Write monitoring results to Delta tables
# MAGIC
# MAGIC Three tables:
# MAGIC - `monitoring_log` — one row per monitoring run (the headline summary)
# MAGIC - `csi_results` — one row per feature per run (the detail for trend analysis)
# MAGIC - `ae_results` — one row per segment per run
# MAGIC
# MAGIC We set a 7-year retention policy on the monitoring log. FCA record-keeping requirements
# MAGIC are broadly aligned with this window. Without explicitly setting `logRetentionDuration`,
# MAGIC time travel fails after 30 days even if data files are present.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

from pyspark.sql import Row

# -----------------------------------------------------------------------
# Write the monitoring summary (one row per run)
# -----------------------------------------------------------------------
summary_row = {
    "run_date":              RUN_DATE,
    "model_name":            MODEL_NAME,
    "model_version":         MODEL_VERSION,
    "reference_date":        REFERENCE_DATE,
    "current_date":          CURRENT_DATE,
    "overall_traffic_light": summary["overall_traffic_light"],
    "psi_score":             float(summary["metrics"]["psi_score"]["value"]),
    "psi_traffic_light":     summary["metrics"]["psi_score"]["traffic_light"],
    "ae_ratio":              float(summary["metrics"]["ae_ratio"]["value"]),
    "ae_ci_lower":           float(summary["metrics"]["ae_ratio"]["ci_lower"]),
    "ae_ci_upper":           float(summary["metrics"]["ae_ratio"]["ci_upper"]),
    "ae_traffic_light":      summary["metrics"]["ae_ratio"]["traffic_light"],
    "gini_ref":              float(summary["metrics"]["gini"]["gini_ref"]),
    "gini_cur":              float(summary["metrics"]["gini"]["gini_cur"]),
    "gini_p_value":          float(summary["metrics"]["gini"]["p_value"]),
    "gini_traffic_light":    summary["metrics"]["gini"]["traffic_light"],
    "reference_n":           int(len(df_reference)),
    "current_n":             int(len(df_current)),
    "actual_claims":         int(actual_cur.sum()),
    "expected_claims":       float(expected_cur.sum()),
}

summary_df = spark.createDataFrame([Row(**summary_row)])

(summary_df
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["monitoring_log"]))

print(f"Monitoring summary written to {TABLES['monitoring_log']}")

# COMMAND ----------

# -----------------------------------------------------------------------
# Write CSI detail (one row per feature per run)
# -----------------------------------------------------------------------
csi_rows = []
for feature, result in csi_scores.items():
    csi_rows.append({
        "run_date":      RUN_DATE,
        "model_name":    MODEL_NAME,
        "current_date":  CURRENT_DATE,
        "feature":       feature,
        "csi":           float(result.csi),
        "traffic_light": result.traffic_light,
        "n_bins":        N_BINS,
    })

csi_df = spark.createDataFrame([Row(**r) for r in csi_rows])

(csi_df
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["csi_results"]))

print(f"CSI detail written to {TABLES['csi_results']}  ({len(csi_rows)} rows)")

# COMMAND ----------

# -----------------------------------------------------------------------
# Write A/E results (portfolio level)
# -----------------------------------------------------------------------
ae_rows = [{
    "run_date":      RUN_DATE,
    "model_name":    MODEL_NAME,
    "current_date":  CURRENT_DATE,
    "segment":       "portfolio",
    "segment_value": "all",
    "ae_ratio":      float(ae_portfolio.ratio),
    "ci_lower":      float(ae_portfolio.ci_lower),
    "ci_upper":      float(ae_portfolio.ci_upper),
    "actual":        float(actual_cur.sum()),
    "expected":      float(expected_cur.sum()),
    "traffic_light": ae_portfolio.traffic_light,
}]

ae_df = spark.createDataFrame([Row(**r) for r in ae_rows])

(ae_df
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["ae_results"]))

print(f"A/E results written to {TABLES['ae_results']}")

# COMMAND ----------

# -----------------------------------------------------------------------
# Set 7-year retention policy on the monitoring log
# -----------------------------------------------------------------------
spark.sql(f"""
    ALTER TABLE {TABLES["monitoring_log"]}
    SET TBLPROPERTIES (
        'delta.deletedFileRetentionDuration' = 'interval 7 years',
        'delta.logRetentionDuration'         = 'interval 7 years'
    )
""")
print("Retention policy set to 7 years on monitoring_log.")

# COMMAND ----------

# Query the monitoring trend (useful once you have multiple runs)
trend_query = f"""
SELECT
    current_date,
    ae_ratio,
    ae_ci_lower,
    ae_ci_upper,
    overall_traffic_light,
    psi_score,
    gini_cur,
    gini_p_value
FROM {TABLES["monitoring_log"]}
WHERE model_name = '{MODEL_NAME}'
ORDER BY current_date DESC
LIMIT 12
"""

trend_df = spark.sql(trend_query)
trend_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the complete monitoring workflow for a deployed pricing model:
# MAGIC
# MAGIC | Step | What it tests | Key output |
# MAGIC |------|---------------|-----------|
# MAGIC | PSI | Score distribution stability | Traffic light + bin breakdown |
# MAGIC | CSI | Feature distribution stability | Ranked feature table |
# MAGIC | A/E | Calibration (mean accuracy) | Ratio + 95% CI |
# MAGIC | Gini drift | Discrimination (ranking accuracy) | Z-stat + p-value |
# MAGIC | MonitoringReport | Combined interpretation | Overall traffic light |
# MAGIC
# MAGIC **What to do with the results:**
# MAGIC - All GREEN: log and move on
# MAGIC - Elevated PSI, green A/E, stable Gini: mix shift — no model action
# MAGIC - Elevated PSI + A/E, stable Gini: calibration problem — apply recalibration factor
# MAGIC - Green PSI, elevated A/E, falling Gini: concept drift — escalate and retrain
# MAGIC - Single high-CSI feature, else green: investigate the feature
# MAGIC
# MAGIC **Regulatory note:** The Delta monitoring log with 7-year retention satisfies the PRA
# MAGIC SS1/23 expectation that monitoring outcomes are recorded and available for review.
# MAGIC Use `VERSION AS OF` or `TIMESTAMP AS OF` queries for audit-point reproduction.
# MAGIC
# MAGIC **Next step:** Schedule this notebook as a Databricks Job (Part 11 of the tutorial)
# MAGIC to run automatically on the first Monday of each month.

# COMMAND ----------

print("Module 11 notebook complete.")
print(f"  Monitoring log:  {TABLES['monitoring_log']}")
print(f"  CSI detail:      {TABLES['csi_results']}")
print(f"  A/E detail:      {TABLES['ae_results']}")
print(f"  Overall status:  {summary['overall_traffic_light']}")
