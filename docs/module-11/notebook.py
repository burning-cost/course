# Databricks notebook source
# MAGIC %md
# MAGIC # Module 11: Model Monitoring and Drift Detection
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC A trained model is not a finished product. It is an assumption that the world it was
# MAGIC trained on still resembles the world it is scoring. This notebook shows you how to test
# MAGIC that assumption systematically, using the `insurance-monitoring` library.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates synthetic UK motor policy data with reference and current periods
# MAGIC 2. Trains a CatBoost frequency model on the reference period
# MAGIC 3. Simulates a drifted "current" dataset with shifted feature distributions
# MAGIC 4. Computes Population Stability Index (PSI) on predicted scores
# MAGIC 5. Computes Characteristic Stability Index (CSI) on all model features
# MAGIC 6. Computes Actual vs Expected ratios with Poisson confidence intervals
# MAGIC 7. Runs a Gini drift test
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
# MAGIC **Key concept:** PSI and CSI tell you what has changed. A/E and Gini tell you whether
# MAGIC it matters for pricing accuracy. Run them together. Act on the combination, not any
# MAGIC single metric in isolation.

# COMMAND ----------

%pip install insurance-monitoring catboost polars mlflow scikit-learn --quiet
# Use %pip in Databricks notebooks. Outside Databricks: uv add insurance-monitoring catboost polars mlflow scikit-learn

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
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score, roc_curve

from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi, csi
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

warnings.filterwarnings("ignore", category=UserWarning)

print(f"Today: {date.today()}")
print(f"Polars:              {pl.__version__}")
print("insurance-monitoring:", __import__("insurance_monitoring").__version__)
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
REFERENCE_DATE = "2023-12-31"
CURRENT_DATE   = "2024-06-30"

MODEL_NAME    = "motor_frequency_catboost"
MODEL_VERSION = "1"

N_BINS = 10  # PSI/CSI bin count. 10 is standard; reduce to 5 for sparse features

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

N_REF = 60_000
N_CUR = 20_000

FEATURE_NAMES = [
    "driver_age", "vehicle_age", "vehicle_group",
    "annual_mileage", "ncd_years", "region",
]
CAT_FEATURES = ["region"]


def generate_motor_data(n, period, rng, drift=False):
    if drift:
        driver_age = np.where(
            rng.random(n) < 0.25,
            rng.integers(17, 25, n),
            rng.integers(25, 80, n),
        )
        annual_mileage = rng.gamma(shape=4.5, scale=3200, size=n).astype(int).clip(2000, 50000)
    else:
        driver_age = np.where(
            rng.random(n) < 0.15,
            rng.integers(17, 25, n),
            rng.integers(25, 80, n),
        )
        annual_mileage = rng.gamma(shape=4.0, scale=3000, size=n).astype(int).clip(2000, 50000)

    vehicle_age   = rng.integers(0, 20, n)
    vehicle_group = rng.integers(1, 51, n)
    ncd_years     = rng.integers(0, 10, n)
    region        = rng.choice(
        ["North", "Midlands", "London", "South", "Scotland", "Wales"],
        size=n,
        p=[0.20, 0.20, 0.20, 0.20, 0.12, 0.08],
    )
    exposure = rng.uniform(0.25, 1.0, n)

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
        + rng.normal(0, 0.10, n)
    )

    if drift:
        log_rate += np.where(driver_age < 25, 0.25, 0.0)

    freq        = np.exp(log_rate)
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
# MAGIC We use CatBoost with a Poisson loss function. The exposure is passed as a
# MAGIC log-offset — not as a weight. The model is trained on the reference period only.

# COMMAND ----------

ref_pd = df_reference.to_pandas()

split   = int(0.8 * len(ref_pd))
X_train = ref_pd[:split][FEATURE_NAMES]
y_train = ref_pd[:split]["claim_count"]
e_train = ref_pd[:split]["exposure"]

X_val   = ref_pd[split:][FEATURE_NAMES]
y_val   = ref_pd[split:]["claim_count"]
e_val   = ref_pd[split:]["exposure"]

train_pool = Pool(
    X_train, y_train,
    cat_features=CAT_FEATURES,
    baseline=np.log(e_train.values),
)
val_pool = Pool(
    X_val, y_val,
    cat_features=CAT_FEATURES,
    baseline=np.log(e_val.values),
)

model = CatBoostRegressor(
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
# MAGIC We generate predicted frequencies for every policy in both the reference and current
# MAGIC windows. PSI runs on the raw frequency predictions. A/E runs on predicted claim counts
# MAGIC (frequency × exposure). Gini runs on the actual vs predicted comparison.

# COMMAND ----------

ref_pd = df_reference.to_pandas()
cur_pd = df_current.to_pandas()

ref_pool_pred = Pool(ref_pd[FEATURE_NAMES], cat_features=CAT_FEATURES)
cur_pool_pred = Pool(cur_pd[FEATURE_NAMES], cat_features=CAT_FEATURES)

pred_ref = model.predict(ref_pool_pred)
pred_cur = model.predict(cur_pool_pred)

exposure_ref = ref_pd["exposure"].values
exposure_cur = cur_pd["exposure"].values

actual_ref = ref_pd["claim_count"].values.astype(float)
actual_cur = cur_pd["claim_count"].values.astype(float)

print(f"Reference predictions: mean={pred_ref.mean():.4f}, "
      f"min={pred_ref.min():.4f}, max={pred_ref.max():.4f}")
print(f"Current predictions:   mean={pred_cur.mean():.4f}, "
      f"min={pred_cur.min():.4f}, max={pred_cur.max():.4f}")
print()
# AUC for information (not the primary monitoring metric — Gini is)
from sklearn.metrics import roc_auc_score
y_ref_binary = (actual_ref > 0).astype(int)
y_cur_binary = (actual_cur > 0).astype(int)
print(f"Reference AUC: {roc_auc_score(y_ref_binary, pred_ref):.4f}")
print(f"Current AUC:   {roc_auc_score(y_cur_binary, pred_cur):.4f}")

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
# MAGIC - PSI 0.10-0.25: moderate change, worth monitoring
# MAGIC - PSI > 0.25: significant change, investigate

# COMMAND ----------

from insurance_monitoring.drift import psi

psi_score = psi(
    reference=pred_ref,
    current=pred_cur,
    n_bins=N_BINS,
    exposure_weights=exposure_cur,
    reference_exposure=exposure_ref,
)

print(f"PSI (score distribution): {psi_score:.4f}")
if psi_score < 0.10:
    psi_band = "green"
elif psi_score < 0.25:
    psi_band = "amber"
else:
    psi_band = "red"
print(f"Traffic light:            {psi_band}")

# COMMAND ----------

# Visualise score distributions
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pred_ref, bins=50, alpha=0.6, label="Reference", color="steelblue", density=True)
ax.hist(pred_cur, bins=50, alpha=0.6, label="Current",   color="tomato",    density=True)
ax.axvline(np.median(pred_ref), color="steelblue", linestyle="--", alpha=0.7, label="Ref median")
ax.axvline(np.median(pred_cur), color="tomato",    linestyle="--", alpha=0.7, label="Cur median")
ax.set_xlabel("Predicted frequency (annualised)")
ax.set_ylabel("Density")
ax.set_title(f"Score distribution  (PSI = {psi_score:.3f}, {psi_band})")
ax.legend()
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
# MAGIC explained by the individual features. A high PSI with one high-CSI feature points
# MAGIC directly to the cause.

# COMMAND ----------

from insurance_monitoring.drift import csi

# csi() returns a Polars DataFrame with columns: feature, csi, band
csi_df = csi(
    reference_df=df_reference,
    current_df=df_current,
    features=FEATURE_NAMES,
    n_bins=N_BINS,
)

print(f"{'Feature':<25} {'CSI':>8}  {'Status'}")
print("-" * 50)
for row in csi_df.sort("csi", descending=True).iter_rows(named=True):
    flag = " <-- investigate" if row["csi"] > 0.25 else ""
    print(f"{row['feature']:<25} {row['csi']:>8.4f}  {row['band']}{flag}")

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
notable_features = csi_df.filter(pl.col("csi") > 0.10)["feature"].to_list()

if not notable_features:
    print("No features with CSI > 0.10. Book mix is stable.")
else:
    n_feat = len(notable_features)
    fig, axes = plt.subplots(1, n_feat, figsize=(6 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]

    for ax, feature in zip(axes, notable_features):
        ref_vals = df_reference[feature].to_numpy()
        cur_vals = df_current[feature].to_numpy()
        csi_val  = float(csi_df.filter(pl.col("feature") == feature)["csi"][0])

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
# MAGIC We use `ae_ratio_ci()` for the portfolio level — it returns the point estimate plus a
# MAGIC Poisson confidence interval (Garwood exact method). If 1.0 is inside the 95% CI, there
# MAGIC is no statistically significant evidence of a calibration problem.

# COMMAND ----------

from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

ae_result = ae_ratio_ci(
    actual=actual_cur,
    predicted=pred_cur,
    exposure=exposure_cur,
    alpha=0.05,
    method="poisson",
)

# ae_ratio_ci returns: {"ae": float, "lower": float, "upper": float, "n_claims": float, "n_expected": float}
ae_val    = ae_result["ae"]
ae_lower  = ae_result["lower"]
ae_upper  = ae_result["upper"]
n_claims  = ae_result["n_claims"]
n_expected = ae_result["n_expected"]

print(f"Portfolio A/E ratio: {ae_val:.4f}")
print(f"95% CI:              [{ae_lower:.4f}, {ae_upper:.4f}]")
print(f"Actual claims:       {n_claims:.0f}")
print(f"Expected claims:     {n_expected:.1f}")
print()
if ae_lower > 1.0:
    print("CI excludes 1.0 from below. Model is systematically under-predicting.")
elif ae_upper < 1.0:
    print("CI excludes 1.0 from above. Model is systematically over-predicting.")
else:
    print("CI contains 1.0. No statistically significant evidence of calibration drift.")

# COMMAND ----------

# Segment A/E breakdown by driver age band
df_cur_with_preds = df_current.with_columns([
    pl.Series("expected", pred_cur * exposure_cur),
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

    seg_result = ae_ratio_ci(
        actual=seg["claim_count"].to_numpy().astype(float),
        predicted=seg["expected"].to_numpy(),
        method="poisson",
    )
    # predicted here is already expected counts (freq * exposure), so pass exposure=None
    # but ae_ratio_ci with predicted = expected_counts and no exposure divides sum(actual)/sum(expected)
    print(f"{label:<15} {seg_result['ae']:>8.4f}  {seg_result['lower']:>10.4f}  "
          f"{seg_result['upper']:>10.4f}  {seg['claim_count'].sum():>8.0f}  "
          f"{seg['expected'].sum():>10.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: Gini drift test
# MAGIC
# MAGIC The Gini coefficient measures discrimination: does the model rank high-risk policies
# MAGIC above low-risk ones? A falling Gini means the model is less able to distinguish
# MAGIC claimants from non-claimants. This is concept drift.
# MAGIC
# MAGIC This is a different signal from A/E:
# MAGIC - A/E off, Gini OK: calibration problem. Apply a recalibration factor.
# MAGIC - Gini dropping, A/E may or may not be affected: discrimination has weakened. Retraining required.
# MAGIC - Both off: serious. Escalate immediately.

# COMMAND ----------

from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(actual_ref, pred_ref, exposure=exposure_ref)
gini_cur = gini_coefficient(actual_cur, pred_cur, exposure=exposure_cur)

# gini_drift_test returns a GiniDriftResult with fields:
#   reference_gini, current_gini, gini_change, z_statistic, p_value, significant
gini_result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    reference_exposure=exposure_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    current_exposure=exposure_cur,
    n_bootstrap=200,
)

print(f"Gini (reference): {gini_result.reference_gini:.4f}")
print(f"Gini (current):   {gini_result.current_gini:.4f}")
print(f"Change:           {gini_result.gini_change:+.4f}")
print(f"Z-statistic:      {gini_result.z_statistic:.4f}")
print(f"P-value:          {gini_result.p_value:.4f}")
print(f"Significant:      {gini_result.significant}")
print()
if gini_result.p_value < 0.05:
    drop = gini_ref - gini_cur
    if drop > 0.03:
        print("Statistically significant Gini drop >= 0.03. Discrimination has weakened. "
              "Retraining is required.")
    else:
        print("Statistically significant change but drop < 0.03. Monitor closely.")
else:
    print("No statistically significant Gini change. Discrimination is stable.")

# COMMAND ----------

# ROC curves for reference and current periods
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

from sklearn.metrics import roc_curve

fpr_ref, tpr_ref, _ = roc_curve(y_ref_binary, pred_ref)
axes[0].plot(fpr_ref, tpr_ref, color="steelblue", linewidth=2,
             label=f"Reference (Gini={gini_result.reference_gini:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC — Reference period")
axes[0].legend()

fpr_cur, tpr_cur, _ = roc_curve(y_cur_binary, pred_cur)
axes[1].plot(fpr_cur, tpr_cur, color="tomato", linewidth=2,
             label=f"Current (Gini={gini_result.current_gini:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC — Current period")
axes[1].legend()

plt.suptitle(
    f"Gini drift: {gini_result.reference_gini:.3f} → {gini_result.current_gini:.3f}  "
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
# MAGIC `MonitoringReport` is a dataclass that runs all checks at construction time. Pass your
# MAGIC arrays and feature DataFrames in; it immediately computes A/E, Gini drift, CSI, and
# MAGIC an optional Murphy decomposition. Access results via `results_`, `recommendation`,
# MAGIC `to_dict()`, or `to_polars()`.
# MAGIC
# MAGIC The `recommendation` property implements the three-stage decision tree from
# MAGIC arXiv 2510.04556:
# MAGIC - Gini OK + A/E OK → NO_ACTION
# MAGIC - A/E bad only → RECALIBRATE
# MAGIC - Gini bad → REFIT
# MAGIC - Multiple conflicting signals → INVESTIGATE

# COMMAND ----------

from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=df_reference,
    feature_df_current=df_current,
    features=FEATURE_NAMES,
    murphy_distribution="poisson",   # Murphy decomposition: sharpens RECALIBRATE vs REFIT
    gini_bootstrap=False,            # set True to add percentile CIs on Gini (slower)
)

# COMMAND ----------

# MonitoringReport computes everything in __post_init__.
# results_ is a dict; recommendation is the decision string.
print("Recommendation:", report.recommendation)
print()

results = report.results_

ae_res   = results["ae_ratio"]
gini_res = results["gini"]

print(f"A/E ratio:    {ae_res['value']:.4f}  (CI: [{ae_res['lower_ci']:.4f}, {ae_res['upper_ci']:.4f}])  [{ae_res['band']}]")
print(f"Gini current: {gini_res['current']:.4f}  (ref: {gini_res['reference']:.4f})  [{gini_res['band']}]")
print(f"Gini p-value: {gini_res['p_value']:.4f}")
print()

if "max_csi" in results:
    mc = results["max_csi"]
    print(f"Max CSI:      {mc['value']:.4f}  ({mc['worst_feature']})  [{mc['band']}]")

if report.murphy_available:
    m = results["murphy"]
    print()
    print(f"Murphy discrimination %:  {m['discrimination_pct']:.1f}%")
    print(f"Murphy miscalibration %:  {m['miscalibration_pct']:.1f}%")
    print(f"Murphy verdict:           {m['verdict']}")

# COMMAND ----------

# Flat Polars DataFrame version — one row per metric
print("\nMonitoring results (flat):")
print(report.to_polars())

# COMMAND ----------

# Full dict — suitable for JSON serialisation and Delta write
report_dict = report.to_dict()
report_json = json.dumps(report_dict, indent=2, default=str)
report_path = f"/tmp/monitoring_report_{CURRENT_DATE}.json"
with open(report_path, "w") as f:
    f.write(report_json)
print(f"Report saved to {report_path}")
print()
print("Report JSON (first 600 chars):")
print(report_json[:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: Interpret the signals
# MAGIC
# MAGIC The five scenarios to recognise:
# MAGIC 1. All green: log and move on
# MAGIC 2. Elevated PSI, green A/E, stable Gini: mix shift absorbed by model — no action
# MAGIC 3. Elevated PSI, elevated A/E, stable Gini: calibration problem — apply recalibration factor
# MAGIC 4. Green PSI, elevated A/E, falling Gini: concept drift — escalate, begin retraining
# MAGIC 5. Elevated CSI on one feature, all else green: investigate the feature, no immediate model action

# COMMAND ----------

def interpret_monitoring_signals(report):
    """Print a structured interpretation of monitoring signals."""
    recommendation = report.recommendation
    results        = report.results_

    ae_band   = results["ae_ratio"]["band"]
    ae_val    = results["ae_ratio"]["value"]
    gini_band = results["gini"]["band"]
    gini_p    = results["gini"]["p_value"]
    gini_drop = results["gini"]["reference"] - results["gini"]["current"]

    red_csi_features = []
    if "csi" in results:
        red_csi_features = [r["feature"] for r in results["csi"] if r["band"] == "red"]

    print(f"RECOMMENDATION: {recommendation}")
    print("=" * 60)

    if recommendation == "NO_ACTION":
        print("All metrics within tolerance. No model action required.")
        print("Log this result and file as evidence of ongoing monitoring.")
        return

    if recommendation == "REFIT":
        print("PATTERN: Gini has degraded. Discrimination has weakened.")
        print("The model is less able to separate high-risk from low-risk policies.")
        print("A recalibration factor will fix the average but not the ranking.")
        print("ACTION: Escalate to head of pricing. Begin retraining on recent data.")
        print("Apply a temporary recalibration factor (1/A/E) as a holding measure.")
        return

    if recommendation == "RECALIBRATE":
        print("PATTERN: A/E shifted, Gini stable.")
        print(f"Model is {'under' if ae_val > 1.0 else 'over'}-predicting overall "
              f"(A/E={ae_val:.3f}) but discrimination is intact.")
        print(f"ACTION: Apply a recalibration factor of {1/ae_val:.3f} to all predictions.")
        print("Investigate root cause (mix shift, inflation, underwriting rule change).")
        return

    if recommendation == "INVESTIGATE":
        print("PATTERN: Multiple signals in conflict. Manual review required.")
        print(f"  A/E band: {ae_band},  Gini band: {gini_band}")
        print("ACTION: Bring to pricing committee with the full monitoring pack.")
        return

    if recommendation == "MONITOR_CLOSELY":
        print("PATTERN: Amber signals but no red. Watch the trend.")
        print("No immediate model action required.")
        if red_csi_features:
            print(f"Feature(s) with elevated CSI: {red_csi_features}")
            print("Investigate whether the shift is data quality or genuine book change.")
        return

    print(f"Combined signals: A/E={ae_band}, Gini={gini_band}")
    print("Review individual metrics above.")


interpret_monitoring_signals(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 10: Write monitoring results to Delta tables
# MAGIC
# MAGIC Three tables:
# MAGIC - `monitoring_log` — one row per monitoring run (headline summary)
# MAGIC - `csi_results` — one row per feature per run (for trend analysis)
# MAGIC - `ae_results` — one row per segment per run
# MAGIC
# MAGIC We set a 7-year retention policy on the monitoring log. FCA record-keeping requirements
# MAGIC are broadly aligned with this window.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

from pyspark.sql import Row

# Write the monitoring summary (one row per run)
summary_row = {
    "run_date":              RUN_DATE,
    "model_name":            MODEL_NAME,
    "model_version":         MODEL_VERSION,
    "reference_date":        REFERENCE_DATE,
    "current_date":          CURRENT_DATE,
    "recommendation":        report.recommendation,
    "ae_ratio":              float(results["ae_ratio"]["value"]),
    "ae_ci_lower":           float(results["ae_ratio"]["lower_ci"]),
    "ae_ci_upper":           float(results["ae_ratio"]["upper_ci"]),
    "ae_band":               results["ae_ratio"]["band"],
    "gini_ref":              float(results["gini"]["reference"]),
    "gini_cur":              float(results["gini"]["current"]),
    "gini_p_value":          float(results["gini"]["p_value"]),
    "gini_band":             results["gini"]["band"],
    "psi_score":             float(psi_score),
    "psi_band":              psi_band,
    "reference_n":           int(len(df_reference)),
    "current_n":             int(len(df_current)),
    "actual_claims":         int(actual_cur.sum()),
    "expected_claims":       float((pred_cur * exposure_cur).sum()),
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

# Write CSI detail (one row per feature per run)
csi_rows = []
for row in csi_df.iter_rows(named=True):
    csi_rows.append({
        "run_date":      RUN_DATE,
        "model_name":    MODEL_NAME,
        "current_date":  CURRENT_DATE,
        "feature":       row["feature"],
        "csi":           float(row["csi"]),
        "band":          row["band"],
        "n_bins":        N_BINS,
    })

csi_spark = spark.createDataFrame([Row(**r) for r in csi_rows])

(csi_spark
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["csi_results"]))

print(f"CSI detail written to {TABLES['csi_results']}  ({len(csi_rows)} rows)")

# COMMAND ----------

# Write A/E results (portfolio level)
ae_rows = [{
    "run_date":      RUN_DATE,
    "model_name":    MODEL_NAME,
    "current_date":  CURRENT_DATE,
    "segment":       "portfolio",
    "segment_value": "all",
    "ae_ratio":      float(results["ae_ratio"]["value"]),
    "ci_lower":      float(results["ae_ratio"]["lower_ci"]),
    "ci_upper":      float(results["ae_ratio"]["upper_ci"]),
    "actual":        float(actual_cur.sum()),
    "expected":      float((pred_cur * exposure_cur).sum()),
    "band":          results["ae_ratio"]["band"],
}]

ae_spark = spark.createDataFrame([Row(**r) for r in ae_rows])

(ae_spark
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["ae_results"]))

print(f"A/E results written to {TABLES['ae_results']}")

# COMMAND ----------

# Set 7-year retention policy on the monitoring log
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
    recommendation,
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
# MAGIC | PSI (`psi()`) | Score distribution stability | Float (< 0.10 green, > 0.25 red) |
# MAGIC | CSI (`csi()`) | Feature distribution stability | Polars DataFrame, one row per feature |
# MAGIC | A/E (`ae_ratio_ci()`) | Calibration (mean accuracy) | Dict: ae, lower, upper, n_claims |
# MAGIC | Gini drift (`gini_drift_test()`) | Discrimination (ranking accuracy) | GiniDriftResult: z_statistic, p_value |
# MAGIC | `MonitoringReport` | Combined interpretation | recommendation, results_, to_polars() |
# MAGIC
# MAGIC **What to do with the results:**
# MAGIC - All green: log and move on
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
print(f"  Recommendation:  {report.recommendation}")
