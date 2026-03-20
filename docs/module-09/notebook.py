# Databricks notebook source
# MAGIC %md
# MAGIC # Module 9: Demand Modelling and Price Elasticity
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC The risk model tells you what a policy costs. It says nothing about whether
# MAGIC the customer will buy. This notebook builds the missing piece: a causal demand model
# MAGIC that estimates how price changes affect conversion and renewal behaviour.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates synthetic conversion and renewal datasets using built-in generators
# MAGIC 2. Builds a conversion model (logistic and CatBoost) with one-way diagnostics
# MAGIC 3. Builds a retention model with price sensitivity analysis
# MAGIC 4. Runs the near-deterministic price diagnostic before fitting any elasticity model
# MAGIC 5. Estimates causal price elasticity using Double Machine Learning (DML)
# MAGIC 6. Estimates heterogeneous CATE per customer using CausalForestDML
# MAGIC 7. Visualises the elasticity surface across NCD and age dimensions
# MAGIC 8. Builds a portfolio demand curve showing the volume-profit trade-off
# MAGIC 9. Runs per-policy profit-maximising optimisation subject to the FCA ENBP ceiling
# MAGIC 10. Produces the ENBP compliance audit trail
# MAGIC 11. Writes results to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 30-45 minutes on a 4-core cluster (CausalForestDML takes 5-8 minutes).
# MAGIC
# MAGIC **Prerequisites:** No prior module required — this notebook generates its own data.
# MAGIC
# MAGIC **Key concept:** The naive regression of conversion on price is biased. Risk composition
# MAGIC confounds the price effect: higher-risk customers receive higher quoted prices AND lapse more.
# MAGIC OLS attributes some of the risk-driven lapse to price sensitivity. DML corrects this by
# MAGIC residualising both outcome and treatment on the same set of confounders before estimating
# MAGIC the price coefficient.

# COMMAND ----------

%pip install insurance-optimise insurance-causal catboost econml polars --quiet
# Both commands must be in the same cell so the restart follows the install.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import date

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_auc_score

from insurance_optimise.demand import ConversionModel, RetentionModel, ElasticityEstimator
from insurance_optimise.demand import DemandCurve, OptimalPrice
from insurance_optimise.demand.datasets import generate_conversion_data, generate_retention_data
from insurance_optimise.demand.compliance import ENBPChecker

from insurance_causal.elasticity.data import make_renewal_data
from insurance_causal.elasticity.fit import RenewalElasticityEstimator
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics
from insurance_causal.elasticity.surface import ElasticitySurface
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve

print(f"Today: {date.today()}")
print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print("All imports: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic datasets
# MAGIC
# MAGIC Two datasets with known true elasticities — essential for verifying that
# MAGIC DML recovers the right answer. In production you replace these generators
# MAGIC with a spark.table() call on your quote or renewal portfolio.
# MAGIC
# MAGIC The conversion dataset: 150,000 new business quotes.
# MAGIC True price elasticity embedded in the DGP: -2.0.
# MAGIC Conversion rate ~12%, consistent with a UK PCW motor book.
# MAGIC
# MAGIC The renewal dataset: 50,000 renewal observations.
# MAGIC True elasticity varies by NCD band (NCD=0: -3.5, NCD=5: -1.0).
# MAGIC The heterogeneity is what CausalForestDML is designed to recover.

# COMMAND ----------

# New business conversion quotes
df_quotes = generate_conversion_data(n_quotes=150_000, seed=42)

print("=== Conversion dataset ===")
print(f"Shape:            {df_quotes.shape}")
print(f"Conversion rate:  {df_quotes['converted'].mean():.3f}")
print(f"True elasticity:  {df_quotes['true_elasticity'].mean():.3f}")
print(f"Channels:         {df_quotes['channel'].unique().sort().to_list()}")
print()
print("Price ratio stats:")
print(df_quotes.select(["price_ratio", "log_price_ratio"]).describe())

# COMMAND ----------

# Renewal dataset
df_renewals = make_renewal_data(n=50_000, seed=42)

print("=== Renewal dataset ===")
print(f"Shape:          {df_renewals.shape}")
print(f"Renewal rate:   {df_renewals['renewed'].mean():.3f}")
print(f"True ATE:       {df_renewals['true_elasticity'].mean():.3f}")
print()
print("Log price change distribution:")
print(df_renewals.select("log_price_change").describe())

# COMMAND ----------

print("Conversion columns:", df_quotes.columns)
print()
print("Renewal columns:", df_renewals.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Conversion model
# MAGIC
# MAGIC The conversion model predicts purchase probability given features and price.
# MAGIC Start with logistic regression to get interpretable coefficients and a quick
# MAGIC sanity check. Then upgrade to CatBoost for predictive accuracy.
# MAGIC
# MAGIC The key diagnostic is the one-way plot: observed vs fitted conversion rate
# MAGIC by each rating factor. Gaps indicate model misspecification or missing interactions.
# MAGIC
# MAGIC Do not confuse the logistic coefficient on log_price_ratio with the causal
# MAGIC elasticity. It is biased upwards (more negative) due to risk composition
# MAGIC confounding. DML in Part 4 corrects that bias.

# COMMAND ----------

conv_logistic = ConversionModel(
    base_estimator="logistic",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
    logistic_C=1.0,
)
conv_logistic.fit(df_quotes)

print(conv_logistic.summary())

# COMMAND ----------

conv_probs = conv_logistic.predict_proba(df_quotes)
print(f"Mean predicted conversion:  {conv_probs.mean():.3f}")
print(f"Mean observed conversion:   {df_quotes['converted'].mean():.3f}")

# COMMAND ----------

# One-way diagnostics: the gate before using the model
channel_ow = conv_logistic.oneway(df_quotes, "channel")
print("One-way by channel:")
print(channel_ow.to_string())

# COMMAND ----------

ncd_ow = conv_logistic.oneway(df_quotes, "ncd_years")
print("One-way by NCD years:")
print(ncd_ow.to_string())

# COMMAND ----------

age_ow = conv_logistic.oneway(df_quotes, "age", bins=10)
print("One-way by age decile:")
print(age_ow.to_string())

# COMMAND ----------

# CatBoost conversion model: better predictive accuracy, same API
conv_catboost = ConversionModel(
    base_estimator="catboost",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
    cat_features=["area", "channel"],
)
conv_catboost.fit(df_quotes)

print("CatBoost feature importances:")
print(conv_catboost.summary())

# COMMAND ----------

# Compare AUC: logistic vs CatBoost
y_true = df_quotes["converted"].to_numpy()
auc_logistic = roc_auc_score(y_true, conv_logistic.predict_proba(df_quotes).to_numpy())
auc_catboost = roc_auc_score(y_true, conv_catboost.predict_proba(df_quotes).to_numpy())

print(f"Logistic AUC: {auc_logistic:.4f}")
print(f"CatBoost AUC: {auc_catboost:.4f}")

# COMMAND ----------

# Quantify the logistic bias: naive coefficient vs true elasticity
true_elas = df_quotes["true_elasticity"].mean()
logistic_summary = conv_logistic.summary()
price_coef = logistic_summary.loc[logistic_summary["feature"] == "log_price_ratio", "coefficient"].values[0]

print(f"True elasticity (DGP):      {true_elas:.3f}")
print(f"Logistic price coefficient: {price_coef:.3f}")
print(f"Bias: {abs(price_coef - true_elas):.3f}")
print()
print("The logistic model is too negative because it cannot distinguish the price")
print("effect from the risk-composition effect. DML corrects this in Part 4.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Retention (renewal) model
# MAGIC
# MAGIC For renewals the treatment variable is the price change from last year, not
# MAGIC the absolute price. A customer paying £600 who receives a £630 offer reacts
# MAGIC to a 5% increase. The standard encoding is log(renewal_price / prior_year_price).
# MAGIC
# MAGIC The logistic retention model is the right default: interpretable, auditable,
# MAGIC understood by governance. Use CatBoost retention only when one-way diagnostics
# MAGIC show persistent lift problems across multiple segments.

# COMMAND ----------

# The make_renewal_data dataset uses 'renewed'. RetentionModel expects 'lapsed'.
df_renewals_with_lapsed = df_renewals.with_columns(
    (1 - pl.col("renewed")).alias("lapsed")
)

retention_model = RetentionModel(
    model_type="logistic",
    outcome_col="lapsed",
    price_change_col="log_price_change",
    feature_cols=["tenure_years", "ncd_years", "payment_method",
                  "age", "channel", "region"],
    cat_features=["payment_method", "channel", "region"],
)
retention_model.fit(df_renewals_with_lapsed)

lapse_probs = retention_model.predict_proba(df_renewals_with_lapsed)
print(f"Mean predicted lapse rate: {lapse_probs.mean():.3f}")
print(f"Mean observed lapse rate:  {df_renewals_with_lapsed['lapsed'].mean():.3f}")

# COMMAND ----------

print(retention_model.summary())

# COMMAND ----------

# Renewal probability is the complement of lapse probability
renewal_probs = retention_model.predict_renewal_proba(df_renewals_with_lapsed)
print(f"Mean predicted renewal prob: {renewal_probs.mean():.3f}")
print(f"Observed renewal rate:       {df_renewals['renewed'].mean():.3f}")

# COMMAND ----------

# Price sensitivity by channel: the segmentation that matters most commercially
sensitivity = retention_model.price_sensitivity(df_renewals_with_lapsed)
print(f"Mean price sensitivity: {sensitivity.mean():.4f}")
print("(negative = higher price increase causes more lapse)")

sensitivity_pl = (
    df_renewals
    .select(["channel", "ncd_years"])
    .with_columns(pl.Series("sensitivity", sensitivity.to_numpy()))
    .group_by("channel")
    .agg(pl.col("sensitivity").mean().alias("mean_sensitivity"))
    .sort("mean_sensitivity")
)

print("\nMean price sensitivity by channel:")
print(sensitivity_pl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Near-deterministic price diagnostic
# MAGIC
# MAGIC This check is mandatory before fitting any DML elasticity model.
# MAGIC
# MAGIC DML identifies the price effect by residualising: fit a model of price change
# MAGIC on confounders, take the residuals, use them for identification. If the price
# MAGIC change is almost entirely predictable from risk features, the residuals have
# MAGIC near-zero variance. The resulting elasticity estimate is meaningless — the
# MAGIC confidence intervals blow up and the point estimate is driven by noise.
# MAGIC
# MAGIC The diagnostic computes Var(D_tilde) / Var(D): the fraction of price change
# MAGIC variance that survives conditioning on observables. Below 10%: do not proceed.
# MAGIC Above 10%: proceed with caution and document the fraction as a modelling assumption.
# MAGIC
# MAGIC If your real data fails this check, the remedies are: run a randomised A/B price
# MAGIC test, use panel data with within-customer variation, or exploit quasi-experiments
# MAGIC from bulk re-rating events.

# COMMAND ----------

confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df_renewals,
    treatment="log_price_change",
    confounders=confounders,
)
print(report.summary())

# COMMAND ----------

# Simulate what a near-deterministic price dataset looks like
df_ndp = make_renewal_data(n=50_000, seed=42, near_deterministic=True)

report_ndp = diag.treatment_variation_report(
    df_ndp,
    treatment="log_price_change",
    confounders=confounders,
)
print("Near-deterministic example (what to avoid):")
print(report_ndp.summary())

# COMMAND ----------

# Calibration check: renewal rate should fall monotonically as price change increases
# If there is no monotone pattern, confounding is severe.
cal_summary = diag.calibration_summary(
    df_renewals,
    outcome="renewed",
    treatment="log_price_change",
    n_bins=10,
)
print("Renewal rate by decile of price change (should be monotonically decreasing):")
print(cal_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. DML elasticity estimation (conversion)
# MAGIC
# MAGIC The PLR (Partially Linear Regression) estimator is the simplest DML variant.
# MAGIC It runs OLS of the outcome residual on the treatment residual after cross-fitting
# MAGIC nuisance models for both the outcome and treatment.
# MAGIC
# MAGIC The coefficient is on the probability scale, not the logit scale: a linear
# MAGIC probability model coefficient. For a 10% price increase (log change ~0.095),
# MAGIC the change in conversion probability is elasticity × 0.095.
# MAGIC
# MAGIC This cell takes 3-5 minutes. The 5-fold cross-fitting runs CatBoost nuisance
# MAGIC models 10 times (5 outcome × 5 treatment). Do not interrupt it.

# COMMAND ----------

est_conversion = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    outcome_model="catboost",
    treatment_model="catboost",
    heterogeneous=False,
)

print("Fitting DML elasticity estimator... (3-5 minutes)")
est_conversion.fit(df_quotes)

print("\n=== DML Elasticity Summary ===")
print(est_conversion.summary())

# COMMAND ----------

# Compare DML estimate to naive logistic and to ground truth
print(f"True elasticity (DGP):      {df_quotes['true_elasticity'].mean():.3f}")
print(f"DML estimate:               {est_conversion.elasticity_:.3f}")
print(f"DML 95% CI:                 [{est_conversion.elasticity_ci_[0]:.3f}, {est_conversion.elasticity_ci_[1]:.3f}]")

logistic_summary = conv_logistic.summary()
naive_coef = logistic_summary.loc[logistic_summary["feature"] == "log_price_ratio", "coefficient"].values[0]
print(f"Naive logistic coefficient: {naive_coef:.3f}")
print(f"Bias in naive estimate:     {abs(naive_coef - df_quotes['true_elasticity'].mean()):.3f}")

# COMMAND ----------

# Sensitivity analysis: how robust is the estimate to unobserved confounding?
sensitivity = est_conversion.sensitivity_analysis()
if sensitivity is not None:
    print("Sensitivity to unobserved confounding:")
    print(sensitivity)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Heterogeneous elasticity with CausalForestDML
# MAGIC
# MAGIC The global ATE from Part 5 tells you the portfolio average. But pricing decisions
# MAGIC are per-customer. CausalForestDML estimates individual CATE values — each customer's
# MAGIC own price sensitivity — using a non-parametric forest method.
# MAGIC
# MAGIC Commercial question answered: are NCD-0 customers more elastic than NCD-5?
# MAGIC Are PCW customers more elastic than direct? The GATE (Group Average Treatment Effect)
# MAGIC table answers both.
# MAGIC
# MAGIC This cell takes 5-8 minutes. The forest trains 200 trees in 5-fold cross-fitting.
# MAGIC Read the demand curve section (Part 7) while it trains.

# COMMAND ----------

est_renewal = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
)

print("Fitting CausalForestDML... (5-8 minutes on Databricks Free Edition)")
est_renewal.fit(
    df_renewals,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)

ate, lb, ub = est_renewal.ate()
print(f"\nATE:      {ate:.3f}")
print(f"95% CI:   [{lb:.3f}, {ub:.3f}]")
print(f"True ATE: {df_renewals['true_elasticity'].mean():.3f}")

# COMMAND ----------

# Per-customer CATE estimates
cate_values = est_renewal.cate(df_renewals)

print("CATE distribution:")
print(f"  Mean:    {cate_values.mean():.3f}")
print(f"  Std:     {cate_values.std():.3f}")
print(f"  Min:     {cate_values.min():.3f}")
print(f"  Max:     {cate_values.max():.3f}")
print()
print("Customers with CATE < -3.0 are highly elastic: small price increases cause")
print("large drops in renewal probability. Price these carefully.")
print("Customers with CATE > -1.0 are inelastic: safe to price towards the ENBP ceiling.")

# COMMAND ----------

# Group average treatment effects (GATE) by segment
gate_ncd = est_renewal.gate(df_renewals, by="ncd_years")
print("GATE by NCD years:")
print(gate_ncd)

# COMMAND ----------

gate_channel = est_renewal.gate(df_renewals, by="channel")
print("\nGATE by channel:")
print(gate_channel)

# COMMAND ----------

gate_age = est_renewal.gate(df_renewals, by="age_band")
print("\nGATE by age band:")
print(gate_age)

# COMMAND ----------

# Validation: compare recovered GATEs to known true values from the DGP
true_by_ncd = {0: -3.5, 1: -3.0, 2: -2.5, 3: -2.0, 4: -1.5, 5: -1.0}

print("Validation: recovered vs true elasticity by NCD")
print(f"{'NCD':>6} {'True':>8} {'Recovered':>12} {'Lower':>8} {'Upper':>8}")
for row in gate_ncd.iter_rows(named=True):
    ncd = row["ncd_years"]
    true_val = true_by_ncd.get(ncd, float("nan"))
    print(f"{ncd:>6} {true_val:>8.1f} {row['elasticity']:>12.3f} {row['ci_lower']:>8.3f} {row['ci_upper']:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Elasticity surface
# MAGIC
# MAGIC The surface is the deliverable for the pricing committee. It shows how price
# MAGIC sensitivity varies across two dimensions simultaneously.
# MAGIC
# MAGIC Dark red cells (more negative elasticity): elastic customers who react strongly
# MAGIC to price changes. Price these carefully; they are the ones most likely to lapse
# MAGIC following a renewal increase.
# MAGIC
# MAGIC Light cells (near-zero elasticity): inelastic customers. These are the safe
# MAGIC candidates for ENBP ceiling pricing — they will renew regardless of small
# MAGIC price changes.

# COMMAND ----------

surface = ElasticitySurface(est_renewal)

# Heatmap: NCD years x age band — the two dimensions that drive heterogeneity
fig_surface = surface.plot_surface(df_renewals, dims=["ncd_years", "age_band"])
fig_surface.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# Bar chart: average elasticity by channel
fig_gate_channel = surface.plot_gate(df_renewals, by="channel")
fig_gate_channel.savefig("/tmp/gate_by_channel.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Portfolio demand curve
# MAGIC
# MAGIC The demand curve translates the CATE estimates into a concrete view of the
# MAGIC volume-profit trade-off across a range of uniform price changes. It answers
# MAGIC the commercial director's question: "What happens to volume and profit if we
# MAGIC raise rates by X%?"
# MAGIC
# MAGIC The profit-maximising price change is where expected profit per policy peaks —
# MAGIC not necessarily at 0%. The demand curve makes this visible before you commit
# MAGIC to a rate action.

# COMMAND ----------

# Sweep price changes from -25% to +25% in 50 steps
demand_df = demand_curve(
    est_renewal,
    df_renewals,
    price_range=(-0.25, 0.25, 50),
)

print("Demand curve (±15% range shown):")
print(
    demand_df.select([
        "pct_price_change",
        "predicted_renewal_rate",
        "predicted_profit",
    ]).filter(
        pl.col("pct_price_change").is_between(-0.15, 0.15)
    ).with_columns(
        (pl.col("pct_price_change") * 100).round(1).alias("pct_change_%"),
        (pl.col("predicted_renewal_rate") * 100).round(2).alias("renewal_rate_%"),
        pl.col("predicted_profit").round(2),
    ).select(["pct_change_%", "renewal_rate_%", "predicted_profit"])
)

# COMMAND ----------

# Find the profit-maximising price change
max_profit_row = demand_df.sort("predicted_profit", descending=True).row(0, named=True)
print(f"Profit-maximising price change: {max_profit_row['pct_price_change']*100:.1f}%")
print(f"Expected renewal rate:          {max_profit_row['predicted_renewal_rate']*100:.1f}%")
print(f"Expected profit per policy:     £{max_profit_row['predicted_profit']:.2f}")

# COMMAND ----------

fig_demand = plot_demand_curve(demand_df, show_profit=True)
fig_demand.savefig("/tmp/demand_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Per-policy renewal pricing optimisation
# MAGIC
# MAGIC The demand curve gives the portfolio-level optimum. The RenewalPricingOptimiser
# MAGIC finds the per-customer optimal price subject to:
# MAGIC - A floor at the technical premium (do not price below cost)
# MAGIC - A ceiling at the ENBP (FCA PS21/5 hard requirement)
# MAGIC
# MAGIC The optimiser uses each customer's CATE as their individual price sensitivity.
# MAGIC Inelastic customers (small negative CATE) get prices near the ENBP ceiling.
# MAGIC Elastic customers (large negative CATE) get lower prices to preserve renewal.
# MAGIC
# MAGIC The `enbp_headroom` column in the output measures how far below the ENBP ceiling
# MAGIC the optimal price sits. Zero headroom means the regulation is binding: the
# MAGIC unconstrained profit-maximising price is above ENBP, but we cannot charge it.

# COMMAND ----------

opt = RenewalPricingOptimiser(
    est_renewal,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,  # do not price below technical premium
)

priced_df = opt.optimise(df_renewals, objective="profit")

print("Optimisation results:")
print(f"Policies optimised:       {len(priced_df):,}")
print(f"Mean optimal price:       £{priced_df['optimal_price'].mean():.2f}")
print(f"Mean last premium:        £{priced_df['last_premium'].mean():.2f}")
print(f"Mean price change:        {((priced_df['optimal_price'] / priced_df['last_premium']).mean() - 1) * 100:.1f}%")
print(f"Mean ENBP headroom:       £{priced_df['enbp_headroom'].mean():.2f}")
print(f"Predicted renewal rate:   {priced_df['predicted_renewal_prob'].mean():.3f}")
print(f"Expected profit/policy:   £{priced_df['expected_profit'].mean():.2f}")

# COMMAND ----------

# How often is the ENBP constraint binding?
binding = (priced_df["enbp_headroom"] < 1.0).sum()
total   = len(priced_df)
print(f"ENBP constraint binding: {binding:,} of {total:,} policies ({100*binding/total:.1f}%)")
print()
print("A high binding rate means the profitable action would be to charge more,")
print("but regulation prevents it. This is the quantified cost of PS21/5.")

# COMMAND ----------

# Visualise optimal price distribution and expected profit by NCD
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: distribution of optimal prices relative to ENBP
ratio = priced_df["optimal_price"] / priced_df["enbp"]
axes[0].hist(ratio.to_numpy(), bins=40, color="#1f77b4", alpha=0.7)
axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="ENBP ceiling")
axes[0].set_xlabel("Optimal price / ENBP")
axes[0].set_ylabel("Count")
axes[0].set_title("Optimal price as fraction of ENBP ceiling")
axes[0].legend()

# Right: expected profit by NCD band
ncd_profit = (
    priced_df.group_by("ncd_years")
    .agg(pl.col("expected_profit").mean().alias("mean_profit"))
    .sort("ncd_years")
)
axes[1].bar(
    ncd_profit["ncd_years"].to_numpy(),
    ncd_profit["mean_profit"].to_numpy(),
    color="#2ca02c", alpha=0.8,
)
axes[1].set_xlabel("NCD years")
axes[1].set_ylabel("Expected profit per policy (£)")
axes[1].set_title("Expected profit by NCD band")

plt.tight_layout()
plt.savefig("/tmp/optimisation_results.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. ENBP compliance audit
# MAGIC
# MAGIC Every renewal pricing run must produce a per-policy compliance audit before
# MAGIC prices are issued. "On average" is not sufficient — the FCA can ask for a
# MAGIC per-policy breakdown. This is the evidence for a section 166 review.
# MAGIC
# MAGIC The optimiser enforces ENBP as a hard ceiling, so there should be zero breaches.
# MAGIC Any breach indicates a bug in the optimiser or a data quality problem (ENBP
# MAGIC below the technical premium floor). Investigate before proceeding.

# COMMAND ----------

audit = opt.enbp_audit(priced_df)

n_breaches = int((audit["compliant"] == False).sum())
print("ENBP audit results:")
print(f"  Total policies: {len(audit):,}")
print(f"  Compliant:      {(audit['compliant']).sum():,}")
print(f"  Breaches:       {n_breaches:,}")
print(f"  Breach rate:    {100 * n_breaches / len(audit):.2f}%")

if n_breaches == 0:
    print("\nALL POLICIES COMPLIANT with FCA ICOBS 6B.2")
else:
    print("\nWARNING: Review breach detail before issuing prices")
    print(audit.filter(pl.col("compliant") == False).head(10))

# COMMAND ----------

# Audit table schema: what gets written to the regulatory archive
print("Audit table schema:")
for col in audit.columns:
    print(f"  {col}: {audit[col].dtype}")

# COMMAND ----------

# ENBPChecker from insurance_optimise.demand: compliance officer view
# This is a higher-level wrapper producing summary reports.
df_renewals_for_checker = df_renewals.rename({
    "enbp": "nb_equivalent_price",
}).with_columns(
    (pl.col("last_premium") * pl.col("log_price_change").exp()).alias("renewal_price")
)

checker = ENBPChecker(tolerance=0.0)
try:
    compliance_report = checker.check(df_renewals_for_checker)
    print(f"ENBPChecker — Breaches detected: {compliance_report.n_breaches}")
    print(f"By channel: {compliance_report.by_channel}")
except (KeyError, ValueError) as e:
    # Column names must match exactly: policy_id, renewal_price, nb_equivalent_price,
    # lapsed, tenure_years. Adjust if your data uses different names.
    print(f"Schema error (adjust column names to match ENBPChecker requirements): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Write results to Unity Catalog
# MAGIC
# MAGIC Write three artefacts:
# MAGIC 1. Per-policy optimal prices and ENBP compliance status (regulatory archive)
# MAGIC 2. Demand curve data (pricing committee presentation)
# MAGIC 3. CATE distribution summary (model monitoring baseline)
# MAGIC
# MAGIC The compliance table uses append mode — every run adds one batch of records.
# MAGIC This gives a permanent per-policy audit trail tied to the run date.

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"

# COMMAND ----------

with mlflow.start_run(run_name="demand_elasticity_m09"):
    mlflow.log_param("conversion_n",         150_000)
    mlflow.log_param("renewal_n",            50_000)
    mlflow.log_param("dml_n_folds",          5)
    mlflow.log_param("causal_forest_trees",  200)
    mlflow.log_metric("conversion_ate",      float(est_conversion.elasticity_))
    mlflow.log_metric("renewal_ate",         float(ate))
    mlflow.log_metric("mean_cate",           float(cate_values.mean()))
    mlflow.log_metric("std_cate",            float(cate_values.std()))
    mlflow.log_metric("enbp_binding_pct",    float(100 * binding / total))
    mlflow.log_metric("enbp_breach_count",   float(n_breaches))
    run_id = mlflow.active_run().info.run_id

print(f"MLflow run: {run_id}")

# COMMAND ----------

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    # Per-policy compliance table: append mode for audit trail
    priced_pd = priced_df.to_pandas()
    priced_pd["run_date"]    = str(date.today())
    priced_pd["mlflow_run"]  = run_id

    (
        spark.createDataFrame(priced_pd)
        .write.format("delta")
        .mode("append")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.renewal_pricing_audit")
    )
    print(f"Compliance audit appended to {CATALOG}.{SCHEMA}.renewal_pricing_audit")

    # Demand curve table: overwrite with latest run
    demand_pd = demand_df.to_pandas()
    demand_pd["run_date"]  = str(date.today())
    demand_pd["mlflow_run"] = run_id

    (
        spark.createDataFrame(demand_pd)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.demand_curve")
    )
    print(f"Demand curve written to {CATALOG}.{SCHEMA}.demand_curve")

    # CATE summary by NCD: for monitoring drift in elasticity estimates over time
    cate_summary = (
        df_renewals
        .select(["ncd_years", "channel"])
        .with_columns(pl.Series("cate", cate_values.to_numpy()))
        .group_by(["ncd_years", "channel"])
        .agg([
            pl.col("cate").mean().alias("mean_cate"),
            pl.col("cate").std().alias("std_cate"),
            pl.col("cate").count().alias("n_policies"),
        ])
        .to_pandas()
    )
    cate_summary["run_date"]   = str(date.today())
    cate_summary["mlflow_run"] = run_id

    (
        spark.createDataFrame(cate_summary)
        .write.format("delta")
        .mode("append")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.cate_monitoring")
    )
    print(f"CATE monitoring appended to {CATALOG}.{SCHEMA}.cate_monitoring")

except Exception as e:
    print(f"Could not write to Unity Catalog: {e}")
    print("Results available in memory as `priced_df`, `demand_df`, `cate_values`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What this notebook built:
# MAGIC
# MAGIC | Step | Output |
# MAGIC |------|--------|
# MAGIC | Conversion model (logistic) | Baseline with interpretable price coefficient; AUC comparison |
# MAGIC | Conversion model (CatBoost) | Improved predictive accuracy; same prediction API |
# MAGIC | Retention model | Lapse probability by price change; sensitivity by channel |
# MAGIC | Treatment variation diagnostic | Confirmed sufficient price variation for DML |
# MAGIC | DML elasticity (conversion) | Causal estimate close to true -2.0; naive bias documented |
# MAGIC | CausalForestDML (renewal) | Per-customer CATE; GATE by NCD, channel, age confirmed |
# MAGIC | Elasticity surface | NCD × age heatmap; channel bar chart |
# MAGIC | Demand curve | Profit-maximising price change identified |
# MAGIC | Renewal optimiser | Per-policy optimal prices with ENBP constraint |
# MAGIC | ENBP audit | Zero breaches confirmed; per-policy audit trail ready |
# MAGIC | Delta tables | renewal_pricing_audit (append), demand_curve, cate_monitoring |
# MAGIC
# MAGIC The near-deterministic price diagnostic is the gate that protects you from fitting
# MAGIC a DML model on data that cannot support causal identification. Run it on every dataset
# MAGIC before fitting. If it fails, take the report to the pricing director and use it to
# MAGIC make the case for an A/B experiment or a formal quasi-experimental design.
# MAGIC
# MAGIC Next: Module 10 - Automated Interaction Detection
# MAGIC Using neural interaction detection to find the interactions your GLM is missing.

# COMMAND ----------

print("=" * 60)
print("MODULE 9 COMPLETE")
print("=" * 60)
print()
print(f"Conversion dataset:    {len(df_quotes):,} quotes")
print(f"Renewal dataset:       {len(df_renewals):,} policies")
print(f"DML conversion ATE:    {est_conversion.elasticity_:.3f}")
print(f"CausalForest ATE:      {ate:.3f}")
print(f"CATE std deviation:    {cate_values.std():.3f}")
print(f"ENBP binding:          {100*binding/total:.1f}% of policies")
print(f"ENBP breaches:         {n_breaches}")
print()
print("Next: Module 10 - Automated Interaction Detection")
