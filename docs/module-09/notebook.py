# Databricks notebook source
# MAGIC %md
# MAGIC # Module 9: Demand Elasticity
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC The risk model tells you what a policy costs. It cannot tell you whether the customer
# MAGIC will renew. This notebook estimates causal price elasticity — how renewal probability
# MAGIC responds to price changes — using Double Machine Learning, and applies those estimates
# MAGIC to find the profit-maximising renewal price for every policy in the book.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Demonstrates the confounding bias in naive logistic regression
# MAGIC 2. Runs the pre-flight diagnostic before fitting any elasticity model
# MAGIC 3. Estimates causal price elasticity with `RenewalElasticityEstimator` (CausalForestDML)
# MAGIC 4. Computes group average treatment effects by NCD, age, and channel
# MAGIC 5. Builds the elasticity surface — the pricing committee deliverable
# MAGIC 6. Runs per-policy profit-maximising optimisation subject to the FCA PS21/5 ENBP ceiling
# MAGIC 7. Produces the per-policy ENBP compliance audit trail
# MAGIC 8. Logs everything to MLflow and writes to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 20–30 minutes on Databricks Free Edition (CausalForestDML: 5–8 minutes).
# MAGIC
# MAGIC **Key concept:** The naive regression of renewal on price is biased. Risk factors drive both
# MAGIC price (through the rating system) and renewal probability (through market alternatives and
# MAGIC risk preferences). DML removes this confounding by residualising outcome and treatment on
# MAGIC the same confounders before estimating the price coefficient.

# COMMAND ----------

%pip install insurance-causal catboost econml polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import warnings
from datetime import date

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mlflow
import statsmodels.formula.api as smf

from insurance_causal.elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age
from insurance_causal.elasticity.fit import RenewalElasticityEstimator
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics, TreatmentVariationReport
from insurance_causal.elasticity.surface import ElasticitySurface
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve

print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print("All imports: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Synthetic renewal dataset

# COMMAND ----------

df = make_renewal_data(n=50_000, seed=42, price_variation_sd=0.08)

print(f"Shape:                 {df.shape}")
print(f"Renewal rate:          {df['renewed'].mean():.3f}")
print(f"Mean log price change: {df['log_price_change'].mean():.4f}")
print(f"Std log price change:  {df['log_price_change'].std():.4f}")
print()
print(df.select(["age", "ncd_years", "channel", "last_premium", "enbp",
                 "log_price_change", "renewed", "true_elasticity"]).head(5))

# COMMAND ----------

# Ground-truth elasticity by segment (for validation later)
true_ncd = true_gate_by_ncd(df)
print("True elasticity by NCD band (DGP):")
print(true_ncd)

true_age = true_gate_by_age(df)
print("\nTrue elasticity by age band (DGP):")
print(true_age)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Naive regression benchmark

# COMMAND ----------

df_pd = df.with_columns([
    pl.col("channel").cast(pl.Utf8),
    pl.col("vehicle_group").cast(pl.Utf8),
    pl.col("region").cast(pl.Utf8),
]).to_pandas()

formula = (
    "renewed ~ log_price_change + age + ncd_years "
    "+ C(vehicle_group) + C(region) + C(channel)"
)
naive_logit = smf.logit(formula, data=df_pd).fit(disp=0)

naive_coef = naive_logit.params["log_price_change"]
true_ate   = float(df["true_elasticity"].mean())

print(f"True ATE (DGP):         {true_ate:.3f}")
print(f"Naive logistic coef:    {naive_coef:.3f}")
print(f"Absolute bias:          {abs(naive_coef - true_ate):.3f}")
print(f"Relative bias:          {abs(naive_coef - true_ate) / abs(true_ate) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Treatment variation diagnostic

# COMMAND ----------

confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=confounders,
)
print(report.summary())

# COMMAND ----------

# Secondary check: renewal rate should fall as price change rises
cal_summary = diag.calibration_summary(
    df,
    outcome="renewed",
    treatment="log_price_change",
    n_bins=10,
)
print("Renewal rate by price change decile:")
print(cal_summary)

# COMMAND ----------

# Demonstrate the near-deterministic price problem
df_ndp = make_renewal_data(n=50_000, seed=42, near_deterministic=True)
report_ndp = diag.treatment_variation_report(
    df_ndp,
    treatment="log_price_change",
    confounders=confounders,
)
print("Near-deterministic data:")
print(report_ndp.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 7: DML elasticity estimation
# MAGIC
# MAGIC **This cell takes 5–8 minutes.** Start it and read Parts 8–9 of the tutorial while it runs.

# COMMAND ----------

est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
    binary_outcome=True,
    random_state=42,
)

print("Fitting CausalForestDML with CatBoost nuisance models...")
est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)
print("Done.")

# COMMAND ----------

ate, lb, ub = est.ate()

print(f"True ATE (DGP):       {true_ate:.3f}")
print(f"DML estimate:         {ate:.3f}")
print(f"95% CI:               [{lb:.3f}, {ub:.3f}]")
print(f"Naive logistic:       {naive_coef:.3f}")
print()
print(f"DML bias:             {abs(ate - true_ate):.3f}")
print(f"Naive bias:           {abs(naive_coef - true_ate):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 8: Group average treatment effects

# COMMAND ----------

gate_ncd = est.gate(df, by="ncd_years")

# Side-by-side vs. ground truth
comparison_ncd = gate_ncd.join(
    true_ncd.rename({"true_elasticity_mean": "true_ate"}),
    on="ncd_years",
    how="left",
).with_columns(
    (pl.col("elasticity") - pl.col("true_ate")).alias("bias")
)
print("GATE by NCD band — estimated vs. true:")
print(comparison_ncd.select(["ncd_years", "elasticity", "ci_lower", "ci_upper",
                              "true_ate", "bias", "n"]))

# COMMAND ----------

gate_channel = est.gate(df, by="channel")
print("GATE by channel:")
print(gate_channel)

gate_age = est.gate(df, by="age_band")
print("\nGATE by age band:")
print(gate_age)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 9: Per-customer CATE estimates

# COMMAND ----------

cate_values = est.cate(df)

print("CATE distribution across 50,000 customers:")
print(f"  Mean:  {cate_values.mean():.3f}  (ATE: {ate:.3f})")
print(f"  Std:   {cate_values.std():.3f}")
print(f"  10th:  {np.percentile(cate_values, 10):.3f}")
print(f"  90th:  {np.percentile(cate_values, 90):.3f}")

# COMMAND ----------

lb_vals, ub_vals = est.cate_interval(df, alpha=0.05)

df_with_cate = df.with_columns([
    pl.Series("cate", cate_values),
    pl.Series("cate_lower", lb_vals),
    pl.Series("cate_upper", ub_vals),
])

n_significant = int((df_with_cate["cate_upper"] < 0).sum())
print(f"Customers with significantly negative elasticity: {n_significant:,} "
      f"({100 * n_significant / len(df):.1f}%)")

# COMMAND ----------

# Validate CATE recovery against ground truth
cate_bias = cate_values - df["true_elasticity"].to_numpy()
correlation = float(np.corrcoef(cate_values, df["true_elasticity"].to_numpy())[0, 1])

print("CATE recovery validation:")
print(f"  Mean bias:    {cate_bias.mean():.4f}")
print(f"  RMSE:         {np.sqrt(np.mean(cate_bias**2)):.4f}")
print(f"  Correlation (estimated vs. true): {correlation:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 10: Elasticity surface

# COMMAND ----------

surface = ElasticitySurface(est)

# NCD × channel segment summary
summary_ncd_channel = surface.segment_summary(df, by=["ncd_years", "channel"])
print("Elasticity surface — NCD years × channel:")
print(summary_ncd_channel)

# COMMAND ----------

# Heatmap: NCD × age
fig_surface = surface.plot_surface(df, dims=["ncd_years", "age_band"])
fig_surface.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# GATE bar chart: channel
fig_channel = surface.plot_gate(df, by="channel")
fig_channel.savefig("/tmp/gate_by_channel.png", dpi=150, bbox_inches="tight")
plt.show()

# GATE bar chart: NCD
fig_ncd = surface.plot_gate(df, by="ncd_years")
fig_ncd.savefig("/tmp/gate_by_ncd.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 11: Portfolio demand curve

# COMMAND ----------

demand_df = demand_curve(
    est,
    df,
    price_range=(-0.25, 0.25, 50),
)

max_profit_row = demand_df.sort("predicted_profit", descending=True).row(0, named=True)
print(f"Profit-maximising price change: {max_profit_row['pct_price_change'] * 100:.1f}%")
print(f"Renewal rate at optimum:        {max_profit_row['predicted_renewal_rate'] * 100:.1f}%")
print(f"Expected profit per policy:     £{max_profit_row['predicted_profit']:.2f}")

# COMMAND ----------

fig_demand = plot_demand_curve(demand_df, show_profit=True)
fig_demand.savefig("/tmp/demand_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 12: Per-policy optimisation

# COMMAND ----------

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,
)

priced_df = opt.optimise(df, objective="profit")

print(f"Policies optimised:      {len(priced_df):,}")
print(f"Mean optimal price:      £{priced_df['optimal_price'].mean():.2f}")
print(f"Mean last premium:       £{priced_df['last_premium'].mean():.2f}")
pct_change = ((priced_df["optimal_price"] / priced_df["last_premium"]).mean() - 1) * 100
print(f"Mean price change:       {pct_change:.1f}%")
print(f"Mean ENBP headroom:      £{priced_df['enbp_headroom'].mean():.2f}")
print(f"Predicted renewal rate:  {priced_df['predicted_renewal_prob'].mean():.3f}")
print(f"Expected profit/policy:  £{priced_df['expected_profit'].mean():.2f}")

# COMMAND ----------

binding = (priced_df["enbp_headroom"] < 1.0).sum()
total = len(priced_df)
print(f"ENBP constraint binding: {binding:,} of {total:,} ({100 * binding / total:.1f}%)")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ratio = (priced_df["optimal_price"] / priced_df["enbp"]).to_numpy()
axes[0].hist(ratio, bins=40, color="#1f77b4", alpha=0.8)
axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="ENBP ceiling")
axes[0].set_xlabel("Optimal price / ENBP")
axes[0].set_ylabel("Count")
axes[0].set_title("Optimal price relative to ENBP ceiling")
axes[0].legend()

ncd_profit = (
    priced_df
    .group_by("ncd_years")
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
# MAGIC ## Part 13: ENBP compliance audit

# COMMAND ----------

audit = opt.enbp_audit(priced_df)

n_breaches = int((~audit["compliant"]).sum())
print(f"ENBP audit results:")
print(f"  Total policies:  {len(audit):,}")
print(f"  Compliant:       {audit['compliant'].sum():,}")
print(f"  Breaches:        {n_breaches:,}")
print(f"  Breach rate:     {100 * n_breaches / len(audit):.3f}%")

if n_breaches == 0:
    print("\n  ALL POLICIES COMPLIANT with FCA ICOBS 6B.2")
else:
    print("\n  WARNING: Review breach detail before issuing prices")
    print(audit.filter(~pl.col("compliant")).head(20))

# COMMAND ----------

print("ENBP headroom distribution (£):")
print(audit.select("margin_to_enbp").describe())

tight = (audit["margin_to_enbp"] < 10.0).sum()
print(f"\nPolicies with < £10 headroom: {tight:,} ({100 * tight / len(audit):.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 14: MLflow logging

# COMMAND ----------

mlflow.set_experiment("/Users/your.email@company.com/insurance-pricing/elasticity")

with mlflow.start_run(run_name="renewal_elasticity_causal_forest"):
    mlflow.log_metric("ate", ate)
    mlflow.log_metric("ate_ci_lower", lb)
    mlflow.log_metric("ate_ci_upper", ub)
    mlflow.log_metric("naive_logistic_coef", naive_coef)
    mlflow.log_metric("naive_bias", abs(naive_coef - true_ate))
    mlflow.log_metric("treatment_variation_fraction", report.variation_fraction)
    mlflow.log_metric("enbp_breach_count", n_breaches)
    mlflow.log_metric("enbp_binding_fraction", binding / total)
    mlflow.log_metric("mean_predicted_renewal_prob",
                      float(priced_df["predicted_renewal_prob"].mean()))
    mlflow.log_metric("cate_true_correlation", correlation)

    mlflow.log_param("n_training_records", len(df))
    mlflow.log_param("cate_model", "causal_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("catboost_iterations", 500)
    mlflow.log_param("n_folds", 5)
    mlflow.log_param("confounders", str(confounders))

    run_id = mlflow.active_run().info.run_id
    print(f"Logged to MLflow. Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write results to Unity Catalog
# MAGIC
# MAGIC Uncomment to write to your own catalog and schema.

# COMMAND ----------

audit_with_meta = audit.with_columns([
    pl.lit(str(date.today())).alias("pricing_run_date"),
    pl.lit("v1.0").alias("model_version"),
    pl.lit(run_id).alias("mlflow_run_id"),
])

# spark.createDataFrame(priced_df.to_pandas()).write.format("delta") \
#     .mode("overwrite").saveAsTable("pricing.motor.renewal_optimal_prices")

# spark.createDataFrame(audit_with_meta.to_pandas()).write.format("delta") \
#     .mode("append").saveAsTable("pricing.motor.enbp_audit_log")

# spark.createDataFrame(
#     surface.segment_summary(df, by=["ncd_years", "channel", "age_band"]).to_pandas()
# ).write.format("delta").mode("overwrite").saveAsTable("pricing.motor.elasticity_surface")

print("Tables to write (uncomment to activate):")
print(f"  pricing.motor.renewal_optimal_prices — {len(priced_df):,} rows")
print(f"  pricing.motor.enbp_audit_log         — {len(audit_with_meta):,} rows")
print(f"  pricing.motor.elasticity_surface     — segment surface")
