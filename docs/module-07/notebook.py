# Databricks notebook source
# MAGIC %md
# MAGIC # Module 7: Constrained Rate Optimisation
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC Replace the Excel scenario with a formally stated optimisation problem.
# MAGIC The `rate-optimiser` library solves SLSQP with nonlinear constraints encoding
# MAGIC the LR target, volume floor, per-factor movement caps, and FCA ENBP requirement.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates a synthetic UK motor renewal book with demand model
# MAGIC 2. Wraps data in PolicyData and FactorStructure
# MAGIC 3. Declares and checks four constraints (LR, volume, ENBP, factor bounds)
# MAGIC 4. Solves for the optimal factor adjustments
# MAGIC 5. Traces the efficient frontier across a range of LR targets
# MAGIC 6. Extracts shadow prices to identify the binding constraint
# MAGIC 7. Produces updated factor tables for implementation
# MAGIC 8. Writes results to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 10-15 minutes (optimisation converges in < 60 seconds).
# MAGIC
# MAGIC **Prerequisites:** rate-optimiser library installed. No prior module required.

# COMMAND ----------

%pip install rate-optimiser polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit
import matplotlib.pyplot as plt

from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

print(f"Today: {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor renewal portfolio
# MAGIC
# MAGIC In production, this cell is replaced by:
# MAGIC ```python
# MAGIC df = spark.table("pricing.motor.renewal_portfolio").toPandas()
# MAGIC ```
# MAGIC
# MAGIC The data must contain:
# MAGIC - policy_id, channel, renewal_flag: identifiers
# MAGIC - technical_premium: your GLM or GBM output (claims proxy)
# MAGIC - current_premium: what you are currently charging
# MAGIC - market_premium: competitive price (from PCW monitoring or proxy)
# MAGIC - renewal_prob: current renewal probability from your demand model
# MAGIC - f_<factor>: one column per rating factor, containing the current relativity value
# MAGIC
# MAGIC The book in this tutorial is running at ~75% LR against a 72% target.
# MAGIC Most real-world exercises start with a book that needs rate.

# COMMAND ----------

rng = np.random.default_rng(2026)
N   = 5_000

# Factor relativities: what the current tariff produces for each policy
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N, p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N, p=[0.20, 0.40, 0.25, 0.15])
tenure      = rng.integers(0, 10, N).astype(float)
tenure_disc = np.ones(N)  # currently neutral (no tenure discount applied)

base_rate         = 350.0
technical_premium = (
    base_rate
    * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)
)

# Current premium: book running at ~75% LR (underpriced)
current_premium = technical_premium / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: competitive market slightly below current rates
market_premium = technical_premium / 0.73 * rng.uniform(0.90, 1.10, N)

renewal_flag = rng.random(N) < 0.60
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.70, 0.30]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# Demand model: logistic with price semi-elasticity = -2.0
# This is typical for a UK PCW-heavy motor book
price_ratio = current_premium / market_premium
logit_p     = 1.0 + (-2.0) * np.log(price_ratio) + 0.05 * tenure
renewal_prob = expit(logit_p)

# Note: This synthetic data generation uses pd.DataFrame because the rate-optimiser
# library expects pandas input at the policy boundary. In production, load from Delta
# and call .toPandas() at that same boundary. Polars is used for factor table updates
# (see cell 7 below) where pandas is not needed.
df = pd.DataFrame({
    "policy_id":         [f"MTR{i:06d}" for i in range(N)],
    "channel":           channel,
    "renewal_flag":      renewal_flag,
    "technical_premium": technical_premium,
    "current_premium":   current_premium,
    "market_premium":    market_premium,
    "renewal_prob":      renewal_prob,
    "tenure":            tenure,
    "f_age":             age_rel,
    "f_ncb":             ncb_rel,
    "f_vehicle":         vehicle_rel,
    "f_region":          region_rel,
    "f_tenure_discount": tenure_disc,
})

print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean():.1%})")
print(f"Channels:  {df.groupby('channel')['policy_id'].count().to_dict()}")
print(f"Current LR: {df['technical_premium'].sum() / df['current_premium'].sum():.3f}")
print(f"Mean renewal prob: {df['renewal_prob'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wrap data and declare factor structure
# MAGIC
# MAGIC `PolicyData` validates the input and computes derived statistics.
# MAGIC `FactorStructure` declares which factors are renewal-only.
# MAGIC
# MAGIC The `renewal_factor_names` parameter is critical for ENBP compliance.
# MAGIC Tenure discounts, NCB-at-renewal adjustments, and any other factors that
# MAGIC new business does not receive must be listed here. If in doubt, include it.
# MAGIC A false positive (treating an NB factor as renewal-only) is conservative.
# MAGIC A false negative means the ENBP constraint computes the wrong NB equivalent.

# COMMAND ----------

data = PolicyData(df)
print(f"PolicyData loaded:")
print(f"  n_policies: {data.n_policies}")
print(f"  n_renewals: {data.n_renewals}")
print(f"  channels:   {data.channels}")
print(f"  Current LR: {data.current_loss_ratio():.4f}")

FACTOR_NAMES = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=FACTOR_NAMES,
    factor_values=df[FACTOR_NAMES],
    renewal_factor_names=["f_tenure_discount"],  # only renewals get tenure discount
)

print(f"\nFactorStructure:")
print(f"  n_factors:          {fs.n_factors}")
print(f"  renewal-only:       {fs.renewal_factor_names}")
print(f"  ENBP-relevant:      these factors are excluded from NB equivalent calculation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define the demand model
# MAGIC
# MAGIC The demand model translates price ratio -> renewal probability.
# MAGIC The key parameter is the price semi-elasticity (-2.0 here):
# MAGIC a 1% price increase above market reduces renewal probability by ~2pp.
# MAGIC
# MAGIC UK motor PCW elasticity: typically -1.5 to -3.0.
# MAGIC Direct channel: less elastic (-0.5 to -1.5), because direct customers are
# MAGIC less price-sensitive than PCW customers who are comparing quotes.
# MAGIC
# MAGIC Validate your demand model against observed lapse rates before running
# MAGIC the optimiser. A miscalibrated elasticity produces rate strategies that
# MAGIC look good in the model and fail in market.

# COMMAND ----------

params = LogisticDemandParams(
    intercept=1.0,
    price_coef=-2.0,   # log-price semi-elasticity
    tenure_coef=0.05,  # tenure effect: longer-tenured customers are stickier
)
demand = make_logistic_demand(params)

# Check the implied elasticity at market price (price_ratio = 1.0)
test_ratios = np.ones(100)
elasticities = demand.elasticity_at(test_ratios)
print(f"Price elasticity at market price: {elasticities.mean():.2f}")
print("(Expected: -1.5 to -2.5 for PCW-heavy UK motor)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build the optimisation problem and check feasibility
# MAGIC
# MAGIC Before solving, check whether all constraints can be satisfied.
# MAGIC If the LR constraint is violated at current rates, the solver needs to find
# MAGIC factor adjustments that bring LR below the target.
# MAGIC
# MAGIC The four constraints:
# MAGIC 1. LR bound: expected LR at new rates must be below target
# MAGIC 2. Volume bound: retention must not fall below floor
# MAGIC 3. ENBP: renewal premiums must not exceed NB equivalent (FCA PS21/5)
# MAGIC 4. Factor bounds: each factor adjustment must stay within approved caps
# MAGIC
# MAGIC The objective function minimises sum of squared deviations from 1.0
# MAGIC (minimum dislocation: we want the smallest rate change that achieves the target).

# COMMAND ----------

LR_TARGET    = 0.72   # target loss ratio
VOLUME_FLOOR = 0.97   # retain at least 97% of expected volume
FACTOR_LOWER = 0.90   # minimum factor adjustment (10% reduction)
FACTOR_UPPER = 1.15   # maximum factor adjustment (15% increase)

opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=FACTOR_LOWER, upper=FACTOR_UPPER, n_factors=fs.n_factors))

print("Constraint configuration:")
print(f"  LR target:      {LR_TARGET:.2%}")
print(f"  Volume floor:   {VOLUME_FLOOR:.2%}")
print(f"  Factor bounds:  [{FACTOR_LOWER:.2%}, {FACTOR_UPPER:.2%}]")
print(f"  ENBP channels:  PCW, direct")
print()
print("Feasibility at current rates (m = 1.0 for all factors):")
print(opt.feasibility_report())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Solve for optimal factor adjustments
# MAGIC
# MAGIC SLSQP (Sequential Least Squares Programming) handles the nonlinear constraints.
# MAGIC The objective is convex in the adjustments (quadratic), but the constraints
# MAGIC are nonlinear because renewal probability enters through the logistic demand model.
# MAGIC SLSQP handles this correctly.
# MAGIC
# MAGIC Convergence typically requires 30-80 iterations. If convergence fails, check:
# MAGIC - Is the problem feasible? (Exercise 1 feasibility check)
# MAGIC - Are the factor bounds tight enough that no solution exists?
# MAGIC - Is the demand model's elasticity plausible?

# COMMAND ----------

result = opt.solve()

print(result.summary())
print()
print("Factor adjustments (multiplicative; apply to all levels of each factor):")
for factor, m in result.factor_adjustments.items():
    direction = "up" if m > 1.0 else ("down" if m < 1.0 else "unchanged")
    print(f"  {factor:<25}: {m:.4f} ({(m-1)*100:+.1f}% - {direction})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Trace the efficient frontier
# MAGIC
# MAGIC A single solve gives you one point. The frontier gives you the full trade-off
# MAGIC between loss ratio improvement and volume retention.
# MAGIC
# MAGIC The shadow price on the LR constraint is the most important output for the
# MAGIC pricing committee conversation. It answers: "What does it cost, in dislocation
# MAGIC terms, to push one more percentage point of LR improvement?"
# MAGIC
# MAGIC The knee of the frontier is the natural stopping point: beyond it, the shadow
# MAGIC price rises faster than the LR improvement you are gaining.

# COMMAND ----------

frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)

print("Efficient frontier:")
print(frontier_df.to_string())

# COMMAND ----------

print("\nShadow price summary:")
print(frontier.shadow_price_summary())

# COMMAND ----------

# Plot the frontier
feasible = frontier.feasible_points()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: LR vs volume frontier
ax1.plot(
    feasible["expected_lr"] * 100,
    feasible["expected_volume"] * 100,
    marker="o", color="steelblue", linewidth=2, markersize=6,
)
ax1.axhline(VOLUME_FLOOR * 100, linestyle="--", color="grey", alpha=0.7,
            label=f"Volume floor ({VOLUME_FLOOR:.0%})")
ax1.set_xlabel("Expected loss ratio (%)")
ax1.set_ylabel("Expected volume retention (%)")
ax1.set_title("Efficient frontier: LR vs volume retention")
ax1.invert_xaxis()
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: shadow price
ax2.plot(
    feasible["lr_target"] * 100,
    feasible["shadow_lr"],
    marker="o", color="darkorange", linewidth=2, markersize=6,
)
ax2.axvline(LR_TARGET * 100, linestyle="--", color="steelblue", alpha=0.7,
            label=f"Target LR ({LR_TARGET:.0%})")
ax2.set_xlabel("LR target (%)")
ax2.set_ylabel("Shadow price (marginal dislocation cost)")
ax2.set_title("Shadow price on loss ratio constraint")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Rate Optimisation: Efficient Frontier", fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Translate adjustments into updated factor tables
# MAGIC
# MAGIC The factor adjustment multipliers from the solver apply uniformly to all levels
# MAGIC of each factor. The shape of the factor table (the relativities between levels)
# MAGIC does not change. Only the overall scale changes.
# MAGIC
# MAGIC This is appropriate for a uniform rate action. If you want to change the shape
# MAGIC of a factor (e.g., steepen the NCD gradient) that is a separate modelling decision
# MAGIC that requires regulatory justification and pricing committee sign-off.

# COMMAND ----------

# Simulated current factor tables - in production, load from Delta
current_tables = {
    "f_age": pl.DataFrame({
        "band": ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "current_relativity": [2.00, 1.50, 1.20, 1.00, 0.92, 0.95, 1.10],
    }),
    "f_ncb": pl.DataFrame({
        "ncd_years": [0, 1, 2, 3, 4, 5],
        "current_relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
    "f_vehicle": pl.DataFrame({
        "group": ["Standard", "Performance", "High-perf", "Prestige"],
        "current_relativity": [0.90, 1.00, 1.10, 1.30],
    }),
    "f_region": pl.DataFrame({
        "region": ["Rural", "National", "Urban", "London"],
        "current_relativity": [0.85, 1.00, 1.10, 1.20],
    }),
    "f_tenure_discount": pl.DataFrame({
        "tenure_years": list(range(10)),
        "current_relativity": [1.00] * 10,
    }),
}

factor_adj = result.factor_adjustments
updated_tables = {}

print("Updated factor tables:")
for fname, tbl in current_tables.items():
    m = factor_adj.get(fname, 1.0)
    updated = tbl.with_columns([
        (pl.col("current_relativity") * m).alias("new_relativity"),
        ((pl.col("current_relativity") * m / pl.col("current_relativity") - 1) * 100).alias("pct_change"),
    ])
    updated_tables[fname] = updated
    print(f"\n{fname} (factor adjustment: {m:.4f} = {(m-1)*100:+.1f}%):")
    print(updated.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Policy-level premium impact analysis
# MAGIC
# MAGIC Before submitting the rate action for sign-off, characterise the distribution
# MAGIC of individual premium changes. Some policies may see larger-than-average increases
# MAGIC due to compounding factor adjustments.
# MAGIC
# MAGIC Consumer Duty requires you to confirm no customer segment is being treated
# MAGIC unfairly. A distribution that is bimodal, or where a specific demographic
# MAGIC sees systematically larger increases, needs to be investigated.

# COMMAND ----------

# Compute policy-level combined adjustment
combined_adj = np.ones(N)
for fname in FACTOR_NAMES:
    m = factor_adj.get(fname, 1.0)
    combined_adj *= m

new_premium    = df["current_premium"].values * combined_adj
pct_change_pol = (new_premium / df["current_premium"].values - 1) * 100

print(f"Portfolio-level premium impact:")
print(f"  Mean increase:    {pct_change_pol.mean():.2f}%")
print(f"  Median increase:  {np.median(pct_change_pol):.2f}%")
print(f"  10th percentile:  {np.quantile(pct_change_pol, 0.10):.2f}%")
print(f"  90th percentile:  {np.quantile(pct_change_pol, 0.90):.2f}%")
print(f"  Policies with > 10% increase: {(pct_change_pol > 10).sum():,} ({(pct_change_pol > 10).mean():.1%})")

# Cross-subsidy analysis: absolute premium changes by age band
# Under uniform factor adjustment, all policies see the same percentage change.
# The cross-subsidy concern is about absolute premium levels: young drivers (high age
# relativities) pay a larger absolute increase because their premiums start higher.
abs_change = new_premium - df["current_premium"].values
print("\nAbsolute premium change by age relativity band:")
age_bands = sorted(df["f_age"].unique())
print(f"  {'Age rel':>10} {'N':>6} {'Mean abs increase':>18} {'Mean pct change':>16}")
for band in age_bands:
    mask = df["f_age"].values == band
    n_b = mask.sum()
    mean_abs = abs_change[mask].mean()
    mean_pct = pct_change_pol[mask].mean()
    print(f"  {band:>10.2f} {n_b:>6,} {mean_abs:>17.2f}  {mean_pct:>15.2f}%")
print("Note: percentage change is uniform across age bands (it is a uniform factor action).")
print("Absolute increase is larger for young drivers (high age relativities) because")
print("their base premiums are higher. This is the cross-subsidy to highlight in sign-off.")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(pct_change_pol, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(pct_change_pol.mean(), color="firebrick", linestyle="--",
           label=f"Mean: {pct_change_pol.mean():.1f}%")
ax.set_xlabel("Individual premium change (%)")
ax.set_ylabel("Number of policies")
ax.set_title("Distribution of individual premium changes")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write results to Unity Catalog
# MAGIC
# MAGIC Write two artefacts:
# MAGIC 1. The optimal factor adjustments (for the pricing team and the data team)
# MAGIC 2. The full frontier data (for pricing committee presentation)

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"

# Factor adjustments table
adj_records = [
    {
        "run_date":         str(date.today()),
        "factor_name":      fname,
        "adjustment":       float(factor_adj.get(fname, 1.0)),
        "pct_change":       float((factor_adj.get(fname, 1.0) - 1) * 100),
        "lr_target":        LR_TARGET,
        "volume_floor":     VOLUME_FLOOR,
        "factor_lower_cap": FACTOR_LOWER,
        "factor_upper_cap": FACTOR_UPPER,
    }
    for fname in FACTOR_NAMES
]

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    (
        spark.createDataFrame(adj_records)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_factors")
    )
    print(f"Factor adjustments written to {CATALOG}.{SCHEMA}.rate_action_factors")

    # Frontier data
    (
        spark.createDataFrame(frontier_df)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.efficient_frontier")
    )
    print(f"Frontier data written to {CATALOG}.{SCHEMA}.efficient_frontier")

except Exception as e:
    print(f"Could not write to Unity Catalog: {e}")
    print("Results available in memory as `adj_records` and `frontier_df`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What this notebook built:
# MAGIC
# MAGIC | Step | Output |
# MAGIC |------|--------|
# MAGIC | PolicyData + FactorStructure | Validated renewal portfolio with 5 factors |
# MAGIC | Demand model | Logistic elasticity = -2.0, tenure effect = +0.05 |
# MAGIC | Feasibility check | LR constraint violated at current rates: rate needed |
# MAGIC | Solve | Optimal factor adjustments at 72% LR target |
# MAGIC | Efficient frontier | 20 LR targets from 68% to 78%; shadow prices computed |
# MAGIC | Factor tables | Updated relativities for all 5 factors |
# MAGIC | Delta tables | rate_action_factors, efficient_frontier |
# MAGIC
# MAGIC The shadow price is the key number for the pricing committee.
# MAGIC At the target LR, it tells you the marginal cost of tightening the target further.
# MAGIC The knee of the frontier is where that cost exceeds twice its initial value.
# MAGIC
# MAGIC Next: Module 8 - End-to-End Pipeline
# MAGIC This module ties together Modules 1-7 in a single reproducible Databricks pipeline.

# COMMAND ----------

print("=" * 60)
print("MODULE 7 COMPLETE")
print("=" * 60)
print()
print(result.summary())
print()
print(f"Efficient frontier: {len(frontier_df)} points traced")
print(f"Feasible points:    {frontier_df['feasible'].sum()}")
print()
print("Next: Module 8 - End-to-End Pipeline")
