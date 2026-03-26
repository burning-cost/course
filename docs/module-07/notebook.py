# Databricks notebook source
# MAGIC %md
# MAGIC # Module 7: Constrained Rate Optimisation
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC Replace the Excel scenario with a formally stated optimisation problem.
# MAGIC The `insurance-optimise` library solves SLSQP with nonlinear constraints encoding
# MAGIC the LR target, volume retention floor, per-policy rate change cap, and FCA ENBP requirement.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates a synthetic UK motor renewal book with per-policy demand model
# MAGIC 2. Declares and checks constraints (LR, retention, ENBP, rate change bounds)
# MAGIC 3. Solves for optimal per-policy price multipliers
# MAGIC 4. Inspects shadow prices to identify the binding constraint
# MAGIC 5. Traces the efficient frontier across a range of retention targets
# MAGIC 6. Verifies ENBP compliance per-policy
# MAGIC 7. Extends to the stochastic chance-constraint formulation
# MAGIC 8. Writes results to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 5-10 minutes on a single-node cluster.
# MAGIC
# MAGIC **Prerequisites:** `insurance-optimise` library installed. No prior module required.

# COMMAND ----------

%pip install insurance-optimise polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from datetime import date

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from insurance_optimise import (
    PortfolioOptimiser,
    ConstraintConfig,
    EfficientFrontier,
    ClaimsVarianceModel,
)

print(f"Today: {date.today()}")
import insurance_optimise
print(f"insurance-optimise: {insurance_optimise.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor renewal portfolio
# MAGIC
# MAGIC In production, this cell is replaced by:
# MAGIC ```python
# MAGIC df = spark.table("pricing.motor.renewal_portfolio").toPandas()
# MAGIC ```
# MAGIC
# MAGIC The data must contain per-policy:
# MAGIC - `technical_price`: GLM or GBM output (expected cost + expense loading)
# MAGIC - `expected_loss_cost`: expected claims component of technical price
# MAGIC - `p_demand`: renewal probability at current rates (from your demand model)
# MAGIC - `elasticity`: price elasticity (d log x / d log p), negative
# MAGIC - `renewal_flag`: True if this is a renewal policy (ENBP applies)
# MAGIC - `enbp`: equivalent new business price per PS21/11 (renewals only)
# MAGIC
# MAGIC The book in this tutorial is running at ~75% LR against a 72% target.

# COMMAND ----------

rng = np.random.default_rng(2026)
N   = 5_000

# ---------- Risk segments ----------
# Draw four rating factors for each policy.
# These determine the technical price.
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N, p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N, p=[0.20, 0.40, 0.25, 0.15])
tenure      = rng.integers(0, 10, N).astype(float)

# ---------- Premiums ----------
base_rate = 350.0

# Technical price: expected cost + expense loading.
# The expense loading here is 25% of the risk cost, giving a technical LR of 0.80.
expected_loss_cost = (
    base_rate
    * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)
)
# Technical price = cost / 0.80 (20% expense + profit loading built in)
technical_price = expected_loss_cost / 0.80

# Current premiums: book running at ~75% LR (underpriced vs target of 72%)
# Current premium = cost / 0.75 approximately, with spread
current_premium = expected_loss_cost / 0.75 * rng.uniform(0.96, 1.04, N)

# ---------- Demand model ----------
# Per-policy elasticity: PCW customers are more price-sensitive than direct.
# These are the inputs to the optimiser's log-linear demand model.
renewal_flag = rng.random(N) < 0.60
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.70, 0.30]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)
# PCW elasticity -2.0, direct -1.2 (direct customers less price-sensitive)
elasticity = np.where(channel == "PCW", -2.0, -1.2)
# Tenure stickiness: longer-tenured customers are slightly less elastic
elasticity = elasticity + 0.03 * tenure
elasticity = np.clip(elasticity, -3.5, -0.5)

# Baseline renewal probability at current premium (multiplier = 1.0)
# Logistic: p(renew) = sigmoid(intercept + price_coef * log(price_ratio))
# Here price_ratio = current_premium / market_premium.
# Use p_demand = 0.80 base with tenure lift.
market_premium = expected_loss_cost / 0.73 * rng.uniform(0.90, 1.10, N)
log_price_ratio = np.log(current_premium / market_premium)
logit_p = 1.2 + (-2.0) * log_price_ratio + 0.05 * tenure
p_demand = 1.0 / (1.0 + np.exp(-logit_p))
p_demand = np.clip(p_demand, 0.05, 0.95)

# ---------- ENBP (FCA PS21/11) ----------
# Renewal premium must not exceed the equivalent new business price.
# ENBP is the price a new customer with the same risk profile would be quoted.
# For renewals, ENBP is set just above current premium (slight NB discount baked in).
# For new business, ENBP is unused.
enbp = np.where(renewal_flag, current_premium * rng.uniform(0.98, 1.05, N), 0.0)

# ---------- Prior multiplier (year-on-year rate change tracking) ----------
# Assume all policies have multiplier = 1.0 in prior year (current rates = technical rates)
prior_multiplier = np.ones(N)

print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {renewal_flag.sum():,} ({renewal_flag.mean():.1%})")
print(f"PCW:       {(channel == 'PCW').sum():,}")
print(f"Direct:    {(channel == 'direct').sum():,}")
print()
print(f"Current LR (cost/premium):  {expected_loss_cost.sum() / current_premium.sum():.4f}")
print(f"Mean renewal probability:   {p_demand[renewal_flag].mean():.3f}")
print(f"Mean elasticity:            {elasticity.mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build a Polars DataFrame for analysis
# MAGIC
# MAGIC The optimiser works on numpy arrays directly. We also keep a Polars DataFrame
# MAGIC for all premium impact analysis, output tables, and Unity Catalog writes.

# COMMAND ----------

df = pl.DataFrame({
    "policy_id":          [f"MTR{i:06d}" for i in range(N)],
    "channel":            channel.tolist(),
    "renewal_flag":       renewal_flag.tolist(),
    "tenure":             tenure.tolist(),
    "technical_price":    technical_price.tolist(),
    "expected_loss_cost": expected_loss_cost.tolist(),
    "current_premium":    current_premium.tolist(),
    "market_premium":     market_premium.tolist(),
    "p_demand":           p_demand.tolist(),
    "elasticity":         elasticity.tolist(),
    "enbp":               enbp.tolist(),
})

print(df.head(5))
print(f"\nShape: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure constraints and build the optimiser
# MAGIC
# MAGIC The four constraints:
# MAGIC 1. **LR bound**: aggregate expected LR at new rates must be below target
# MAGIC 2. **Retention floor**: expected renewal retention must not fall below floor
# MAGIC 3. **ENBP**: renewal premiums must not exceed new business equivalent (FCA PS21/11)
# MAGIC 4. **Rate change cap**: per-policy year-on-year rate change capped at ±20%
# MAGIC
# MAGIC All constraints go into a single `ConstraintConfig` dataclass.
# MAGIC ENBP and rate change bounds are encoded as per-policy multiplier bounds (box constraints),
# MAGIC which SLSQP handles separately from the inequality constraints.

# COMMAND ----------

LR_TARGET      = 0.72   # target loss ratio
RETENTION_FLOOR = 0.85  # retain at least 85% of renewal customers by count
MAX_RATE_CHANGE = 0.20  # allow at most ±20% year-on-year rate movement

config = ConstraintConfig(
    lr_max=LR_TARGET,
    retention_min=RETENTION_FLOOR,
    max_rate_change=MAX_RATE_CHANGE,
    enbp_buffer=0.0,      # tight to ENBP (no safety margin)
    technical_floor=True, # prices must be at or above cost
)

print("Constraint configuration:")
print(f"  LR target:       {LR_TARGET:.2%}")
print(f"  Retention floor: {RETENTION_FLOOR:.2%}")
print(f"  Rate change cap: ±{MAX_RATE_CHANGE:.0%}")
print(f"  ENBP:            active for {renewal_flag.sum():,} renewal policies")
print(f"  Technical floor: prices must be >= technical_price")

# COMMAND ----------

# Build the optimiser
opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    constraints=config,
    demand_model="log_linear",  # constant elasticity model
    solver="slsqp",
    n_restarts=1,
    seed=42,
)

print(f"Optimiser built: {N:,} policies, {opt.n_constraints} portfolio constraints")
print()
# Check baseline metrics (at multiplier = 1.0)
baseline = opt.portfolio_summary(m=np.ones(N))
print("Baseline metrics (at current rates, m=1.0 for all policies):")
print(f"  Profit:    £{baseline['profit']:,.0f}")
print(f"  GWP:       £{baseline['gwp']:,.0f}")
print(f"  Loss ratio: {baseline['loss_ratio']:.4f}  (target: {LR_TARGET:.4f})")
print(f"  Retention:  {baseline['retention']:.4f}  (floor: {RETENTION_FLOOR:.4f})")
print()
print(f"LR gap to close:  {(baseline['loss_ratio'] - LR_TARGET)*100:.1f}pp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Solve for optimal price multipliers
# MAGIC
# MAGIC SLSQP (Sequential Least Squares Programming) from `scipy.optimize` handles the
# MAGIC nonlinear inequality constraints. The objective is to maximise expected profit —
# MAGIC sum of (price - cost) * demand across all policies.
# MAGIC
# MAGIC Decision variables are per-policy price **multipliers** m_i = p_i / technical_price_i.
# MAGIC Operating in multiplier space keeps variables O(1) in magnitude and makes the
# MAGIC ENBP upper bound directly comparable across policies of different sizes.
# MAGIC
# MAGIC Convergence typically requires 50-150 iterations for N=5,000 policies.

# COMMAND ----------

result = opt.optimise()

print(result)
print()
print(f"Converged:          {result.converged}")
print(f"Solver message:     {result.solver_message}")
print(f"Iterations:         {result.n_iter}")
print()
print(f"Expected profit:    £{result.expected_profit:,.0f}")
print(f"Expected GWP:       £{result.expected_gwp:,.0f}")
print(f"Expected LR:        {result.expected_loss_ratio:.4f}  (target: {LR_TARGET:.4f})")
print(f"Expected retention: {result.expected_retention:.4f}  (floor: {RETENTION_FLOOR:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect the solution
# MAGIC
# MAGIC The result carries the per-policy summary DataFrame. Every policy gets its own
# MAGIC optimal multiplier. The rate change distribution tells you how much disruption
# MAGIC this rate action causes at the individual level.

# COMMAND ----------

# Per-policy summary
summary = result.summary_df
print("Per-policy result (sample):")
print(summary.head(10))
print()

# Distribution of rate changes
rate_changes = summary["rate_change_pct"].to_numpy()
print("Distribution of individual rate changes:")
print(f"  Mean:           {rate_changes.mean():+.1f}%")
print(f"  Median:         {np.median(rate_changes):+.1f}%")
print(f"  10th pctile:    {np.percentile(rate_changes, 10):+.1f}%")
print(f"  90th pctile:    {np.percentile(rate_changes, 90):+.1f}%")
print(f"  Policies > +10%: {(rate_changes > 10).sum():,} ({(rate_changes > 10).mean():.1%})")
print(f"  Policies < -10%: {(rate_changes < -10).sum():,} ({(rate_changes < -10).mean():.1%})")
print()

# ENBP binding: which renewal policies hit the ENBP upper bound
enbp_binding = summary["enbp_binding"].to_numpy()
print(f"ENBP constraint binding:  {enbp_binding.sum():,} of {renewal_flag.sum():,} renewal policies")
print(f"  ({enbp_binding.sum() / renewal_flag.sum():.1%} of renewals are priced at their ENBP cap)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Shadow prices — the binding constraint question
# MAGIC
# MAGIC The shadow price (Lagrange multiplier) on a constraint measures the marginal profit
# MAGIC gain from relaxing that constraint by one unit.
# MAGIC
# MAGIC A positive shadow price on `lr_max` means the LR constraint is binding: if you
# MAGIC raised the LR cap by 0.001 (from 72% to 72.1%), expected profit would improve by
# MAGIC `shadow_price["lr_max"] * 0.001`.
# MAGIC
# MAGIC Zero shadow price means the constraint is not binding at the solution.

# COMMAND ----------

print("Shadow prices at optimal solution:")
for constraint, sp in result.shadow_prices.items():
    binding = "BINDING" if abs(sp) > 1e-6 else "not binding"
    print(f"  {constraint:<20}: {sp:+.4f}  [{binding}]")

print()
print("Reading the shadow prices:")
print(f"  A shadow price on lr_max of {result.shadow_prices.get('lr_max', 0):.4f} means:")
print(f"  - The LR constraint IS {'binding' if abs(result.shadow_prices.get('lr_max', 0)) > 1e-6 else 'NOT binding'}.")
if abs(result.shadow_prices.get('lr_max', 0)) > 1e-6:
    sp_lr = result.shadow_prices.get('lr_max', 0)
    print(f"  - Relaxing the LR cap by 1pp (+0.01) would improve profit by approximately £{sp_lr * 0.01:,.0f}")
    print(f"  - Tightening the LR cap by 1pp (-0.01) would cost approximately £{sp_lr * 0.01:,.0f} in profit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cross-subsidy and consumer impact analysis
# MAGIC
# MAGIC Before submitting the rate action for sign-off, characterise the distribution
# MAGIC of individual premium changes. Consumer Duty (PS22/9) requires you to confirm
# MAGIC no customer segment is being treated unfairly.
# MAGIC
# MAGIC Unlike a uniform factor-table adjustment, per-policy optimisation produces
# MAGIC heterogeneous rate changes. Some policies see increases, some see decreases —
# MAGIC the optimiser trades them off to achieve the portfolio-level objectives.

# COMMAND ----------

# Build premium impact DataFrame (Polars)
df_impact = df.with_columns([
    pl.Series("new_premium",      result.new_premiums.tolist()),
    pl.Series("multiplier",       result.multipliers.tolist()),
    pl.Series("rate_change_pct",  summary["rate_change_pct"].to_list()),
]).with_columns([
    (pl.col("new_premium") - pl.col("current_premium")).alias("abs_change_gbp"),
])

print("Portfolio premium impact:")
print(f"  Mean change:    {df_impact['rate_change_pct'].mean():+.1f}%")
print(f"  Median change:  {df_impact['rate_change_pct'].median():+.1f}%")
mean_abs = df_impact["abs_change_gbp"].mean()
print(f"  Mean abs change: £{mean_abs:.2f}")
print()

# Cross-subsidy by age band (how are different risk segments affected?)
age_analysis = (
    df_impact
    .group_by("channel")
    .agg([
        pl.len().alias("n_policies"),
        pl.col("rate_change_pct").mean().alias("mean_pct_change"),
        pl.col("abs_change_gbp").mean().alias("mean_abs_change_gbp"),
        pl.col("new_premium").mean().alias("mean_new_premium"),
    ])
    .sort("channel")
)
print("Premium impact by channel:")
print(age_analysis)
print()
print("Note: different channels may see different rate changes because per-policy elasticities")
print("differ by channel. The optimiser weights premium increases more heavily on less-elastic")
print("(direct) customers where the demand response is lower.")

# COMMAND ----------

# Distribution chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: rate change distribution
ax1.hist(
    result.summary_df["rate_change_pct"].to_numpy(),
    bins=50, color="steelblue", edgecolor="white", linewidth=0.3,
)
ax1.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax1.axvline(rate_changes.mean(), color="firebrick", linestyle="--",
            label=f"Mean: {rate_changes.mean():+.1f}%")
ax1.set_xlabel("Individual rate change (%)")
ax1.set_ylabel("Number of policies")
ax1.set_title("Distribution of individual rate changes")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: multiplier distribution
ax2.hist(
    result.multipliers,
    bins=50, color="darkorange", edgecolor="white", linewidth=0.3,
)
ax2.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="m=1.0 (no change)")
ax2.set_xlabel("Price multiplier (new premium / technical price)")
ax2.set_ylabel("Number of policies")
ax2.set_title("Distribution of optimal price multipliers")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Rate Optimisation: Policy-Level Impact", fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. ENBP compliance verification
# MAGIC
# MAGIC After solving, always run a per-policy ENBP check independently of the constraint.
# MAGIC The solver enforces ENBP via the multiplier upper bounds, but an independent check
# MAGIC is the compliance evidence for the FCA audit trail.

# COMMAND ----------

# ENBP check: for every renewal policy, new_premium <= enbp
new_premiums  = result.new_premiums
enbp_arr      = enbp
renewal_mask  = renewal_flag.astype(bool)

enbp_excess = new_premiums[renewal_mask] - enbp_arr[renewal_mask]
violations  = enbp_excess > 0.01   # 1p tolerance for floating-point

n_renewals    = renewal_mask.sum()
n_violations  = violations.sum()

print("ENBP compliance verification:")
print(f"  Renewal policies checked: {n_renewals:,}")
print(f"  ENBP violations:          {n_violations}")

if n_violations == 0:
    print("  RESULT: All renewal premiums are at or below ENBP.")
    print("  ENBP constraint satisfied per-policy.")
else:
    print("  RESULT: ENBP violations detected. Do not proceed to sign-off.")
    top5 = np.sort(enbp_excess[violations])[::-1][:5]
    print(f"  Top 5 violation amounts (£): {[f'{x:.2f}' for x in top5]}")

print()
print("ENBP summary:")
print(f"  Max excess (should be <= 0):    £{enbp_excess.max():.4f}")
print(f"  Mean excess on binding policies: £{enbp_excess[enbp_excess > -0.01].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. The efficient frontier
# MAGIC
# MAGIC A single solve gives one point. The efficient frontier sweeps the retention
# MAGIC constraint from loose (low floor, more pricing freedom) to tight (high floor,
# MAGIC less room to take rate). Each point is an independent solve. The curve shows
# MAGIC the profit cost of each retention target.
# MAGIC
# MAGIC This is the tool for the pricing committee conversation. Instead of asking
# MAGIC "should we use a 85% or 87% retention floor?", you show them the frontier
# MAGIC and let them choose the trade-off.

# COMMAND ----------

frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="volume_retention",
    sweep_range=(0.80, 0.97),
    n_points=15,
    n_jobs=1,
)

frontier_result = frontier.run()

print("Efficient frontier (all points):")
print(frontier_result.data)
print()

# Converged points only
pareto = frontier_result.pareto_data()
print(f"Converged points: {len(pareto)} of {len(frontier_result.data)}")

# COMMAND ----------

# Plot the frontier
pareto_pd = pareto.to_pandas()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: profit vs retention
ax1.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["profit"] / 1000,
    marker="o", color="steelblue", linewidth=2, markersize=6,
)
ax1.axvline(RETENTION_FLOOR * 100, linestyle="--", color="grey", alpha=0.7,
            label=f"Retention floor ({RETENTION_FLOOR:.0%})")
ax1.set_xlabel("Expected retention (%)")
ax1.set_ylabel("Expected profit (£k)")
ax1.set_title("Efficient frontier: profit vs retention")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: loss ratio vs retention
ax2.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["loss_ratio"] * 100,
    marker="o", color="darkorange", linewidth=2, markersize=6,
)
ax2.axhline(LR_TARGET * 100, linestyle="--", color="firebrick", alpha=0.7,
            label=f"LR target ({LR_TARGET:.0%})")
ax2.set_xlabel("Expected retention (%)")
ax2.set_ylabel("Expected loss ratio (%)")
ax2.set_title("Loss ratio across retention targets")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Motor renewal book — Q2 2026 rate action frontier", fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Stochastic extension — chance constraints
# MAGIC
# MAGIC The base optimiser targets an expected LR. A chance constraint asks the stronger
# MAGIC question: with 90% probability, the portfolio LR must not exceed the target.
# MAGIC
# MAGIC Under the normal approximation (CLT), this reformulates as:
# MAGIC
# MAGIC     E[LR] + z_alpha * sigma[LR] <= lr_max
# MAGIC
# MAGIC where sigma[LR] is the portfolio LR standard deviation estimated from per-policy
# MAGIC claims variance. The Chebyshev-based z_alpha = sqrt(alpha / (1 - alpha)) is more
# MAGIC conservative than the normal z-score and does not require the normal assumption.
# MAGIC
# MAGIC Enable this by setting `stochastic_lr=True` in `ConstraintConfig` and passing
# MAGIC `claims_variance` (from `ClaimsVarianceModel`) to the optimiser.

# COMMAND ----------

# Build per-policy claims variance from Tweedie GLM parameters
# dispersion=1.2, power=1.5 are typical for UK motor (compound Poisson-gamma)
var_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=expected_loss_cost,
    dispersion=1.2,
    power=1.5,
)
print(var_model)

# COMMAND ----------

# Stochastic config: same constraints but with the chance-constrained LR
stoch_config = ConstraintConfig(
    lr_max=LR_TARGET,
    retention_min=RETENTION_FLOOR,
    max_rate_change=MAX_RATE_CHANGE,
    stochastic_lr=True,
    stochastic_alpha=0.90,   # P(LR <= target) >= 0.90
)

stoch_opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    claims_variance=var_model.variance_claims,
    constraints=stoch_config,
    demand_model="log_linear",
    seed=42,
)

stoch_result = stoch_opt.optimise()
print(stoch_result)
print()
print(f"Converged:                  {stoch_result.converged}")
print(f"Expected LR (mean):         {stoch_result.expected_loss_ratio:.4f}")
print(f"Expected retention:         {stoch_result.expected_retention:.4f}")
print(f"Expected profit:            £{stoch_result.expected_profit:,.0f}")

# COMMAND ----------

# Compare deterministic vs stochastic
det_mults   = result.multipliers
stoch_mults = stoch_result.multipliers

pct_diff = (stoch_mults - det_mults) / det_mults * 100

print("Deterministic vs stochastic multiplier comparison:")
print(f"  Mean multiplier (det):   {det_mults.mean():.4f}")
print(f"  Mean multiplier (stoch): {stoch_mults.mean():.4f}")
print()
print(f"  Mean difference:         {pct_diff.mean():+.2f}%")
print(f"  Median difference:       {np.median(pct_diff):+.2f}%")
print()
print("The stochastic solution requires higher prices (prudence loading) to ensure")
print("90% probability of meeting the LR target, not just expected value equality.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save the audit trail
# MAGIC
# MAGIC The `audit_trail` on every result is a JSON-serialisable dict that captures
# MAGIC the full optimisation record: inputs, constraints, solution, convergence info.
# MAGIC This is required for FCA Consumer Duty regulatory evidence under PS22/9.

# COMMAND ----------

# Save audit trail locally (in production: write to Unity Catalog or ADLS)
import json

audit_path = f"/tmp/motor_rate_action_{date.today().isoformat()}_audit.json"
result.save_audit(audit_path)
print(f"Audit trail saved to: {audit_path}")

# Show the structure (truncated)
audit = result.audit_trail
print("\nAudit trail structure:")
for k, v in audit.items():
    if isinstance(v, dict):
        print(f"  {k}: {{...}} ({len(v)} keys)")
    elif isinstance(v, list) and len(v) > 5:
        print(f"  {k}: [...] ({len(v)} elements)")
    else:
        print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Write results to Unity Catalog

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"

# Per-policy optimal premiums
policy_records = (
    df
    .with_columns([
        pl.Series("optimal_multiplier",   result.multipliers.tolist()),
        pl.Series("optimal_premium",      result.new_premiums.tolist()),
        pl.Series("expected_demand",      result.expected_demand.tolist()),
        pl.Series("rate_change_pct",      result.summary_df["rate_change_pct"].to_list()),
        pl.Series("enbp_binding",         result.summary_df["enbp_binding"].to_list()),
        pl.lit(str(date.today())).alias("run_date"),
        pl.lit(LR_TARGET).alias("lr_target"),
        pl.lit(RETENTION_FLOOR).alias("retention_floor"),
    ])
)

# Frontier data
frontier_records = frontier_result.data.with_columns(
    pl.lit(str(date.today())).alias("run_date"),
)

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    (
        spark.createDataFrame(policy_records.to_pandas())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_policies")
    )
    print(f"Policy premiums written to {CATALOG}.{SCHEMA}.rate_action_policies")

    (
        spark.createDataFrame(frontier_records.to_pandas())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.efficient_frontier")
    )
    print(f"Frontier data written to {CATALOG}.{SCHEMA}.efficient_frontier")

except Exception as e:
    print(f"Could not write to Unity Catalog: {e}")
    print("Results available in memory as `policy_records` and `frontier_records`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What this notebook built:
# MAGIC
# MAGIC | Step | Output |
# MAGIC |------|--------|
# MAGIC | Synthetic portfolio | 5,000 policies with per-policy elasticity, demand, ENBP |
# MAGIC | ConstraintConfig | LR max 72%, retention floor 85%, rate change cap ±20%, ENBP |
# MAGIC | PortfolioOptimiser | Profit-maximising SLSQP solve, analytical gradients |
# MAGIC | OptimisationResult | Per-policy multipliers, premiums, demand, shadow prices |
# MAGIC | ENBP check | Per-policy compliance verification |
# MAGIC | EfficientFrontier | 15 retention targets; profit-retention Pareto curve |
# MAGIC | Stochastic extension | Chance-constrained LR via Branda (2014) + Tweedie variance |
# MAGIC | Delta tables | rate_action_policies, efficient_frontier |
# MAGIC
# MAGIC The shadow price on `lr_max` is the key number for the pricing committee.
# MAGIC It tells you the marginal profit cost of tightening the LR target further.
# MAGIC The knee of the frontier is where that cost escalates past the expected benefit.
# MAGIC
# MAGIC Next: Module 8 — End-to-End Pipeline
# MAGIC This module ties together Modules 1-7 in a single reproducible Databricks pipeline.

# COMMAND ----------

print("=" * 60)
print("MODULE 7 COMPLETE")
print("=" * 60)
print()
print(result)
print()
print(f"Efficient frontier: {len(frontier_result.data)} points traced")
print(f"Converged points:   {len(frontier_result.pareto_data())}")
print()
print("Next: Module 8 — End-to-End Pipeline")
