## Part 13: Stage 9 — Rate optimisation

The rate optimiser receives the pure premium from Stage 6 and finds the factor adjustments that minimise loss ratio subject to volume, ENBP, and factor-bounds constraints. This is Module 7's optimiser connected to the pipeline's own model outputs — not a standalone exercise with synthetic inputs.

Add a markdown cell:

```python
%md
## Stage 9: Rate optimisation
```

### Building the renewal portfolio

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier
from scipy.special import expit

# Use test-year policies as the renewal portfolio proxy.
# In production, the renewal portfolio comes from the policy administration system.
# The connection to Stage 6: pure_premium here is the actual model output —
# frequency * severity from the trained GBM — not a separately generated proxy.

n_renewal = min(5_000, len(df_test))
rng_opt   = np.random.default_rng(seed=9999)
ren_idx   = rng_opt.choice(len(df_test), n_renewal, replace=False)
df_ren    = df_test.iloc[ren_idx].copy()
pp_ren    = pure_premium[ren_idx]

# Technical premium: pure premium from freq * sev models
tech_prem = pp_ren

# Current premium: assume book running at LR_TARGET (break-even starting position)
# In production, current_premium comes from the policy administration system.
curr_prem = tech_prem / LR_TARGET

# Market (new-business equivalent) premium: assume market is 2pp softer than current LR
mkt_prem  = tech_prem / (LR_TARGET - 0.02)

print(f"Renewal portfolio: {n_renewal:,} policies")
print(f"Mean technical premium: £{tech_prem.mean():,.2f}")
print(f"Mean current premium:   £{curr_prem.mean():,.2f}")
print(f"Mean market premium:    £{mkt_prem.mean():,.2f}")
print(f"Starting LR:            {(tech_prem / curr_prem).mean():.3f}")
```

### Demand model

The elasticity determines how sensitive renewal probability is to price changes. A price coefficient of -2.0 means a 10% price increase reduces renewal probability by approximately `1 - sigmoid(1 + (-2) * log(1.10)) ≈ 17%` at the portfolio mean. This is consistent with PCW-driven UK motor markets.

```python
# Logistic renewal probability:
# P(renew) = sigmoid(intercept + price_coef * log(curr / mkt))
# price_coef = -2.0: moderate price sensitivity
# intercept = 1.2: baseline renewal probability ~77% at equal prices

intercept  = 1.2
price_coef = -2.0
tenure_coef = 0.04   # longer-tenure customers are less price-sensitive

tenure = rng_opt.integers(0, 10, n_renewal).astype(float)

log_price_ratio = np.log(np.clip(curr_prem / mkt_prem, 1e-6, None))
p_renew = expit(intercept + price_coef * log_price_ratio + tenure_coef * tenure)

# Demand elasticity: d(log P(renew)) / d(log price)
# For the logistic model at mean: approximately price_coef * P * (1 - P)
# We pass a fixed elasticity per policy for the optimiser's gradient computation
elasticity = np.full(n_renewal, price_coef * p_renew.mean() * (1 - p_renew.mean()))

channels = rng_opt.choice(["PCW", "direct"], n_renewal, p=[0.68, 0.32])
renewal_flag = np.ones(n_renewal, dtype=bool)   # all policies are renewals

# ENBP: the new-business equivalent price.
# For ENBP compliance under PS21/11, the renewal price must not exceed
# the equivalent new-business price for equivalent risk. We use mkt_prem
# as the ENBP benchmark — in production this comes from the NB pricing model.
enbp = mkt_prem * 1.01   # 1% uplift above market: a conservative ENBP ceiling

print(f"\nDemand model:")
print(f"  Mean renewal probability: {p_renew.mean():.3f}")
print(f"  Mean elasticity:          {elasticity.mean():.3f}")
print(f"  Channel split:            PCW={sum(channels=='PCW'):,}, "
      f"direct={sum(channels=='direct'):,}")
```

### Running the optimiser

```python
config = ConstraintConfig(
    lr_max=LR_TARGET,            # loss ratio must not exceed 72%
    retention_min=VOLUME_FLOOR,  # retain at least 97% of policies by volume
    max_rate_change=FACTOR_UPPER - 1.0,   # maximum increase per factor: 15%
)

opt = PortfolioOptimiser(
    technical_price=tech_prem,
    expected_loss_cost=tech_prem,   # for this example: cost = tech premium
    p_demand=p_renew,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config,
)

result = opt.optimise()

print(f"\nOptimisation result:")
print(f"  Converged:              {result.converged}")
print(f"  Expected LR:            {result.expected_loss_ratio:.4f}  "
      f"(target: {LR_TARGET:.4f})")
print(f"  Expected volume:        {result.expected_volume_ratio:.4f}  "
      f"(floor: {VOLUME_FLOOR:.4f})")
print(f"  Expected profit:        £{result.expected_profit:,.0f}")
print(f"  ENBP violations:        {result.enbp_violations}")
```

**What `result.converged` means.** The SLSQP solver converges when the gradient of the objective function is below a tolerance threshold (default: 1e-9) and all constraints are satisfied within tolerance. If the optimiser does not converge, the result is the best feasible point found — it may satisfy constraints but is not guaranteed to be optimal. Always check `converged` before presenting results.

If the optimiser does not converge with the default settings, try:
1. Relaxing the LR target by 0.5pp — the feasible region may be very small
2. Checking whether the volume and LR constraints are simultaneously feasible (run `opt.check_feasibility()` first)
3. Increasing `n_restarts=3` in the PortfolioOptimiser constructor

### Efficient frontier

```python
# Trace the efficient frontier: solve the optimiser at a range of LR targets
frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="lr_max",
)
frontier_df = frontier.trace(
    lr_range=(0.68, 0.78),
    n_points=12,
)

print("\nEfficient frontier:")
print(frontier_df.to_string() if hasattr(frontier_df, "to_string") else frontier_df)

# Write to Delta
(
    spark.createDataFrame(frontier_df.to_pandas() if hasattr(frontier_df, "to_pandas") else frontier_df)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["efficient_frontier"])
)
print(f"Efficient frontier written to {TABLES['efficient_frontier']}")
```

### Writing the rate action factors to Delta

```python
# The optimiser returns a multiplier per policy.
# For reporting, summarise: mean multiplier by policy segment or factor level.
# Here we report the portfolio-level summary statistics.

rate_summary = pl.DataFrame({
    "run_date":            [RUN_DATE],
    "lr_target":           [LR_TARGET],
    "volume_floor":        [VOLUME_FLOOR],
    "optimiser_converged": [bool(result.converged)],
    "expected_lr":         [round(float(result.expected_loss_ratio), 4)],
    "expected_volume":     [round(float(result.expected_volume_ratio), 4)],
    "expected_profit":     [round(float(result.expected_profit), 2)],
    "n_enbp_violations":   [int(result.enbp_violations)],
    "n_renewal_policies":  [n_renewal],
    "freq_run_id":         [freq_run_id],
    "sev_run_id":          [sev_run_id],
})

(
    spark.createDataFrame(rate_summary.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["rate_change"])
)

spark.sql(f"""
    ALTER TABLE {TABLES['rate_change']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

print(f"Rate action summary written to {TABLES['rate_change']}")
```

**Presenting the frontier.** The pricing committee decision point is not the single-optimum result — it is a point on the frontier. The frontier shows the full trade-off between loss ratio and volume retention. A committee that accepts a 0.74 LR target instead of 0.72 gains 1.5pp of additional volume retention. That is a quantified trade-off, not a judgment call made in a spreadsheet. The efficient_frontier table is the input to that conversation.
