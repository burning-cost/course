## Part 14: Stochastic extension — chance constraints

The base optimiser finds multipliers that satisfy the LR constraint in expectation: the expected LR at new rates must be at or below the target. But expected values are means. The actual LR in any single year will differ from the expectation due to claims randomness.

A chance constraint reformulation asks a stronger question: with 90% probability, the portfolio LR at new rates must be at or below the target.

### Formal statement

The chance constraint is:

```
P(LR(m) <= target) >= alpha
```

where `alpha = 0.90`. This says: there must be at most a 10% chance of the LR exceeding the target in the realised year.

Under the Chebyshev-based (Branda 2014) reformulation, this converts to a deterministic constraint:

```
E[LR(m)] + z_alpha * sigma[LR(m)] <= target
```

where:
- `E[LR(m)]` is the expected portfolio LR at rates m
- `sigma[LR(m)]` is the standard deviation of the portfolio LR at rates m
- `z_alpha = sqrt(alpha / (1 - alpha))` is the Chebyshev-based quantile (more conservative than normal)

For `alpha = 0.90`: `z_0.90 = sqrt(0.90 / 0.10) = 3.0`.

**Note on the normal approximation.** The Chebyshev-based formulation is distribution-free and conservative. For large portfolios (50,000+ policies) where the CLT applies well, you can use `z_alpha = 1.282` (the 90th percentile of the standard normal). The library uses the Chebyshev formulation by default because it makes no normality assumption.

### Setting up the stochastic optimiser

The stochastic extension requires per-policy claims variance. Use `ClaimsVarianceModel` to build this from your Tweedie GLM parameters:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, ClaimsVarianceModel

# Per-policy Tweedie variance model
# dispersion=1.2, power=1.5 are typical for UK motor (compound Poisson-gamma)
var_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=expected_loss_cost,
    dispersion=1.2,
    power=1.5,
)
print(var_model)

# Stochastic config
stoch_config = ConstraintConfig(
    lr_max=LR_TARGET,
    retention_min=RETENTION_FLOOR,
    max_rate_change=MAX_RATE_CHANGE,
    stochastic_lr=True,
    stochastic_alpha=0.90,    # P(LR <= target) >= 0.90
)

# Build stochastic optimiser (same interface, just add claims_variance)
stoch_opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=np.ones(N),
    claims_variance=var_model.variance_claims,   # required for stochastic LR
    constraints=stoch_config,
    demand_model="log_linear",
    seed=42,
)

stoch_result = stoch_opt.optimise()

print(f"Stochastic solve converged: {stoch_result.converged}")
print(f"Expected LR (mean):         {stoch_result.expected_loss_ratio:.4f}")
print(f"Expected retention:         {stoch_result.expected_retention:.4f}")
print(f"Expected profit:            £{stoch_result.expected_profit:,.0f}")
```

### Comparing deterministic and stochastic solutions

```python
det_mults   = result.multipliers
stoch_mults = stoch_result.multipliers
pct_diff    = (stoch_mults - det_mults) / det_mults * 100

print("Deterministic vs stochastic comparison:")
print(f"  Mean multiplier (det):    {det_mults.mean():.4f}")
print(f"  Mean multiplier (stoch):  {stoch_mults.mean():.4f}")
print()
print(f"  Profit (det):             £{result.expected_profit:,.0f}")
print(f"  Profit (stoch):           £{stoch_result.expected_profit:,.0f}")
print(f"  Profit cost of stochastic: £{result.expected_profit - stoch_result.expected_profit:,.0f}")
print()
print(f"  Mean rate increase (det):   {pct_diff.mean() * 0:+.2f}% vs base")
print(f"  Mean rate increase (stoch): {pct_diff.mean():+.2f}% vs deterministic")
```

**What you should see:** The stochastic solution requires higher prices on average. The difference is the **prudence loading** — the additional rate needed to ensure the LR target is met with 90% probability rather than just in expectation. For a typical diversified UK motor book, this loading is 0.5-2.0 percentage points on average, depending on the dispersion of the claims distribution.

### Interpreting the stochastic result

The stochastic solution answers the question: "What rates do we need to be 90% confident of staying below the LR target, accounting for claims randomness?"

The deterministic solution answers: "What rates give us an expected LR equal to the target?"

At the deterministic rates, there is approximately a 50% chance of the realised LR exceeding the target in any given year (assuming the expected LR is exactly at the target, as it should be at the optimum). The stochastic solution buys that probability down to 10%.

For most UK pricing actuaries, the deterministic target is the operational constraint (approved by the pricing committee), and the stochastic analysis is a sensitivity or board-level risk indicator. The two approaches answer different questions:

- Deterministic: "What rate achieves 72% LR on average?"
- Stochastic (90%): "What rate ensures 72% LR is not exceeded 90% of the time?"

Which one you use depends on whether the pricing committee is managing to an expected outcome or to a risk-adjusted outcome.

### ClaimsVarianceModel constructors

If you do not have Tweedie parameters, the library offers alternatives:

```python
# From a frequency-severity decomposition
var_model = ClaimsVarianceModel.from_overdispersed_poisson(
    expected_counts=expected_claim_counts,
    mean_severity=mean_severity_per_claim,
    severity_variance=severity_variance_per_claim,
    overdispersion=1.2,  # from quasi-Poisson GLM
)

# From raw variance estimates (e.g. from a bootstrap)
var_model = ClaimsVarianceModel(
    mean_claims=expected_loss_cost,
    variance_claims=your_variance_array,
)
```
