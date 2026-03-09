## Part 14: Stochastic extension — chance constraints

The base optimiser finds the factor adjustments that satisfy the LR constraint in expectation: the expected LR at new rates must be at or below the target. But expected values are means. The actual LR in any single year will differ from the expectation due to claims randomness.

A chance constraint reformulation asks a stronger question: with 90% probability, the portfolio LR at new rates must be at or below the target. This is the stochastic equivalent of the deterministic constraint.

### Formal statement

The chance constraint is:

```
P(LR(m) <= target) >= alpha
```

where alpha = 0.90 (or 0.95 for more conservative pricing). This says: there must be at most a 10% chance of the LR exceeding the target in the realised year.

To solve this using the SLSQP framework, we convert the chance constraint to a deterministic constraint using the **normal approximation** for the portfolio loss ratio. This approximation assumes that the portfolio loss ratio is approximately normally distributed, which follows from the Central Limit Theorem when the portfolio is large and claims are approximately independent.

Under the normal approximation:

```
P(LR(m) <= target) >= alpha
```

is equivalent to:

```
E[LR(m)] + z_alpha * sigma[LR(m)] <= target
```

where:
- `E[LR(m)]` is the expected portfolio loss ratio at rates m
- `sigma[LR(m)]` is the standard deviation of the portfolio loss ratio at rates m
- `z_alpha` is the alpha-quantile of the standard normal (e.g., z_0.90 = 1.282, z_0.95 = 1.645)

**When is the normal approximation reasonable?** For diversified books with 50,000+ policies where no single risk dominates the portfolio, the CLT applies well and the normal approximation is sound. For smaller or concentrated books — fewer than 10,000 policies, or books with large commercial risks — the tail behaviour of claims may be far from normal, and the normal approximation may understate the probability of extreme outcomes. For those cases, a simulation-based approach is more appropriate.

For our 5,000-policy synthetic book, the normal approximation is borderline. In practice, UK motor books with this approach would have 50,000+ policies. We use it here to demonstrate the method; on a real book of this size, we recommend validating the assumption with a simulation.

### Setting up the stochastic optimiser

The stochastic extension requires a model of per-policy claims variance. We use the Tweedie variance model (as in Module 5), which is parameterised by the dispersion and power parameters of the Tweedie distribution:

```python
from rate_optimiser.stochastic import (
    StochasticRateChangeOptimiser,
    ClaimsVarianceModel,
    ChanceLossRatioConstraint,
)

# Per-policy Tweedie variance model
# dispersion=1.2, power=1.5 are typical for UK motor (Tweedie between Poisson and Gamma)
variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=df_pd["technical_premium"].values,
    dispersion=1.2,
    power=1.5,
)

# Build the stochastic optimiser
stoch_opt = StochasticRateChangeOptimiser(
    data=data,
    demand=demand,
    factor_structure=fs,
    variance_model=variance_model,
)

# Chance constraint at 90% confidence
stoch_opt.add_constraint(ChanceLossRatioConstraint(
    bound=LR_TARGET,
    alpha=0.90,    # require P(LR <= 0.72) >= 0.90
    normal_approx=True,   # use normal approximation
))
stoch_opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
stoch_opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
stoch_opt.add_constraint(FactorBoundsConstraint(
    lower=FACTOR_LOWER,
    upper=FACTOR_UPPER,
    n_factors=fs.n_factors,
))

stoch_result = stoch_opt.solve()

print(f"Stochastic solve converged: {stoch_result.converged}")
print(f"Expected LR (mean):         {stoch_result.expected_loss_ratio:.4f}")
print(f"LR standard deviation:      {stoch_result.lr_std:.4f}")
print(f"90th percentile LR:         {stoch_result.lr_quantile_90:.4f}")
print(f"Expected volume:            {stoch_result.expected_volume_ratio:.4f}")
print()
print("Factor adjustments (stochastic vs deterministic):")
print(f"  {'Factor':<25} {'Deterministic':>15} {'Stochastic':>12} {'Difference':>12}")
print(f"  {'-'*66}")
for fname in FACTOR_NAMES:
    m_det   = result.factor_adjustments.get(fname, 1.0)
    m_stoch = stoch_result.factor_adjustments.get(fname, 1.0)
    print(f"  {fname:<25} {m_det:>15.4f} {m_stoch:>12.4f} {(m_stoch-m_det)*100:>+11.1f}pp")
```

**What you should see:** The stochastic factor adjustments will be larger than the deterministic ones. The difference is the **prudence loading**: the additional rate required to ensure the LR target is met with 90% probability rather than just in expectation. For a typical diversified UK motor book, this loading is 0.5-1.5 percentage points on each factor.

### Interpreting the stochastic result

The stochastic solution is more conservative than the deterministic one. It requires more rate because it must buffer against claims randomness. The `lr_quantile_90` value shows: at the stochastic rates, there is only a 10% chance the realised LR exceeds the target. At the deterministic rates, the expected LR hits the target but there is a roughly 50% chance of exceeding it in any given year.

For most UK pricing actuaries, the deterministic target is the operational constraint (approved by the pricing committee), and the stochastic analysis is a sensitivity or board-level risk indicator. The two approaches answer different questions:

- Deterministic: "What rate achieves 72% LR on average?"
- Stochastic (90%): "What rate ensures 72% LR is not exceeded 90% of the time?"

Which one you use depends on whether the pricing committee is managing to an expected outcome or to a risk-adjusted outcome. Most UK motor books manage to expected LR targets with separate stress testing. The stochastic formulation is more appropriate for boards with formal risk appetite statements about LR exceedance.