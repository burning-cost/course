## Part 4: The demand model

### Why you need one

The LR constraint and the retention constraint both depend on how many customers renew at the new rates. Without a demand model, you have to assume either:

- Everyone renews regardless of price (unrealistic: volume does not change)
- Lapse rates are fixed regardless of price (unrealistic: rates do not affect who stays)

Neither is right. A demand model tells the optimiser: if you raise this customer's premium by 5%, the probability they renew changes from, say, 68% to 65%. The optimiser accounts for this when computing expected LR and expected retention.

Without a demand model, the retention constraint is not meaningful, and the LR constraint is overoptimistic: it ignores the fact that rate increases cause lapses, which change the book composition.

### The log-linear demand model

`insurance-optimise` uses a log-linear (constant elasticity) demand model by default:

```
x(m) = x0 * m^epsilon
```

where:
- `m` is the price multiplier (new premium / technical price)
- `x0` is the baseline demand probability at `m = 1` (current rates)
- `epsilon` is the price elasticity: `d(log x) / d(log m)`, always negative

This says: a 1% increase in the price multiplier causes an `epsilon`% change in demand. For `epsilon = -2.0`, a 1% price increase causes a 2% reduction in renewal probability.

The model is parameterised per-policy. PCW customers are more price-sensitive than direct customers, so they get a more negative elasticity. Longer-tenured customers are stickier, so their elasticity is slightly less negative.

### What the optimiser needs

Two arrays:
- `p_demand`: baseline demand probability per policy at current rates, shape `(N,)`, values in `(0, 1)`
- `elasticity`: price elasticity per policy, shape `(N,)`, values typically in `(-3.5, -0.5)` for UK motor

You compute these upstream from your lapse model and price sensitivity analysis. In the synthetic data:

```python
# PCW elasticity -2.0, direct -1.2
elasticity = np.where(channel == "PCW", -2.0, -1.2)
# Tenure stickiness adjustment
elasticity = elasticity + 0.03 * tenure
elasticity = np.clip(elasticity, -3.5, -0.5)

# Baseline renewal probability at current rates (from your logistic lapse model)
log_price_ratio = np.log(current_premium / market_premium)
logit_p = 1.2 + (-2.0) * log_price_ratio + 0.05 * tenure
p_demand = 1.0 / (1.0 + np.exp(-logit_p))
p_demand = np.clip(p_demand, 0.05, 0.95)
```

These arrays go directly into `PortfolioOptimiser`. There is no separate demand model object to instantiate.

### The logistic demand model (alternative)

Pass `demand_model="logistic"` to `PortfolioOptimiser` to use the logistic specification instead:

```
x(m) = 1 / (1 + exp(alpha_i + beta_i * m_i * tc_i))
```

The library derives the logistic parameters `alpha_i` and `beta_i` from the same `p_demand` and `elasticity` inputs you pass, so the interface is identical. The logistic model is bounded (demand stays in `(0, 1)` by construction), which is more theoretically grounded for renewal modelling. The log-linear model is faster and produces cleaner gradients.

For a 5,000-policy book the difference is negligible. Use `log_linear` (the default) unless you have a specific reason to prefer the logistic specification.

### For UK motor, what elasticities are realistic?

The relevant benchmarks from market research and published lapse analyses are:

- **PCW (price comparison website) channel**: price elasticity typically -1.5 to -3.0. PCW customers have already demonstrated they will shop around. They are the most price-sensitive segment.
- **Direct channel**: -0.5 to -1.5. Direct customers have already chosen not to use a PCW. A modest rate increase is less likely to trigger a lapse.

These are starting points. You must calibrate against your own observed lapse data before running the optimiser. A miscalibrated elasticity produces rate strategies that look good in the model and fail in market.

### What miscalibration looks like

If you use a PCW elasticity of -2.5 when your actual elasticity is -1.2, the optimiser will believe you have far less pricing power than you do. It will think even a small rate increase causes a large volume loss, and it will constrain the rate action more than necessary.

If you use -0.8 when your actual elasticity is -2.0, the optimiser will overestimate pricing power. It will produce solutions that claim high retention at tight LR targets, but in practice the actual lapses will be much higher.

**The demand model must be calibrated before you run the optimiser.** Exercise 1 includes a calibration check.
