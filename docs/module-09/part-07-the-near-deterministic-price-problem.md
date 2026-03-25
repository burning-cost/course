## Part 7: Fitting the DML elasticity estimator

The diagnostic is clear. Now fit the estimator.

`RenewalElasticityEstimator` wraps EconML's `CausalForestDML` with CatBoost nuisance models and insurance-specific defaults. It handles the 5-fold cross-fitting, the log-log treatment specification, and the interface between Polars DataFrames and the underlying numpy arrays.

### Why CausalForestDML rather than LinearDML

`LinearDML` (the PLR estimator from Chernozhukov et al. 2018) assumes a constant treatment effect across customers. It gives a single average semi-elasticity and is faster. `CausalForestDML` estimates heterogeneous treatment effects — a per-customer CATE — which is what you need for segment-level pricing decisions.

The ATE from `CausalForestDML` is the average of the per-customer CATEs. For portfolio-level price decisions the two estimators give similar ATEs. For building the elasticity surface (Part 9) and per-policy optimisation (Part 12), you need the CausalForestDML output.

### Fitting the estimator

```python
%md
## Part 7: DML elasticity estimation
```

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
    binary_outcome=True,
    random_state=42,
)

print("Fitting CausalForestDML with CatBoost nuisance models...")
print("(5–8 minutes on Databricks Free Edition)")

est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)
print("Done.")
```

The fit runs 5-fold cross-fitting for both the outcome nuisance model (CatBoostClassifier predicting renewal) and the treatment nuisance model (CatBoostRegressor predicting log price change). The CausalForestDML then uses the residuals from both models to estimate per-customer CATEs.

`n_estimators=200` must be divisible by `n_folds × 2 = 10` — the library will round up automatically if not. For a more accurate elasticity surface, increase to 400; for faster iteration during development, reduce to 100.

### The average treatment effect

```python
ate, lb, ub = est.ate()
true_ate = float(df["true_elasticity"].mean())

print(f"True ATE (DGP):       {true_ate:.3f}")
print(f"DML estimate:         {ate:.3f}")
print(f"95% CI:               [{lb:.3f}, {ub:.3f}]")
print(f"Naive logistic:       {naive_coef:.3f}")
print()
print(f"DML bias:             {abs(ate - true_ate):.3f}")
print(f"Naive bias:           {abs(naive_coef - true_ate):.3f}")
```

The DML estimate should be within the 95% CI of the true ATE (approximately −2.0). The naive logistic will typically be 20–40% more negative — the confounding bias demonstrated in Part 5. On 50,000 observations the DML 95% CI should be narrow enough to be commercially meaningful: on the order of ±0.15.

### Interpreting the semi-elasticity

The coefficient is the semi-elasticity on the linear probability scale. The treatment is log(offer_price / last_premium). The outcome is binary renewal.

An ATE of −2.0 means: a 1-unit increase in log price change reduces renewal probability by 2.0 percentage points. For practical price changes:

| Log price change | Price change | Renewal prob change |
|-----------------|--------------|---------------------|
| 0.0953 | +10% | −2.0 × 0.0953 = −0.19 pp |
| 0.0488 | +5%  | −2.0 × 0.0488 = −0.10 pp |
| −0.0513 | −5% | −2.0 × −0.0513 = +0.10 pp |

At a base renewal rate of 83%, a 10% price increase is expected to reduce renewal probability by around 0.19 percentage points at the portfolio average. That is the number that feeds the rate optimiser from Module 7. With the naive estimate (say −2.7), the optimiser would have predicted a 0.26 pp drop — overstating the demand response by 37%.
