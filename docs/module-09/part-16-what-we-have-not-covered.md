## Part 16: What we have not covered

Demand modelling is a deep topic. This module has covered the core pipeline that most UK pricing teams need. Several important extensions exist that are worth knowing about, even if we have not implemented them here.

### Survival models for CLV

The `RetentionModel` with `model_type='cox'` or `model_type='weibull'` produces a survival function: the probability of still being a customer at times t=1, 2, 3, 5 years. This is necessary for Customer Lifetime Value calculations. The logistic model only tells you about the next renewal. If you are pricing for CLV (which is increasingly the right commercial framework), you need the survival model.

The survival model requires a `duration_col` column (years since first policy) and handles mid-term cancellations correctly as censored observations. Fitting it is straightforward:

```python
# Example only - not run in this module
# retention_survival = RetentionModel(
#     model_type="cox",
#     duration_col="tenure_years",
#     feature_cols=["ncd_years", "payment_method", "channel", "claim_last_3yr"],
# )
# retention_survival.fit(df_with_duration)
# survival_curves = retention_survival.predict_survival(df, times=[1, 2, 3, 5])
```

### Instrumental variables

When the near-deterministic price problem is present but you have a valid instrument - an external variable that affects price but is independent of individual renewal probability - you can use the IV variant of DML (PLIV). The `ElasticityEstimator` in `insurance_optimise.demand` supports this via the `instrument_col` parameter.

Valid instruments in practice include:
- A bulk rate change indicator (all policies subject to a Q1 2024 10% increase share this exogenous variation)
- A regulatory change dummy (PS21/5 caused forced price reductions for customers above ENBP)
- A competitor price shock (if a major competitor withdrew from the market, causing price spikes on comparison sites in certain regions)

### Multi-product demand

If you sell multiple lines (motor and home, for example), there are cross-price effects: a customer who lapses their motor policy may also cancel their home policy. Single-product demand models miss these. The `insurance_optimise.demand` submodule currently handles single-product demand; multi-product demand is a research-stage problem.