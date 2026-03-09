## Part 7: The near-deterministic price problem

Before fitting any elasticity model, we need to check whether the data is good enough to support the estimation. This is not a formality. On many insurance datasets, the price is so tightly determined by risk factors that there is almost no residual variation left for DML to exploit. The check is mandatory.

### What the problem looks like

The DML estimator works by residualising the treatment (log price change) on the confounders, then using the residuals for identification. If the price change is almost entirely predictable from the risk features, the residuals have near-zero variance. The DML estimator becomes like an IV estimator with a weak instrument: the point estimates are noisy, the confidence intervals blow up, and the result is effectively unusable.

In insurance this happens naturally. The re-rating system sets the offer price as a nearly deterministic function of the technical premium, and the technical premium is itself a function of the observable risk features. If your renewals all went through the same system with the same uplift applied uniformly, the variation in price change across customers is almost entirely explained by risk features. There is no exogenous component left for identification.

The diagnostic measures this directly: it computes Var(D_tilde) / Var(D), the fraction of treatment variation that survives after conditioning on observables. If this is below 10%, the DML results cannot be trusted.

### Running the diagnostic

```python
%md
## Part 7: Treatment variation diagnostic
```

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df_renewals,
    treatment="log_price_change",
    confounders=confounders,
)
print(report.summary())
```

On the synthetic dataset (which was generated with `price_variation_sd=0.08`), you should see a variation fraction well above the 10% threshold and `weak_treatment=False`. The output looks like:

```sql
Treatment Variation Diagnostic
========================================
N observations:          50,000
Var(D):                  0.006823
Var(D̃):                 0.003801
Var(D̃)/Var(D):          0.5570  (OK)
Treatment nuisance R²:   0.4430  (OK)

Treatment variation is sufficient for DML identification.
```

A Var(D_tilde)/Var(D) of 0.557 means 55.7% of the price change variation is not explained by observable confounders. That is healthy. On real data, especially from insurers that have not run any A/B pricing experiments, this fraction can be 0.05 or lower.

Now simulate what a near-deterministic price dataset looks like:

```python
# Generate data with near-zero exogenous price variation
df_ndp = make_renewal_data(n=50_000, seed=42, near_deterministic=True)

report_ndp = diag.treatment_variation_report(
    df_ndp,
    treatment="log_price_change",
    confounders=confounders,
)
print(report_ndp.summary())
```

You will see `weak_treatment=True` with Var(D_tilde)/Var(D) below 0.10. The report will print the standard remedies:

1. Run randomised A/B price tests (gold standard)
2. Use panel data with within-customer variation over multiple renewals
3. Exploit bulk re-rating quasi-experiments where a uniform uplift was applied to the line
4. Use rate change timing heterogeneity (customers with different anniversary dates)
5. Exploit the PS21/5 regression discontinuity for customers who were near the ENBP boundary -- **this approach requires that proximity to the ENBP boundary is unrelated to renewal probability conditional on covariates, an assumption that is unlikely to hold in practice. The ENBP threshold is a function of the customer's own risk characteristics, so customers near the boundary are not a random subset. Use only if you can argue this condition holds for your book.**

If your real data fails this check, do not proceed to the DML fitting. The point estimates will be meaningless. Take the report to the pricing director and use it to make the case for an A/B pricing experiment or a formal quasi-experimental design.

### The calibration summary check

As a secondary sanity check, look at the raw renewal rate by decile of price change:

```python
cal_summary = diag.calibration_summary(
    df_renewals,
    outcome="renewed",
    treatment="log_price_change",
    n_bins=10,
)
print(cal_summary)
```

In a well-identified dataset, the renewal rate should fall monotonically as the price change increases. If you see no monotone relationship - or a positive relationship (higher price change associated with higher renewal) - there is severe confounding that the diagnostic may have missed.

On the synthetic data you should see a clean downward pattern: decile 1 (smallest price increases) has the highest renewal rate, decile 10 (largest price increases) has the lowest. This is the signal that DML will exploit.