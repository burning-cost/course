## Part 6: Pre-flight diagnostic — treatment variation

Before fitting the DML estimator, check whether the data has enough exogenous price variation to identify elasticity. This is not optional. On many insurance renewal datasets, the price change is so tightly determined by the rating system that there is almost nothing left for DML to use. Running the estimator on such data produces confident-looking but meaningless results.

### What the diagnostic measures

The DML estimator residualises the treatment D (log price change) on the confounders X. The residual D̃ = D − E[D|X] is the price variation not explained by risk factors. DML uses D̃ for identification.

The diagnostic reports two things:

- **Var(D̃) / Var(D)**: the fraction of price variation that survives conditioning on observables. Below 0.10 is the danger zone.
- **R² of E[D|X]**: how well the nuisance model predicts the treatment from risk factors. Above 0.90 means the price is nearly deterministic.

If either threshold is breached, the estimator's confidence intervals blow up and the point estimate is unreliable — the same failure mode as a weak instrument in an IV regression.

### Running the diagnostic

```python
%md
## Part 6: Treatment variation diagnostic
```

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=confounders,
)
print(report.summary())
```

On `make_renewal_data(price_variation_sd=0.08)`, you should see:

```
Treatment Variation Diagnostic
========================================
N observations:          50,000
Var(D):                  0.006823
Var(D̃):                 0.003801
Var(D̃)/Var(D):          0.5570  (OK)
Treatment nuisance R²:   0.4430  (OK)

Treatment variation is sufficient for DML identification.
```

A variation fraction of 0.557 means 55.7% of the price change variation is not explained by observable risk factors. Healthy. On real data without A/B pricing experiments, this fraction can easily be 0.05 or lower.

### Simulating the near-deterministic case

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

You will see `weak_treatment=True` with suggestions including: running randomised A/B price tests, using panel data with within-customer variation across renewal years, exploiting bulk re-rating quasi-experiments, and using rate change timing heterogeneity across anniversary dates.

If your real dataset fails this check, stop here. Do not run the DML estimator on it — the output will look like a proper result but the number is not trustworthy. Take the `TreatmentVariationReport` to the pricing director as the basis for a conversation about how to generate identifiable price variation.

### The calibration check

As a secondary sanity check, look at the raw renewal rate by decile of price change:

```python
cal_summary = diag.calibration_summary(
    df,
    outcome="renewed",
    treatment="log_price_change",
    n_bins=10,
)
print(cal_summary)
```

Renewal rate should fall as price change increases. If you see a flat or positive relationship, the confounding is severe enough that the raw data already shows no price signal. This is a red flag even if Var(D̃)/Var(D) is above the threshold.

On the synthetic data you will see a clean downward pattern: the lowest-price-change decile has the highest renewal rate, the highest-price-change decile the lowest. That monotone pattern is the signal DML will exploit.
