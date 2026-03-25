## Part 6: Actual vs Expected ratios

### The simplest and most important metric

The A/E ratio is the ratio of observed claims to predicted claims. A model that predicts 1,000 claims in a month and sees 1,040 actual claims has an A/E of 1.04. That is within normal statistical noise for a portfolio of that size. A model predicting 1,000 claims and seeing 1,200 has an A/E of 1.20 — that is a problem.

```text
A/E = sum(actual claims) / sum(expected claims)
```

The A/E is computed at portfolio level first, then broken down by segment. Portfolio-level A/E is the calibration test. Segment-level A/E is the diagnostic.

### Why statistical confidence intervals matter

With 1,000 expected claims, you would naturally see variation around 1.0 due to random noise alone. The question is whether the observed A/E is outside the range you would expect from random noise — or whether it represents a genuine systematic shift.

`ae_ratio_ci()` computes an exact Poisson confidence interval (Garwood method). Under the null hypothesis (model is correct), the observed claim count follows a Poisson distribution with mean equal to expected. This gives a rigorous CI without relying on the normal approximation, which fails when claim counts are below about 100.

If 1.0 is inside the 95% confidence interval, there is no statistically significant evidence of a calibration shift.

### Portfolio-level A/E

```python
from insurance_monitoring.calibration import ae_ratio_ci

ae_result = ae_ratio_ci(
    actual=actual_cur,
    predicted=pred_cur,
    exposure=exposure_cur,
    alpha=0.05,
    method="poisson",
)

# ae_ratio_ci returns a dict with keys: ae, lower, upper, n_claims, n_expected
print(f"Portfolio A/E ratio: {ae_result['ae']:.4f}")
print(f"95% CI:              [{ae_result['lower']:.4f}, {ae_result['upper']:.4f}]")
print(f"Actual claims:       {ae_result['n_claims']:.0f}")
print(f"Expected claims:     {ae_result['n_expected']:.1f}")
print()

if ae_result["lower"] > 1.0:
    print("CI excludes 1.0 from below. Model is systematically under-predicting.")
elif ae_result["upper"] < 1.0:
    print("CI excludes 1.0 from above. Model is systematically over-predicting.")
else:
    print("CI contains 1.0. No statistically significant evidence of calibration drift.")
```

An A/E ratio with a confidence interval that includes 1.0 is not evidence of a calibration problem. An A/E ratio with a confidence interval entirely above 1.05 is evidence of systematic under-prediction. Report the confidence interval alongside the point estimate — the point estimate alone is meaningless without knowing how much noise to expect.

### Segmented A/E

The portfolio A/E can mask offsetting errors. If the model is over-predicting for young drivers and under-predicting for older drivers, the portfolio A/E might look fine while both segments have problems. Compute A/E by segment using the plain `ae_ratio()` function (returns a float) or loop with `ae_ratio_ci()`:

```python
from insurance_monitoring.calibration import ae_ratio_ci
import polars as pl

age_bands = [(17, 25, "17-24"), (25, 40, "25-39"), (40, 60, "40-59"), (60, 100, "60+")]

# Add expected counts to the current DataFrame
df_cur_with_preds = df_current.with_columns([
    pl.Series("expected", pred_cur * exposure_cur),
])

print("\nA/E by driver age band:")
print(f"{'Band':<15} {'A/E':>8}  {'CI lower':>10}  {'CI upper':>10}  {'Actual':>8}  {'Expected':>10}")
print("-" * 70)

for low, high, label in age_bands:
    seg = df_cur_with_preds.filter(
        (pl.col("driver_age") >= low) & (pl.col("driver_age") < high)
    )
    if len(seg) == 0:
        continue

    result = ae_ratio_ci(
        actual=seg["claim_count"].to_numpy().astype(float),
        predicted=seg["expected"].to_numpy(),
        method="poisson",
    )
    print(f"{label:<15} {result['ae']:>8.4f}  {result['lower']:>10.4f}  "
          f"{result['upper']:>10.4f}  {seg['claim_count'].sum():>8.0f}  "
          f"{seg['expected'].sum():>10.1f}")
```

### A note on development lag

For the purposes of frequency monitoring, we assume claims from the current period are largely complete at the time of the run. For lines with significant IBNR — commercial liability, professional indemnity, or any long-tailed motor book — apply development factors to the actual claim count before computing the A/E ratio. Monitoring against undeveloped actuals will show a spurious downward A/E trend that has nothing to do with model drift.

### What to look for in the segment breakdown

A segment A/E consistently above 1.10 (with a confidence interval that excludes 1.0) means the model is systematically under-predicting for that segment. Before concluding it is concept drift, check whether the segment has grown or shrunk. If it has grown significantly (via CSI), the model's prediction may be fine but the segment now represents a larger share of the portfolio and small errors matter more.
