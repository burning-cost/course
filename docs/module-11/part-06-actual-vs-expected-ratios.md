## Part 6: Actual vs Expected ratios

### The simplest and most important metric

The A/E ratio is the ratio of observed claims to predicted claims. A model that predicts 1,000 claims in a month and sees 1,040 actual claims has an A/E of 1.04. That is within normal statistical noise for a portfolio of that size. A model predicting 1,000 claims and seeing 1,200 has an A/E of 1.20 - that is a problem.

```text
A/E = sum(actual claims) / sum(expected claims)
```

The A/E is computed at portfolio level first, then broken down by segment. Portfolio-level A/E is the calibration test. Segment-level A/E is the diagnostic.

### Why statistical confidence intervals matter

With 1,000 expected claims, you would naturally see variation around 1.0 due to random noise alone. The question is whether the observed A/E is outside the range you would expect from random noise - or whether it represents a genuine systematic shift.

The `AERatio` class computes a 95% confidence interval for the A/E under a Poisson assumption. If the expected claim count is E and the observed count is A:

- Under the null hypothesis (model is correct), A ~ Poisson(E)
- The 95% confidence interval for A/E is approximately [A/E - 1.96*sqrt(A)/E, A/E + 1.96*sqrt(A)/E]

If 1.0 is inside the 95% confidence interval, there is no statistically significant evidence of a calibration shift.

> **Note on the CI formula:** We use `sqrt(A)` (observed variance) rather than `sqrt(E)` (null-hypothesis variance). Under the null, A ~ Poisson(E) and the exact variance is E, so the null-hypothesis CI would use `sqrt(E)`. The Wald CI using `sqrt(A)` is equivalent when A/E is close to 1.0. The difference becomes meaningful when A/E is far from 1.0 — for example, at A/E = 1.30 with E = 1,000, using `sqrt(A)` gives a slightly wider CI than the exact Poisson CI. For routine monitoring, the Wald approximation is adequate; for near-threshold decisions, be aware the CI may be marginally conservative.

### A note on development lag

For the purposes of frequency monitoring, we assume claims from the current period are largely complete at the time of the run. For lines with significant IBNR — commercial liability, professional indemnity, or any long-tailed motor book — apply development factors to the actual claim count before computing the A/E ratio. Monitoring against undeveloped actuals will show a spurious downward A/E trend that has nothing to do with model drift.

### Portfolio-level A/E

```python
from insurance_monitoring import AERatio

ae_calc = AERatio()

# Get actual claims and expected (predicted) counts
actual_ref = df_reference["claim_count"].to_numpy().astype(float)
actual_cur = df_current["claim_count"].to_numpy().astype(float)

# Expected = predicted frequency * exposure
expected_cur = pred_cur * exposure_cur

ae_result = ae_calc.calculate(
    actual=actual_cur,
    expected=expected_cur,
    exposure=exposure_cur,
)

print(f"Portfolio A/E ratio: {ae_result.ratio:.4f}")
print(f"95% CI:              [{ae_result.ci_lower:.4f}, {ae_result.ci_upper:.4f}]")
print(f"Actual claims:       {actual_cur.sum():.0f}")
print(f"Expected claims:     {expected_cur.sum():.1f}")
print(f"Traffic light:       {ae_result.traffic_light}")
```

An A/E ratio with a confidence interval that includes 1.0 is not evidence of a calibration problem. An A/E ratio with a confidence interval entirely above 1.05 is evidence of systematic under-prediction. Report the confidence interval alongside the point estimate - the point estimate alone is meaningless without knowing how much noise to expect.

### Segment-level A/E breakdown

The portfolio A/E can mask offsetting errors. If the model is over-predicting for young drivers and under-predicting for older drivers, the portfolio A/E might look fine while both segments have problems. Compute A/E by segment:

```python
# Define segments to analyse
segments = {
    "driver_age_band": [
        (17, 25, "17-24"),
        (25, 40, "25-39"),
        (40, 60, "40-59"),
        (60, 100, "60+"),
    ],
    "region": None,           # None means use unique values
    "vehicle_age_band": [
        (0, 3, "0-2 years"),
        (3, 7, "3-6 years"),
        (7, 15, "7-14 years"),
        (15, 100, "15+ years"),
    ],
}

# Compute A/E for driver age bands
print("\nA/E by driver age band:")
print(f"{'Band':<15} {'A/E':>8}  {'CI lower':>10}  {'CI upper':>10}  {'Actual':>8}  {'Expected':>10}")
print("-" * 70)

df_cur_with_preds = df_current.with_columns([
    pl.Series("expected", expected_cur),
])

for low, high, label in segments["driver_age_band"]:
    mask = (
        (df_cur_with_preds["driver_age"] >= low) &
        (df_cur_with_preds["driver_age"] < high)
    )
    seg = df_cur_with_preds.filter(mask)

    if seg.shape[0] == 0:
        continue

    result = ae_calc.calculate(
        actual=seg["claim_count"].to_numpy().astype(float),
        expected=seg["expected"].to_numpy(),
        exposure=seg["exposure"].to_numpy(),
    )
    print(f"{label:<15} {result.ratio:>8.4f}  {result.ci_lower:>10.4f}  "
          f"{result.ci_upper:>10.4f}  {seg['claim_count'].sum():>8.0f}  "
          f"{seg['expected'].sum():>10.1f}")
```

Run the same loop for region and vehicle age band. The purpose is to find systematic errors in specific segments, not random noise.

### What to look for in the segment breakdown

A segment A/E consistently above 1.10 (with a confidence interval that excludes 1.0) means the model is systematically under-predicting for that segment. This could be:

1. A real change in risk for that segment (concept drift in a segment)
2. A book mix shift within the segment (e.g., more urban young drivers, who are higher risk than suburban young drivers of the same age)
3. A data quality issue for that segment

Before concluding it is concept drift, check whether the segment has grown or shrunk. If it has grown significantly (via CSI), the model's prediction may be fine but the segment now represents a larger share of the portfolio and small errors matter more.

### Year-on-year comparison

For a more informative view, compute A/E in rolling 3-month windows across the monitoring period. This shows trend rather than a single-point comparison:

```python
# Compute monthly A/E - requires a date column
df_monthly = df_current.with_columns([
    pl.Series("expected", expected_cur),
    pl.col("policy_start_date").dt.strftime("%Y-%m").alias("month"),
])

monthly_ae = (
    df_monthly
    .group_by("month")
    .agg([
        pl.col("claim_count").sum().alias("actual"),
        pl.col("expected").sum().alias("expected"),
        pl.col("exposure").sum().alias("exposure"),
    ])
    .sort("month")
)

print("\nMonthly A/E trend:")
print(f"{'Month':<10} {'A/E':>8}  {'Actual':>8}  {'Expected':>10}  {'Exposure':>10}")
print("-" * 55)

for row in monthly_ae.iter_rows(named=True):
    ae = row["actual"] / row["expected"] if row["expected"] > 0 else float("nan")
    print(f"{row['month']:<10} {ae:>8.4f}  {row['actual']:>8.0f}  "
          f"{row['expected']:>10.1f}  {row['exposure']:>10.1f}")
```

A rising trend in monthly A/E (1.00, 1.02, 1.05, 1.09) is more concerning than a stable elevated A/E (1.05, 1.05, 1.04, 1.05), even if the average is similar. The trend means the gap is widening.

Store the result:

```python
# Store for MonitoringReport
ae_portfolio = ae_result
```
