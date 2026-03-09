## Part 12b: Stage 8.5 -- Calibration testing

The conformal intervals in Stage 8 tell you how uncertain the model's predictions are. Calibration testing asks a different question: are those predictions at the right level? A model can have well-behaved conformal intervals and still be systematically charging 8% too little, because the balance property and auto-calibration are not properties that conformal prediction addresses.

This stage runs three calibration checks on the frequency model's test-set predictions before rate optimisation. The results feed the audit record and, if the Murphy decomposition recommends it, inform whether you apply a multiplicative correction before handing predictions to the rate optimiser.

Install the library:

```python
%md
## Stage 8.5: Calibration testing -- balance property, auto-calibration, Murphy decomposition
```

```python
# In Databricks, install insurance-calibration in the cluster init script or:
# %pip install insurance-calibration

from insurance_calibration import (
    balance_test,
    auto_calibration_test,
    murphy_decomposition,
    calibration_report,
)
import numpy as np
```

**Why calibration is a separate step from conformal prediction.** Conformal prediction works on the residuals of the severity model -- it produces intervals that are correct in coverage regardless of whether the point predictions are biased. Calibration testing works on the frequency model predictions directly. A frequency model that underpredicts by 7% will produce conformal intervals centred on predictions that are 7% too low. The intervals will have correct coverage, but the premiums will be wrong. These are separate failure modes.

---

### The balance test

The balance property requires that the sum of predicted claim counts equals the sum of actual claim counts across the test portfolio. It is the weakest calibration requirement and the most commercially important: a model that fails this test is charging the wrong total premium.

```python
# Frequency model predictions are claim counts (exposure * frequency rate).
# y_test: actual claim counts
# freq_pred_test: predicted claim counts from the frequency model
# w_test: earned exposure in policy-years

balance = balance_test(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    distribution="poisson",
    seed=42,
)

print(f"Balance ratio (alpha):  {balance.balance_ratio:.4f}")
print(f"95% CI:                 [{balance.ci_lower:.4f}, {balance.ci_upper:.4f}]")
print(f"Observed claims:        {balance.observed_total:,.0f}")
print(f"Predicted claims:       {balance.predicted_total:,.0f}")
print(f"Balanced:               {balance.is_balanced}")
```

**Reading the output.** `balance_ratio` = sum(actual) / sum(predicted). A ratio above 1.0 means the model is under-predicting: it is charging less premium than claims will cost. A ratio below 1.0 means the model is over-predicting. The 95% confidence interval is from 999 bootstrap resamples; a CI that excludes 1.0 is a definitive failure, not a borderline one.

The Poisson z-test underpins the p-value. Under the null, observed claim count is approximately Normal(predicted, predicted), giving a z-statistic. This is the actuarially standard test for frequency data; it is not the chi-squared test used in binary classification calibration.

For the synthetic motor dataset in this module, expect a balance ratio close to 1.0 on the held-out test year -- the model was trained on similar data. On a real portfolio after a period of claims inflation, you would commonly see ratios of 1.05-1.15 on out-of-time validation data.

---

### Auto-calibration: the stronger test

Global balance can be satisfied even when the model is systematically wrong within segments. A model that underprices high-risk policies by 15% and overprices low-risk policies by 15% will pass the balance test while cross-subsidising risk cohorts. Auto-calibration requires that each prediction decile is self-financing -- not just the aggregate.

GLMs with canonical links satisfy auto-calibration on training data by construction (the score equations enforce it). CatBoost Poisson models do not: the optimiser minimises Poisson deviance but does not enforce the calibration constraint at the prediction level. You should expect the auto-calibration test to be worth running on any GBM.

```python
auto_cal = auto_calibration_test(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    distribution="poisson",
    n_bins=10,
    method="bootstrap",   # Algorithm 1 from Brauer et al. (arXiv:2510.04556, 2025)
    seed=42,
)

print(f"p-value:                {auto_cal.p_value:.4f}")
print(f"Auto-calibrated:        {auto_cal.is_calibrated}")
print(f"Worst bin deviation:    {auto_cal.worst_bin_ratio:.3f}")

print("\nPer-decile breakdown (the reliability table):")
print(auto_cal.per_bin)
```

The `per_bin` output is a Polars DataFrame with one row per prediction decile: predicted mean, observed mean, ratio, and policy count. This is the reliability diagram in tabular form -- the first thing a reviewing actuary will ask for. A well-calibrated model has ratios close to 1.0 in every bin; a structurally miscalibrated model will show a monotone pattern (consistently too low in the upper bins, too high in the lower bins).

**Significance threshold.** For an initial model sign-off, use the default `significance_level=0.05`. Brauer et al. (2025) recommend `significance_level=0.32` for routine quarterly monitoring, because at 0.05, a one-standard-deviation deterioration has low detection probability. Use 0.05 here at pipeline validation; switch to 0.32 when this check runs as a monitoring step post-deployment.

---

### Murphy decomposition: recalibrate or refit?

When the model fails calibration, the next question is what to do. The Murphy decomposition answers this by splitting the total deviance into three components:

- **UNC** (Uncertainty): the baseline deviance of an intercept-only model. Fixed by the data -- you cannot improve it.
- **DSC** (Discrimination): how much better the model is than the intercept-only model, after removing calibration error. Essentially the Gini expressed as deviance reduction.
- **MCB** (Miscalibration): the excess deviance from wrong price levels. Splits further into:
  - **GMCB** (Global MCB): removable by multiplying all predictions by a single constant (cheap to fix)
  - **LMCB** (Local MCB): residual after balance correction (requires model refit or isotonic recalibration)

```python
murphy = murphy_decomposition(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    distribution="poisson",
)

print(f"Total deviance:     {murphy.total_deviance:.5f}")
print(f"Uncertainty (UNC):  {murphy.uncertainty:.5f}")
print(f"Discrimination:     {murphy.discrimination:.5f}  ({murphy.discrimination_pct:.1f}% of UNC)")
print(f"Miscalibration:     {murphy.miscalibration:.5f}  ({murphy.miscalibration_pct:.1f}% of UNC)")
print(f"  Global MCB:       {murphy.global_mcb:.5f}  <- fixable by scalar recalibration")
print(f"  Local MCB:        {murphy.local_mcb:.5f}  <- requires model refit")
print(f"\nVerdict: {murphy.verdict}")
```

**The verdict logic:**
- `MCB / UNC < 1%` and DSC > 0: **OK** -- calibration is acceptable, proceed to rate optimisation with the raw predictions.
- `GMCB >= LMCB`: **RECALIBRATE** -- the dominant error is a global scale shift. Multiply predictions by the balance ratio. This takes minutes and does not require a governance cycle.
- `LMCB > GMCB`: **REFIT** -- the error is in the model's shape. A scalar correction will not fix it. The model needs rebuilding or isotonic recalibration on a large holdout.

Getting this diagnosis wrong is expensive in either direction. "RECALIBRATE" costs an afternoon. "REFIT" costs weeks plus a governance cycle. The decomposition gives you an objective basis for the decision.

---

### Applying a balance correction if needed

If the verdict is RECALIBRATE, apply the multiplicative correction before passing predictions to the rate optimiser. The correction is a single scalar: sum(actual) / sum(predicted).

```python
from insurance_calibration import rectify_balance

if murphy.verdict == "RECALIBRATE":
    freq_pred_calibrated = rectify_balance(
        y_hat=freq_pred_test,
        y=y_test.astype(float),
        exposure=w_test,
        method="multiplicative",
    )
    correction_factor = freq_pred_calibrated[0] / freq_pred_test[0]
    print(f"Balance correction factor: {correction_factor:.4f}")
    print(f"Applied to all {len(freq_pred_test):,} predictions")

    # Verify the correction restores balance
    check = balance_test(y_test.astype(float), freq_pred_calibrated, w_test,
                         distribution="poisson", seed=42)
    print(f"Post-correction balance ratio: {check.balance_ratio:.4f}  "
          f"(target: 1.0000)")

    # Use corrected predictions downstream
    freq_pred_final = freq_pred_calibrated
else:
    freq_pred_final = freq_pred_test
    print(f"No correction applied. Verdict: {murphy.verdict}")
    print(f"Using raw frequency predictions for rate optimisation.")
```

The correction does not change the model's ranking or relative price structure. It multiplies all predictions by the same scalar, restoring global balance without touching discrimination. Run `balance_test` on the corrected predictions to confirm alpha = 1.0 before proceeding.

**Note for the pipeline.** In Stage 9, the rate optimiser uses `pure_premium`, which is `freq_rate_all * sev_pred_all`. If you apply a balance correction here, you should use `freq_pred_calibrated` in place of `freq_pred_test` when computing `pure_premium`. In this module's synthetic data, the correction factor will typically be close to 1.0 because the test year is drawn from the same distribution as training. On real data following a period of inflation or mix change, it may differ materially.

---

### Compact report

For the audit record, generate a single-call summary:

```python
cal_report = calibration_report(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    distribution="poisson",
    n_bins=10,
    seed=42,
)

# One-row Polars DataFrame: balance_ratio, balance_ci_lower, balance_ci_upper,
# balance_is_balanced, auto_cal_p, auto_cal_is_calibrated, murphy_dsc_pct,
# murphy_mcb_pct, murphy_gmcb, murphy_lmcb, verdict
print(cal_report.to_polars())
```

Store the `to_polars()` output and capture the key fields for the audit record (Part 14):

```python
# Fields to carry forward to the audit record
cal_balance_ratio   = float(balance.balance_ratio)
cal_balance_ok      = bool(balance.is_balanced)
cal_auto_p          = float(auto_cal.p_value)
cal_auto_ok         = bool(auto_cal.is_calibrated)
cal_murphy_verdict  = murphy.verdict
cal_murphy_dsc_pct  = round(float(murphy.discrimination_pct), 2)
cal_murphy_mcb_pct  = round(float(murphy.miscalibration_pct), 2)

print(f"Calibration summary for audit record:")
print(f"  Balance ratio:          {cal_balance_ratio:.4f}  ({'OK' if cal_balance_ok else 'FAIL'})")
print(f"  Auto-calibration p:     {cal_auto_p:.4f}  ({'OK' if cal_auto_ok else 'FAIL'})")
print(f"  Murphy verdict:         {cal_murphy_verdict}")
print(f"  Discrimination (% UNC): {cal_murphy_dsc_pct:.1f}%")
print(f"  Miscalibration (% UNC): {cal_murphy_mcb_pct:.1f}%")
```

**What the audit record entry looks like.** In Part 14, add these fields to the `audit_record` dict:

```python
# Add to audit_record in Part 14:
"cal_balance_ratio":    cal_balance_ratio,
"cal_balance_ok":       cal_balance_ok,
"cal_auto_p":           cal_auto_p,
"cal_auto_ok":          cal_auto_ok,
"cal_murphy_verdict":   cal_murphy_verdict,
"cal_murphy_dsc_pct":   cal_murphy_dsc_pct,
"cal_murphy_mcb_pct":   cal_murphy_mcb_pct,
```

A pipeline that produces a Murphy verdict of REFIT on the test year is a pipeline that should stop before rate optimisation. The audit record preserves the evidence. A pipeline that produces RECALIBRATE should log the correction factor so the reviewer knows the raw model predictions were adjusted before rate action and by how much.
