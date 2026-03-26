## Part 12b: Stage 8.5 — Calibration testing

Conformal intervals tell you how uncertain the model's predictions are. Calibration testing asks a different question: are those predictions at the right level?

A model can produce correctly sized conformal intervals while being systematically miscalibrated. If the frequency model underpredicts by 8%, the intervals will be centred on predictions that are 8% too low. The intervals will have correct coverage — the actual values will fall within them — but the premiums derived from the predictions will be wrong. Conformal prediction and calibration testing address separate failure modes.

This stage runs three checks on the frequency model before handing its predictions to the rate optimiser.

Add a markdown cell:

```python
%md
## Stage 8.5: Calibration testing
```

```python
from insurance_monitoring.calibration import CalibrationChecker
```

### Running the full calibration check

`CalibrationChecker` runs the balance property test, auto-calibration test, and Murphy decomposition in a single call, then returns a structured verdict.

```python
checker = CalibrationChecker(distribution="poisson")
cal_report = checker.check(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    seed=42,
)

print(cal_report.verdict())
print()
print(cal_report)
```

**What the three checks measure:**

**Balance property.** Does `sum(predicted) == sum(actual)`? The balance ratio is `sum(actual) / sum(predicted)`. A ratio above 1.0 means the model is undercharging relative to actual claims. The 95% CI uses 999 bootstrap resamples; a CI that excludes 1.0 is definitive failure.

For the synthetic data, the balance ratio should be close to 1.0. On real data after a period of claims inflation, expect ratios of 1.05-1.15 on out-of-time validation data — the model was trained on pre-inflation claims and is being tested on post-inflation ones.

**Auto-calibration.** Is the model self-financing within each prediction decile? A model can pass the balance test while systematically undercharging high-risk policies and overcharging low-risk ones. Auto-calibration fails when any decile's actual-to-predicted ratio diverges materially from 1.0. CatBoost Poisson models do not satisfy auto-calibration on training data by construction (unlike GLMs with canonical links), so this check is worth running.

**Murphy decomposition.** Decomposes total deviance into:

- **UNC** (Uncertainty): baseline deviance of an intercept-only model. Fixed — cannot be improved.
- **DSC** (Discrimination): how much better the model is than the null model, after removing calibration error. Equivalent to the Gini expressed as deviance reduction.
- **MCB** (Miscalibration): excess deviance from wrong price levels.
  - **GMCB** (Global MCB): removable by multiplying all predictions by a constant.
  - **LMCB** (Local MCB): residual after balance correction; requires model refit.

### Reading the verdict

```python
# Extract fields for downstream use and the audit record
cal_balance_ratio  = cal_report.balance.balance_ratio
cal_balance_ok     = cal_report.balance.is_balanced
cal_auto_p         = cal_report.auto_calibration.p_value
cal_auto_ok        = cal_report.auto_calibration.is_calibrated
cal_murphy_verdict = cal_report.murphy.verdict   # "OK", "RECALIBRATE", or "REFIT"
cal_dsc_pct        = round(cal_report.murphy.discrimination_pct, 2)
cal_mcb_pct        = round(cal_report.murphy.miscalibration_pct, 2)

print(f"Balance ratio:       {cal_balance_ratio:.4f}  ({'OK' if cal_balance_ok else 'FAIL'})")
print(f"Auto-calibration p:  {cal_auto_p:.4f}  ({'OK' if cal_auto_ok else 'FAIL'})")
print(f"Murphy verdict:      {cal_murphy_verdict}")
print(f"Discrimination:      {cal_dsc_pct:.1f}% of UNC")
print(f"Miscalibration:      {cal_mcb_pct:.1f}% of UNC")
```

**Verdict logic:**

| Verdict | Condition | Action |
|---------|-----------|--------|
| `OK` | MCB/UNC < 1% and DSC > 0 | Proceed with raw predictions |
| `RECALIBRATE` | GMCB >= LMCB | Multiply predictions by balance ratio — takes minutes |
| `REFIT` | LMCB > GMCB | Miscalibration is in the model's shape, not just level — rebuild |

A pipeline that returns `REFIT` on the test year should stop before rate optimisation. That is not a pipeline failure — it is the pipeline doing its job. The rate optimiser should not receive miscalibrated inputs.

### Applying a balance correction if needed

```python
if cal_murphy_verdict == "RECALIBRATE":
    from insurance_monitoring.calibration import rectify_balance

    freq_pred_final = rectify_balance(
        y_hat=freq_pred_test,
        y=y_test.astype(float),
        exposure=w_test,
        method="multiplicative",
    )
    correction_factor = float(freq_pred_final.mean() / freq_pred_test.mean())
    print(f"Balance correction applied: {correction_factor:.4f}")

    # Verify: re-run balance check on corrected predictions
    verify = CalibrationChecker(distribution="poisson")
    v_report = verify.check(y_test.astype(float), freq_pred_final, w_test, seed=42)
    print(f"Post-correction balance ratio: {v_report.balance.balance_ratio:.4f}  (target: 1.000)")

elif cal_murphy_verdict == "REFIT":
    print("REFIT verdict: stop the pipeline before rate optimisation.")
    print("Investigate: feature distribution shift, IBNR contamination, or data quality.")
    raise RuntimeError(
        f"Frequency model failed calibration check (verdict: REFIT). "
        f"MCB = {cal_mcb_pct:.1f}% of UNC. Refit the model before proceeding."
    )

else:
    freq_pred_final = freq_pred_test
    print(f"No correction needed (verdict: {cal_murphy_verdict}).")
    print(f"Using raw predictions for rate optimisation.")
```

The balance correction does not change the model's relative price structure. It multiplies all predictions by the same scalar, shifting the overall level. The discrimination (Gini, rank ordering) is unchanged. After correction, re-run the balance check to confirm the ratio is at 1.000 before proceeding to Stage 9.
