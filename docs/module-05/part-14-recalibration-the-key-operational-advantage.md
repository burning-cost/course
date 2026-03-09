## Part 14: Recalibration - the key operational advantage

The practical value of conformal prediction for insurance operations is that calibration is separable from training. When claim frequencies or severities drift, you do not need to retrain the base model. You recalibrate the predictor on recent business.

Retrain: 15-25 minutes for a CatBoost model on 60,000 observations.
Recalibrate: under 10 seconds on 2,000 observations.

In a new cell:

```python
%md
## Part 14: Recalibration
```

```python
import time

# Simulate a quarterly recalibration using only the most recent 2,000 observations
# from the calibration period
X_cal_arr = X_cal.reset_index(drop=True)
y_cal_arr = y_cal.reset_index(drop=True)

X_cal_recent = X_cal_arr.tail(2_000)
y_cal_recent = y_cal_arr.tail(2_000)

print(f"Recalibrating on {len(X_cal_recent):,} most recent calibration observations...")
t0 = time.time()
cp.calibrate(X_cal_recent, y_cal_recent)
recal_time = time.time() - t0
print(f"Recalibration complete in {recal_time:.2f} seconds")

# Check coverage after recalibration
diag_recal = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
recal_coverages = diag_recal["coverage"].to_list()
print(f"\nCoverage after recalibration on {len(X_cal_recent):,} observations:")
print(f"  Marginal: {sum(recal_coverages)/len(recal_coverages):.3f}")
print(f"  Min decile: {min(recal_coverages):.3f}")
print(f"  Max decile: {max(recal_coverages):.3f}")
```

**What you should see:** coverage after recalibration will be close to the full-calibration-set result. With 2,000 calibration observations, coverage estimates will have slightly higher variance (the coverage measurement is less precise with fewer calibration points) but should still pass the 85-95% check.

```python
# Compare full calibration (20,000 obs) vs recent-only (2,000 obs)
print("\nCalibration set size comparison:")
print(f"{'Size':<12} {'Marginal coverage':>20} {'Min decile coverage':>22}")
print(f"{'2,000 (recent)':.<12} {sum(recal_coverages)/len(recal_coverages):>20.3f} {min(recal_coverages):>22.3f}")
print(f"{'20,000 (full)':.<12} {sum(coverages)/len(coverages):>20.3f} {min(coverages):>22.3f}")
```

### When recalibration is sufficient vs when you need to retrain

**Recalibration restores coverage** when the base model's predictions are still directionally correct but the error scale has shifted. This covers:
- Claims inflation (all losses have risen by a percentage)
- Frequency drift (all risks have a higher claims rate by a constant multiplier)

**Recalibration fails** when the model's rankings have deteriorated - it no longer correctly identifies which risks are more or less likely to claim. The diagnostic: if recalibration restores marginal coverage (overall coverage returns to 90%) but coverage-by-decile remains poor (top decile stays at 75%), the model is calibrated at the portfolio level but not within deciles. The model itself is wrong about relative risk, not just about scale.

**The operational rule: recalibrate quarterly, retrain annually, unless coverage-by-decile shows a structural break sooner.**