## Part 8: Calibrating the conformal predictor

Now we calibrate the predictor. This step is fast - it runs on the calibration set once and stores the sorted scores.

In a new cell:

```python
%md
## Part 8: Calibrating the conformal predictor
```

```python
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

cp.calibrate(X_cal, y_cal)

# Inspect the calibration scores
cal_scores = cp.calibration_scores_
print(f"Calibration set size: {len(X_cal):,}")
print(f"Number of calibration scores: {len(cal_scores):,}")
print(f"\nCalibration score distribution:")
print(f"  Min:    {cal_scores.min():.4f}")
print(f"  Median: {np.median(cal_scores):.4f}")
print(f"  90th percentile (alpha=0.10 quantile): {np.quantile(cal_scores, 0.90):.4f}")
print(f"  95th percentile (alpha=0.05 quantile): {np.quantile(cal_scores, 0.95):.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
```

**What this does:** creates the conformal predictor, links it to the trained base model, and runs calibration on the held-out calibration set. The calibration step computes the Pearson residual for every calibration observation and sorts them. The 90th percentile score becomes the threshold for 90% prediction intervals.

**What you should see:**

```
Calibration set size: 20,xxx
Number of calibration scores: 20,xxx

Calibration score distribution:
  Min:    0.0000
  Median: 0.xxxx
  90th percentile (alpha=0.10 quantile): 2.xxxx
  95th percentile (alpha=0.05 quantile): 3.xxxx
  Max:    xx.xxxx
```

The distribution of Pearson residuals is right-skewed: most calibration observations have small residuals (the model predicts them well) and a small number have very large residuals (zero-loss risks where the model predicts a positive premium, or rare large claims). This is expected for insurance data.

### What just happened

The `cp.calibration_scores_` array contains the Pearson residuals for every calibration observation, sorted from smallest to largest. To generate a 90% prediction interval for a new risk, the library will find the range of outcomes that would produce a Pearson residual below the 90th percentile. Any outcome within that range is "consistent" with the model's prediction at the 90% confidence level.

Think of it as the model saying: "for this risk, I predict £342. Looking at how wrong I was on similar calibration observations, and scaling by the expected variance, the range of outcomes I cannot rule out at 90% confidence is [£X, £Y]."