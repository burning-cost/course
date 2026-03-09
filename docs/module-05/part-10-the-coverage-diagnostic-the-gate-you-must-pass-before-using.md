## Part 10: The coverage diagnostic - the gate you must pass before using intervals

Do not skip this step. Do not use intervals for minimum premiums, reserving, or underwriting referral without passing this check.

The coverage diagnostic answers: "do the 90% intervals actually achieve 90% coverage on test data, and is that coverage consistent across risk levels?"

In a new cell:

```python
%md
## Part 10: Coverage validation
```

```python
# Run the full coverage-by-decile diagnostic
diag_90 = cp.coverage_by_decile(X_test, y_test, alpha=0.10)

print("Coverage by decile (target: 0.90)")
print("Decile 1 = lowest predicted premium, Decile 10 = highest predicted premium")
print()
print(diag_90.to_string())
```

**What this does:** bins the test set into 10 equal-count groups by predicted pure premium (so decile 1 is the 10% of risks with the lowest predicted loss, decile 10 is the 10% with the highest predicted loss), then measures what fraction of each decile's actual outcomes fall inside the prediction interval.

**What you should see:** something like this:

```python
Coverage by decile (target: 0.90)
Decile 1 = lowest predicted premium, Decile 10 = highest predicted premium

┌────────┬──────────┬──────────────────┐
│ decile ┆ coverage ┆ n                │
│ ---    ┆ ---      ┆ ---              │
│ i64    ┆ f64      ┆ u32              │
╞════════╪══════════╪══════════════════╡
│ 1      ┆ 0.921    ┆ 2,xxx            │
│ 2      ┆ 0.912    ┆ 2,xxx            │
│ 3      ┆ 0.905    ┆ 2,xxx            │
│ 4      ┆ 0.897    ┆ 2,xxx            │
│ 5      ┆ 0.893    ┆ 2,xxx            │
│ 6      ┆ 0.889    ┆ 2,xxx            │
│ 7      ┆ 0.891    ┆ 2,xxx            │
│ 8      ┆ 0.886    ┆ 2,xxx            │
│ 9      ┆ 0.882    ┆ 2,xxx            │
│ 10     ┆ 0.878    ┆ 2,xxx            │
└────────┴──────────┴──────────────────┘
```

### Reading the diagnostic

**All deciles within 5pp of target (85-95% for a 90% interval).** This is a pass. The `pearson_weighted` score should achieve this on well-behaved insurance data.

**Monotone decline from bottom to top decile.** A small monotone decline (e.g. 92% in decile 1 to 88% in decile 10) is normal and acceptable - some residual heteroscedasticity in the scores is expected. If the decline is steep (e.g. 95% to 72%), the Pearson score has not fully normalised the variance and you should investigate whether the model has structural bias against large risks.

**Coverage below 85% in any decile.** This is a failure. The most common cause is distribution shift between calibration and test data: if the most recent business has had higher claims inflation, the calibration quantile (set on older business) will be too low for the test period. Recalibrate on more recent data (Part 14 covers this).

**Non-monotone pattern** (e.g. low in the middle deciles, high at the extremes). This usually indicates a specific risk segment in the test set that was absent from or underrepresented in calibration. Investigate which features differ between the problematic deciles and the well-covered deciles.

Now run the automated checks:

```python
coverages = diag_90["coverage"].to_list()
spread    = max(coverages) - min(coverages)
min_cov   = min(coverages)

print(f"\nMarginal coverage (all deciles combined): {sum(coverages)/len(coverages):.3f}")
print(f"Min decile coverage:  {min_cov:.3f}")
print(f"Max decile coverage:  {max(coverages):.3f}")
print(f"Coverage spread (max - min): {spread:.3f}")
print()

if spread > 0.10:
    print("WARNING: Coverage spread > 10pp. Try nonconformity='deviance' as an alternative.")
    print("If spread persists, the base model may have structural bias against large risks.")
elif min_cov < 0.85:
    print("WARNING: Minimum decile coverage below 85%. Check for distribution shift.")
    print("Recalibrate on more recent data before using intervals downstream.")
else:
    print("Coverage check PASSED. Intervals may be used for downstream applications.")
    print("Log these results to MLflow for audit.")
```

### Log the coverage diagnostics to MLflow

Coverage metrics are part of the model audit trail. Log them alongside the model parameters:

```python
with mlflow.start_run(run_name="module05_conformal_baseline") as run:
    conf_run_id = run.info.run_id

    # Log hyperparameters
    mlflow.log_params({
        "nonconformity_score":  "pearson_weighted",
        "tweedie_power":        1.5,
        "alpha_90":             0.10,
        "calibration_n":        len(X_cal),
        "model_depth":          5,
        "learning_rate":        0.05,
    })

    # Log coverage metrics
    mlflow.log_metric("marginal_coverage_90",   float(sum(coverages) / len(coverages)))
    mlflow.log_metric("min_decile_coverage_90", float(min(coverages)))
    mlflow.log_metric("max_decile_coverage_90", float(max(coverages)))
    mlflow.log_metric("coverage_spread_90",     float(spread))

    print(f"MLflow run ID: {conf_run_id}")
    print("Coverage metrics logged.")
```