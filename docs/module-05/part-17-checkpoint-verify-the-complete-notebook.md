## Part 17: Checkpoint - verify the complete notebook

Before moving to the exercises, check that your notebook has completed successfully.

Run this verification cell:

```python
# Checkpoint: verify all components have run correctly
print("Module 5 checkpoint")
print("=" * 40)

# 1. Base model
try:
    test_pred = model.predict(test_pool)
    print(f"[PASS] Base model: predicts {len(test_pred):,} test observations")
except Exception as e:
    print(f"[FAIL] Base model: {e}")

# 2. Conformal predictor calibrated
try:
    n_cal_scores = len(cp.calibration_scores_)
    print(f"[PASS] Conformal predictor: {n_cal_scores:,} calibration scores stored")
except Exception as e:
    print(f"[FAIL] Conformal predictor: {e}")

# 3. Coverage check
try:
    diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
    min_cov = diag["coverage"].min()
    status  = "PASS" if min_cov >= 0.85 else "FAIL"
    print(f"[{status}] Coverage: minimum decile coverage = {min_cov:.3f} (threshold: 0.85)")
except Exception as e:
    print(f"[FAIL] Coverage diagnostic: {e}")

# 4. MLflow logging
try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    run    = client.get_run(conf_run_id)
    print(f"[PASS] MLflow run '{run.info.run_name}' logged (run_id: {conf_run_id[:8]}...)")
except Exception as e:
    print(f"[FAIL] MLflow: {e}")

print()
print("If all four checks show [PASS], proceed to the exercises.")
print("If any show [FAIL], rerun the relevant Part section above.")
```

---

## What just happened - a summary before the exercises

You have built a complete conformal prediction pipeline for UK motor insurance data.

The pipeline has five components:

1. **Base model** (Tweedie CatBoost, `p=1.5`): produces point estimates of pure premium. Trained on the oldest 60% of the data. Early stopping uses the calibration pool as validation.

2. **Conformal predictor** (Pearson-weighted non-conformity score): stores the sorted calibration residuals as a lookup table. Takes under 10 seconds to calibrate. Separable from model training: you can recalibrate without retraining.

3. **Coverage diagnostic** (coverage-by-decile): the mandatory check that intervals are valid across the risk spectrum, not just on average. Must pass before using intervals downstream.

4. **Downstream applications**: underwriting referral flag (relative width threshold), minimum premium floors (conformal upper bound), portfolio reserve ranges (naive and independence aggregation).

5. **Operational loop**: coverage monitoring table accumulates over time. Quarterly recalibration refreshes the coverage. Annual retrain refreshes the base model. Coverage-by-decile drives the decision on which action is needed.

The eight documented limitations are the governance deliverable: this is what you present to the FCA or the reserving committee when they ask about the mathematical basis and the assumptions. The more precisely you can state the limitations, the more credible the methodology is.

---

## What comes next

Module 6 covers credibility and Bayesian methods. The thin cells that produce wide conformal intervals in this module are the same thin cells where credibility blending matters most: cells with too few observations to produce a reliable estimate from first principles. Bayesian hierarchical models with informative priors borrow strength from similar cells, which in turn narrows the prediction intervals without sacrificing coverage. The two methodologies are complementary, not competing.