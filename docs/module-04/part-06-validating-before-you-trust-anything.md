## Part 6: Validating before you trust anything

This is the step most people skip. Do not skip it.

In a new cell, type this and run it (Shift+Enter):

```python
checks = sr.validate()

for check_name, result in checks.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {check_name}: {result.message}")
```

You will see:

```
[PASS] reconstruction: Max absolute reconstruction error: 7.4e-06.
[PASS] feature_coverage: All features covered by SHAP.
[PASS] sparse_levels: All factor levels have >= 30 observations.
```

Three checks run:

**reconstruction** - the critical one. It verifies that:

```
exp(shap_values.sum(axis=1) + expected_value) ≈ model.predict(pool)
```

If the SHAP values do not reconstruct the model predictions, something is wrong with the explainer setup. The most common cause is a mismatch between the model's loss function and how SHAP is applied. The `SHAPRelativities` class is designed to prevent this for CatBoost Poisson models, but the check is there to confirm it.

**If reconstruction FAILS:** Stop immediately. Do not extract relativities. The number you see (e.g., max error of 0.15) means SHAP values differ from model predictions by up to 15% in log space - which means the relativities are wrong. Check that `loss_function="Poisson"` was set in the model parameters, and that the `SHAPRelativities` object received the same `X` that was passed to `train_pool`.

**feature_coverage** - confirms every feature in `X` has a corresponding SHAP column. A mismatch here usually means a feature name typo.

**sparse_levels** - flags any categorical level with fewer than 30 observations. The CLT confidence intervals for sparse levels are unreliable. You will see a warning like `"Area G has only 12 observations: CI unreliable"` for any sparse level. This dataset has no sparse levels, but real portfolios often have thin cells in area bands or occupation codes.