## Part 5: Extracting SHAP values

Now we set up the `SHAPRelativities` class and compute SHAP values.

In a new cell, type this and run it (Shift+Enter):

```python
sr = SHAPRelativities(
    model=freq_model,
    X=X_pd,
    exposure=exposure_pd,
    categorical_features=CAT_FEATURES,
    continuous_features=CONT_FEATURES,
    feature_perturbation="tree_path_dependent",
)

sr.fit()
print("SHAP values computed.")
```

You will see:

```
SHAP values computed.
```

This takes 15-45 seconds. The `fit()` call computes SHAP values for all 100,000 observations using CatBoost's native TreeSHAP implementation.

**What `feature_perturbation="tree_path_dependent"` means:** This tells TreeSHAP how to handle the background distribution when computing feature contributions. The two options are:

- `"tree_path_dependent"` (what we are using): uses the training data distribution as it was seen by the tree structure. Fast, no background dataset needed. This is the correct choice for most portfolios.
- `"interventional"`: marginalises over features independently using a separate background dataset. More theoretically rigorous when features are strongly correlated. About 10-50x slower.

For a UK motor book where `driver_age`, `vehicle_group`, and `ncd_years` have modest correlations, `tree_path_dependent` gives reliable relativities. Exercise 2 in the exercises file asks you to compare the two approaches.

**What `categorical_features` and `continuous_features` tell the class:** These lists control how the library aggregates SHAP values. For categorical features, it groups SHAP values by level and computes exposure-weighted means. For continuous features, it fits a smoothed curve through the per-observation SHAP values. The same feature cannot appear in both lists.