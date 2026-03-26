## Part 13: Rebuilding the GLM with interactions

With the suggested interaction pairs identified, we refit the GLM jointly with all approved interactions.

### The difference between test_interactions and build_glm_with_interactions

`test_interactions` (which ran inside `detector.fit()`) tests each pair in isolation: for each candidate, it fits a GLM with just that interaction added and reports the deviance gain. This is correct for deciding which interactions to add.

`build_glm_with_interactions` fits one GLM with all approved interactions simultaneously. The joint deviance gain is typically smaller than the sum of the individual gains, because the interactions overlap: age × vehicle group and age × annual mileage share the age factor, so some of the gain from the second interaction was already captured by the first.

```python
from insurance_interactions import build_glm_with_interactions

# Refit the GLM with the recommended interaction pairs
enhanced_glm, comparison = build_glm_with_interactions(
    X=X,
    y=y,
    exposure=exposure_arr,
    interaction_pairs=suggested,
    family="poisson",
)

print("Model comparison:")
print(comparison)
```

**Expected output:**

```
shape: (2, 8)
┌───────────────────────────┬─────────────┬──────────┬──────────────┬──────────────┬─────────────────┬───────────────────┬──────────────┐
│ model                     ┆ deviance    ┆ n_params ┆ deviance_aic ┆ deviance_bic ┆ delta_deviance  ┆ delta_deviance_pct┆ n_new_params │
│ ---                       ┆ ---         ┆ ---      ┆ ---          ┆ ---          ┆ ---             ┆ ---               ┆ ---          │
│ str                       ┆ f64         ┆ i64      ┆ f64          ┆ f64          ┆ f64             ┆ f64               ┆ i64          │
╞═══════════════════════════╪═════════════╪══════════╪══════════════╪══════════════╪═════════════════╪═══════════════════╪══════════════╡
│ base_glm                  ┆ 98432.x     ┆ 19       ┆ 98470.x      ┆ 98627.x      ┆ 0.0             ┆ 0.0               ┆ 0            │
│ glm_with_interactions     ┆ 96850.x     ┆ 43       ┆ 96936.x      ┆ 97135.x      ┆ 1582.x          ┆ 1.61              ┆ 24           │
└───────────────────────────┴─────────────┴──────────┴──────────────┴──────────────┴─────────────────┴───────────────────┴──────────────┘
```

The exact numbers will vary slightly depending on the random seed and CANN training, but you should see:
- `delta_deviance` of several hundred to a few thousand (capturing the planted interactions)
- `n_new_params` reflecting the parameter cost of your suggested interactions
- `deviance_aic` and `deviance_bic` both lower for the interaction model (lower is better)

Note: the comparison table uses `deviance_aic` and `deviance_bic` (deviance-based information criteria: D + 2k and D + k·log(n)), not the standard AIC from R's `AIC()` function. The delta values are equivalent to standard delta-AIC/BIC and are what matter for model comparison.

### Inspect the interaction GLM coefficients

```python
# The enhanced_glm is a fitted glum GeneralizedLinearRegressor
# glm.coef_ excludes the intercept; add 1 for total parameter count
print(f"Base GLM parameters:      {len(glm_base.coef_) + 1}")
print(f"Enhanced GLM parameters:  {len(enhanced_glm.coef_) + 1}")
print()

# glum stores feature names in feature_names_ (not feature_names_in_)
# Convert to list once to avoid repeated conversion inside the comprehension
coef_name_list = list(enhanced_glm.feature_names_)
ix_cols  = [c for c in coef_name_list if c.startswith("_ix_")]
ix_coefs = [enhanced_glm.coef_[coef_name_list.index(c)] for c in ix_cols]

print("Interaction term coefficients:")
for name, coef in sorted(zip(ix_cols, ix_coefs), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:<50} {coef:+.4f}  (relativity: {np.exp(coef):.3f})")
```

**What this shows:** For a categorical × categorical interaction (e.g., `age_band × vehicle_group`), the library adds separate binary contrast columns for each non-reference level combination. A 6-level age band × 5-level vehicle group band produces 5 × 4 = 20 interaction columns, each named `_ix_age_band_{level}_X_vehicle_group_{level}`. For a categorical × continuous interaction, the columns are named `_ix_{cat_feature}_{level}_{cont_feature}`.

The feature names in glum are accessible via `enhanced_glm.feature_names_`. This is glum's own attribute; scikit-learn's `feature_names_in_` is not populated by glum. Use `enhanced_glm.feature_names_` — it is always available after fitting.

For the planted interaction (age band 17-21, vehicle group 41-50), you should see positive interaction coefficients in the region of +0.25 to +0.35, consistent with the planted 0.30 log-unit bump.
