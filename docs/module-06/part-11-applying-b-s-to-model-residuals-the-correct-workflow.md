## Part 11: Applying B-S to model residuals (the correct workflow)

### Why raw rates are often the wrong input

The tutorial above applies credibility to raw observed claim frequencies. In practice, this is rarely the right application.

If you have already fitted a GLM or GBM on the full dataset, you have a model prediction for each district that already accounts for the other rating factors (vehicle group, NCD, driver age, etc.). The credibility question becomes: should the district factor from the model be adjusted based on the district's own experience relative to what the model expects?

You apply B-S to the district-level O/E ratios (observed over expected), not to raw observed rates. This gives a clean decomposition:

- The GLM or GBM handles main effects
- Credibility on residuals handles district-level adjustments on top of the main-effect model

Applying B-S to raw rates when a GBM has already partially handled thin cells produces double shrinkage. The GBM's regularisation (min_data_in_leaf, L2 leaf penalty) already shrinks thin-cell predictions toward the base rate. Applying B-S on raw rates then shrinks again. The correct approach:

```python
# After fitting your GLM or GBM, compute district-level O/E ratios
# Replace "model_predicted_claims" with your actual model predictions

# Example structure (your column names will differ):
# df has columns: postcode_district, accident_year, earned_years,
#                 claim_count, model_predicted_claims

dist_residuals = (
    df
    .with_columns([
        # O/E ratio: observed over expected per policy-year row
        (pl.col("claim_count") / pl.col("model_predicted_claims").clip(lower_bound=1e-6))
        .alias("oe_ratio")
    ])
    .group_by(["postcode_district", "accident_year"])
    .agg([
        pl.col("oe_ratio").mean().alias("oe_ratio"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
)

# Apply B-S to the O/E ratios in log space
# The credibility estimate is the district-level log-adjustment factor
# to apply multiplicatively on top of the model's predictions.
bs_residuals = buhlmann_straub(
    data=dist_residuals,
    group_col="postcode_district",
    value_col="oe_ratio",
    weight_col="earned_years",
    log_transform=True,   # log because we want a multiplicative adjustment
)

# Grand mean of O/E should be close to 1.0 for a well-calibrated model
print(f"Grand mean O/E ratio: {bs_residuals['grand_mean']:.4f}  (should be near 1.0 for calibrated model)")
print(f"K (for O/E residuals): {bs_residuals['k']:.1f}")
```

**When to use this pattern:** Any time you are refining a GLM or GBM with district-level experience adjustments. The main model handles the structural rating factors; credibility handles the district-level departure from the model's expectation.