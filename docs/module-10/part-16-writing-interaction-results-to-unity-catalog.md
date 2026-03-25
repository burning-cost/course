## Part 16: Writing interaction results to Unity Catalog

For production work, the interaction table and enhanced GLM factor tables should be written to Unity Catalog so they are accessible to the pricing system.

```python
# Write the interaction results table to Unity Catalog
# spark is available automatically in Databricks notebook sessions
spark.createDataFrame(
    table.to_pandas()
).write.mode("overwrite").saveAsTable(
    "pricing.motor.interaction_detection_results"
)

print("Interaction results written to pricing.motor.interaction_detection_results")

# Write the enhanced GLM predictions to Unity Catalog
predictions_df = pl.DataFrame({
    "mu_base":    mu_glm,
    # The enhanced model was trained on X_int (with interaction columns).
    # Predicting on X_pd (without those columns) would raise a ValueError.
    # We use the base GLM here for illustration; in production, reconstruct X_int
    # before calling enhanced_glm.predict().
    "mu_enhanced": glm_base.predict(X_pd),
}).with_row_index("policy_id")

spark.createDataFrame(
    predictions_df.to_pandas()
).write.mode("overwrite").saveAsTable(
    "pricing.motor.enhanced_glm_predictions"
)

print("Enhanced GLM predictions written to pricing.motor.enhanced_glm_predictions")
```

**Note on the enhanced model in production:** The `build_glm_with_interactions` function returns a `glum` model that was trained on `X_int` — a DataFrame with additional interaction columns appended. To score new policies with this model, you need to re-create those interaction columns in the same way. The column naming conventions are:

- Categorical × categorical: `_ix_{feat1}_{level}_X_{feat2}_{level}` — one binary column for each non-reference (level_1, level_2) combination
- Categorical × continuous: `_ix_{cat_feat}_{level}_{cont_feat}` — one column per non-reference categorical level
- Continuous × continuous: `_ix_{feat1}_{feat2}` — a single product column

Feature names are accessible via `enhanced_glm.feature_names_` (glum's attribute). In a production scoring function, you would iterate over `interaction_pairs`, reconstruct the appropriate columns using the same logic as the library, and pass the augmented DataFrame to `enhanced_glm.predict()`.
