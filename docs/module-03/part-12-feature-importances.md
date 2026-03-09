## Part 12: Feature importances

Before moving to the severity model, let us look at what the frequency model is relying on. Create a new cell:

```python
importances = freq_model.get_feature_importance(type="FeatureImportance")

imp_df = (
    pl.DataFrame({"feature": FEATURES, "importance": importances.tolist()})
    .sort("importance", descending=True)
)
print(imp_df)
```

Run this. You will see something like:

```
shape: (5, 2)
feature            importance
ncd_years          35.2
driver_age         28.7
vehicle_group      19.4
area               10.8
conviction_points   5.9
```

The exact values vary, but NCD years and driver age are usually dominant for UK motor frequency.

**What this metric measures:** CatBoost's default importance is PredictionValuesChange - the average change in the prediction when a feature's value is varied across the training data, normalised to sum to 100. It is a portfolio-level summary. It tells you which features the model is using overall, but not how any individual feature affects a specific prediction or what direction the effect is. Module 4 covers SHAP values for that purpose.

For the pricing committee presentation, feature importances answer "what is the model using?" The double lift chart (Part 14) answers "is it finding real risk differentiation?" You need both.