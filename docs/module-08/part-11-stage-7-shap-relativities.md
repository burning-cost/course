## Part 11: Stage 7 -- SHAP relativities

SHAP relativities translate the GBM's internal structure into a factor table that the pricing committee and the rating system can understand. We covered this in detail in Module 4. Here we implement it in the pipeline context.

Add a markdown cell:

```python
%md
## Stage 7: SHAP relativities from the frequency model
```

```python
import shap

# Compute SHAP values for the test set
# The SHAP values are computed in the log-space of the model's output
# (because the model uses a log link). Exponentiating gives relativities.
explainer  = shap.TreeExplainer(freq_model)
shap_vals  = explainer.shap_values(X_test)

# SHAP value for feature j on observation i: the contribution of feature j
# to log(predicted_count) relative to the model's baseline prediction.
# exp(shap_val) gives the multiplicative relativity.

# Mean absolute SHAP by feature: measures overall feature importance
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feature_importance = pl.DataFrame({
    "feature":         FEATURE_COLS,
    "mean_abs_shap":   mean_abs_shap.tolist(),
    "mean_relativity": [float(np.exp(np.abs(shap_vals[:, i])).mean())
                        for i in range(len(FEATURE_COLS))],
}).sort("mean_abs_shap", descending=True)

print("Feature importance (mean absolute SHAP values):")
print(feature_importance)
```

### Building factor relativity tables

For each feature, we compute the mean SHAP relativity at each level. This is the input that the rate optimiser uses to set the factor structure. Note that these relativities are derived directly from the GBM -- they reflect the actual model predictions, not a post-hoc regression.

```python
# For each categorical feature: mean SHAP relativity per category
# For continuous features: binned by decile

shap_relativities = {}

for j, feat in enumerate(FEATURE_COLS):
    shap_col = shap_vals[:, j]
    feat_vals = X_test[feat].values if hasattr(X_test, 'values') else np.array(X_test[feat])

    if feat in CAT_FEATURES:
        # Categorical: average SHAP per category, exponentiate for relativity
        categories = np.unique(feat_vals)
        rel_table  = {}
        for cat in categories:
            mask = feat_vals == cat
            rel_table[cat] = float(np.exp(shap_col[mask].mean()))
        # Normalise so mean relativity = 1.0
        mean_rel = np.mean(list(rel_table.values()))
        rel_table = {k: v / mean_rel for k, v in rel_table.items()}
        shap_relativities[feat] = rel_table
        print(f"\n{feat} relativities (normalised to mean=1):")
        for cat, rel in sorted(rel_table.items()):
            bar = "#" * int(rel * 10)
            print(f"  {cat:<15} {rel:.3f}  {bar}")
    else:
        # Continuous: bin into 5 groups by value and report mean relativity
        decile_labels = pd.qcut(feat_vals, q=5, duplicates="drop")
        rel_by_bin = {}
        for lbl in decile_labels.unique():
            mask = decile_labels == lbl
            rel_by_bin[str(lbl)] = float(np.exp(shap_col[mask].mean()))
        # Normalise
        mean_rel = np.mean(list(rel_by_bin.values()))
        rel_by_bin = {k: v / mean_rel for k, v in rel_by_bin.items()}
        shap_relativities[feat] = rel_by_bin
        print(f"\n{feat} relativities by quintile (normalised to mean=1):")
        for bin_lbl, rel in rel_by_bin.items():
            bar = "#" * int(rel * 10)
            print(f"  {bin_lbl:<25} {rel:.3f}  {bar}")

# Log SHAP relativities to MLflow as a JSON artefact
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_dict(shap_relativities, "shap_relativities.json")
    mlflow.log_dict({"features": feature_importance.to_dict(as_series=False)},
                    "feature_importance.json")

print("\nSHAP relativities logged to MLflow run:", freq_run_id)
```