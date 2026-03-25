## Part 5: Characteristic Stability Index (CSI)

### What CSI measures

CSI is PSI applied to individual features rather than the score distribution. You run it on each input feature to find out which ones have shifted. A model with 15 features might have PSI flagging a significant score shift; CSI tells you that 13 features are stable and 2 features — say, driver age and region — have drifted materially. That narrows the investigation considerably.

The formula is identical to PSI. The difference is what you are computing it on: instead of the predicted score, you compute it on the raw feature values.

For continuous features, the bins are defined by the reference distribution (deciles by default). The current distribution is then mapped onto those same bins. This is important — you do not recompute bins from the current distribution, because that would remove the ability to detect a shift in the overall level.

For categorical features, each category is its own "bin." A category that existed in reference but has disappeared in current, or a new category that appears in current but did not exist in reference, will show up as a large contribution to CSI.

### Running CSI on all features

The `csi()` function accepts two Polars DataFrames and a list of feature names. It returns a Polars DataFrame with columns `feature`, `csi`, and `band`:

```python
from insurance_monitoring.drift import csi

csi_df = csi(
    reference_df=df_reference,
    current_df=df_current,
    features=FEATURE_NAMES,
    n_bins=N_BINS,
)

# csi_df has columns: feature (str), csi (f64), band (str)
print(f"{'Feature':<30} {'CSI':>8}  {'Status'}")
print("-" * 50)
for row in csi_df.sort("csi", descending=True).iter_rows(named=True):
    flag = " <-- investigate" if row["csi"] > 0.25 else ""
    print(f"{row['feature']:<30} {row['csi']:>8.4f}  {row['band']}{flag}")
```

The `band` column uses the PSI thresholds: `green` (< 0.10), `amber` (0.10–0.25), `red` (> 0.25).

### Handling categorical features

The motor dataset has categorical features like `region`. `csi()` handles these through the same PSI formula: each distinct value is treated as a bin. The calculation still works, but inspect the raw category distribution to see if a new category has appeared:

```python
# Check for new categories in categorical features
cat_features = ["region"]

for feature in cat_features:
    ref_cats = set(df_reference[feature].unique().to_list())
    cur_cats = set(df_current[feature].unique().to_list())

    new_cats     = cur_cats - ref_cats
    missing_cats = ref_cats - cur_cats

    if new_cats:
        print(f"{feature}: NEW categories in current: {new_cats}")
    if missing_cats:
        print(f"{feature}: MISSING from current (ref only): {missing_cats}")

print("Category check complete.")
```

A new category appearing in the current data is a hard signal worth investigating. It might be a new distribution channel, a new underwriting segment, or a data quality issue upstream.

### Visualising feature drift

For the features with CSI above 0.10, plot the distribution comparison:

```python
notable_features = csi_df.filter(pl.col("csi") > 0.10)["feature"].to_list()

if not notable_features:
    print("No features with CSI > 0.10. Book mix is stable.")
else:
    n_feat = len(notable_features)
    fig, axes = plt.subplots(1, n_feat, figsize=(6 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]

    for ax, feature in zip(axes, notable_features):
        ref_vals = df_reference[feature].to_numpy()
        cur_vals = df_current[feature].to_numpy()
        csi_val  = float(csi_df.filter(pl.col("feature") == feature)["csi"][0])

        ax.hist(ref_vals, bins=30, alpha=0.6, label="Reference", color="steelblue", density=True)
        ax.hist(cur_vals, bins=30, alpha=0.6, label="Current",   color="tomato",    density=True)
        ax.set_title(f"{feature}  (CSI={csi_val:.3f})")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.show()
```

### What the CSI tells you about the model's risk

CSI on features is diagnostic, not prescriptive. It tells you what has changed; it does not tell you what to do. The action depends on the A/E ratio and Gini drift results.

If a feature has drifted but the A/E ratio is fine, the model is re-applying risk loads to a different mix of risks — which is fine, that is exactly what it should do. The model handles the mix shift without recalibration.

If a feature has drifted and the A/E ratio is elevated in the direction you would expect from that feature shift (e.g., the young driver proportion has grown and the A/E is above 1.0), the model is not adequately capturing the young driver risk — possibly because young drivers were a small segment in training and the learned relationship is weak. This might require retraining with more weight on that segment, or adding a manual adjustment.
