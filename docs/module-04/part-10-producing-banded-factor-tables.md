## Part 10: Producing banded factor tables

Continuous feature curves are good for diagnostics, but factor tables require discrete bands. Actuarial convention and rating system constraints both demand breakpoints.

The key principle: **band the SHAP values, do not band the feature before modelling**. The model was trained on continuous `driver_age`. You cannot pass `age_band` to a model that was trained on `driver_age`. What you can do is extract the SHAP values for `driver_age` (which the model already computed for each observation) and then aggregate those SHAP values by your chosen age bands.

In a new cell, type this and run it (Shift+Enter):

```python
# Define age bands - round numbers that are defensible to the committee
age_breaks = [17, 22, 25, 30, 40, 55, 70, 86]
age_labels  = ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"]

# Add age_band to the Polars DataFrame
df_banded = df.with_columns(
    pl.col("driver_age").cut(
        breaks=age_breaks[1:-1],
        labels=age_labels,
    ).alias("age_band")
)

print("Age band distribution:")
print(
    df_banded.group_by("age_band")
    .agg(
        pl.len().alias("n_obs"),
        pl.col("exposure").sum().alias("total_exposure"),
        pl.col("claim_count").sum().alias("claims"),
    )
    .with_columns((pl.col("claims") / pl.col("total_exposure")).alias("observed_freq"))
    .sort("age_band")
)
```

You will see a table showing the count of policies, exposure, and observed frequency for each age band. Verify that no band has fewer than 500 policies - very sparse bands will have unreliable relativities.

Now extract the per-observation SHAP values and aggregate by age band. In a new cell, type this and run it (Shift+Enter):

```python
shap_vals    = sr.shap_values()            # numpy array, shape (100_000, n_features)
feature_names = sr.feature_names_          # list matching the SHAP columns

age_idx = feature_names.index("driver_age")
age_shap = shap_vals[:, age_idx]

# Build a Polars frame: age_band, age SHAP value, exposure
shap_frame = pl.DataFrame({
    "age_band": df_banded["age_band"].to_list(),
    "age_shap": age_shap.tolist(),
    "exposure":  df["exposure"].to_list(),
})

# Exposure-weighted mean SHAP per band
band_stats = shap_frame.group_by("age_band").agg([
    (pl.col("age_shap") * pl.col("exposure")).sum().alias("weighted_shap_sum"),
    pl.col("exposure").sum().alias("total_exposure"),
    pl.col("exposure").count().alias("n_obs"),
    pl.col("age_shap").std().alias("shap_std"),
]).with_columns(
    (pl.col("weighted_shap_sum") / pl.col("total_exposure")).alias("mean_shap")
)

# Base level: 30-39 (lowest risk mid-range band)
base_shap = band_stats.filter(pl.col("age_band") == "30-39")["mean_shap"][0]

band_rels = band_stats.with_columns(
    (pl.col("mean_shap") - base_shap).exp().alias("relativity")
).sort("age_band")

print("Age band relativities (base: 30-39):")
print(band_rels.select(["age_band", "relativity", "n_obs", "total_exposure"]).sort("age_band"))
```

You will see:

```sql
Age band relativities (base: 30-39):
shape: (7, 4)
┌─────────┬────────────┬───────┬────────────────┐
│ age_band│ relativity │ n_obs │ total_exposure │
╞═════════╪════════════╪═══════╪════════════════╡
│ 17-21   │ 1.823      │  4987 │       3905.1   │
│ 22-24   │ 1.421      │  3519 │       2794.3   │
│ 25-29   │ 1.178      │  7103 │       5661.4   │
│ 30-39   │ 1.000      │ 18241 │      14538.2   │
│ 40-54   │ 0.988      │ 24803 │      19758.9   │
│ 55-69   │ 1.042      │ 21374 │      17023.1   │
│ 70+     │ 1.187      │ 19973 │      15893.6   │
└─────────┴────────────┴───────┴────────────────┘
```

The 17-21 band should show a relativity significantly above 1.0, and the 70+ band a milder uplift. The true DGP has `+0.55` for under-25 and `+0.20` for over-70, giving `exp(0.55) ≈ 1.73` and `exp(0.20) ≈ 1.22`. Your extracted relativities should be in that neighbourhood.