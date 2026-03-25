## Part 11: Stage 7 — SHAP relativities

SHAP relativities translate the GBM's internal structure into a factor table the pricing committee and the rating engine can use. The output format — one row per feature per level, with a multiplicative relativity and confidence interval — is identical to a GLM's `exp(beta)` table. The pricing committee can compare GBM and GLM relativities directly, without needing to understand how SHAP values are computed.

We use the `shap_relativities` library, which handles the normalisation and exposure weighting that raw SHAP values do not provide.

Add a markdown cell:

```python
%md
## Stage 7: SHAP relativities — frequency model
```

```python
from shap_relativities import SHAPRelativities
import polars as pl

# SHAPRelativities computes SHAP values internally via the model's TreeExplainer.
# We pass the test set features and exposure.
# exposure is used to weight each observation's contribution to the mean
# relativity per level — so a policy with 0.3 years exposure influences
# the regional relativity less than a policy with a full year.

X_test_pl = pl.from_pandas(df_test[FEATURE_COLS].reset_index(drop=True))
exposure_test = pl.Series("exposure", w_test.tolist())

sr = SHAPRelativities(
    model=freq_model,
    X=X_test_pl,
    exposure=exposure_test,
    categorical_features=CAT_FEATURES,   # ["region"]
)
sr.fit()

# extract_relativities returns a Polars DataFrame:
# feature | level | relativity | lower_ci | upper_ci | mean_shap | shap_std | n_obs | exposure_weight
#
# normalise_to="mean": portfolio mean relativity = 1.0 for each feature.
# This is directly comparable to a GLM factor table normalised to the mean.
#
# base_levels: define the base level for each factor, which gets relativity = 1.0.
# If base_levels is None, normalise_to="mean" applies.
freq_relativities = sr.extract_relativities(
    normalise_to="mean",
    ci_method="clt",    # Central Limit Theorem CIs — appropriate for n_obs > 30 per level
)

print("Frequency model relativities:")
print(freq_relativities)
```

**What you should see:** A DataFrame with one row per level per feature. For `region`, expect six rows (London, SouthEast, Midlands, North, Scotland, Wales). For `ncb_deficit`, expect six rows (0 through 5). The `relativity` column holds the multiplicative factor — a `relativity` of 1.20 for London means London policies are expected to have 20% more claims than the portfolio average, after controlling for all other factors.

The `lower_ci` and `upper_ci` columns are 95% confidence intervals from the CLT approximation. Wide intervals for a level indicate limited data — the CI for Scotland (10% of the portfolio) will be roughly 2x as wide as the CI for Midlands (22%).

### Writing relativities to Delta

```python
(
    spark.createDataFrame(freq_relativities.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["freq_relativities"])
)

spark.sql(f"""
    ALTER TABLE {TABLES['freq_relativities']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

print(f"Relativities written to {TABLES['freq_relativities']}")
```

### Logging to MLflow

```python
# Log as a dict artefact alongside the frequency model
rel_dict = {
    row["feature"] + "_" + str(row["level"]): float(row["relativity"])
    for row in freq_relativities.iter_rows(named=True)
}

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_dict(rel_dict, "shap_relativities.json")

print(f"Relativities logged to MLflow run {freq_run_id}")
```

### Reading the factor table

For the pricing committee pack, pivot the relativity table into factor format:

```python
# One column per feature — the standard factor table format
for feature in freq_relativities["feature"].unique().to_list():
    feature_rels = (
        freq_relativities
        .filter(pl.col("feature") == feature)
        .select(["level", "relativity", "lower_ci", "upper_ci", "n_obs"])
        .sort("relativity", descending=True)
    )
    print(f"\n{feature} (sorted by relativity):")
    print(feature_rels.with_columns(
        pl.col("relativity").round(3),
        pl.col("lower_ci").round(3),
        pl.col("upper_ci").round(3),
    ))
```

**Comparing to GLM relativities.** If you have run a GLM for this line previously, print the GLM relativities alongside the SHAP relativities. Divergences in rank ordering — not just magnitude — are diagnostically informative. If SHAP says London is at 1.25x and the GLM says 1.40x, the difference is likely from interactions (younger drivers in London, higher vehicle groups) that the GLM cannot express without manual interaction terms. These divergences inform the feature engineering decisions in the next model iteration.

**Using relativities in the rate optimiser.** In Stage 9, the rate optimiser expects factor relativities as input. The SHAP relativity table for `region` is directly usable as the factor relativity input — it is already in the expected format (level → multiplicative relativity). This is one of the key connections in the pipeline: the rate optimiser works from the model's own output, not from a separate GLM run.
