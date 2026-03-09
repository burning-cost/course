## Part 10: Stage 6 -- Final models and MLflow logging

The final models train on all data except the held-out test year. The test year is the most recent accident year in the dataset -- in this case, 2024. It is held out entirely from training and used only for final evaluation. This is separate from the CV folds, which used 2023 and 2024 for validation in different folds.

Add a markdown cell:

```python
%md
## Stage 6: Final models -- frequency (Poisson) and severity (Gamma/Tweedie)
```

### Splitting train and test

```python
import json
import mlflow
import mlflow.catboost

max_year = int(features_pd["accident_year"].max())
print(f"Test year (held out from training): {max_year}")

# Train set: all years except the test year
df_train = features_pd[features_pd["accident_year"] < max_year].copy()
df_test  = features_pd[features_pd["accident_year"] == max_year].copy()

print(f"Training rows: {len(df_train):,}  ({df_train['accident_year'].unique()} years)")
print(f"Test rows:     {len(df_test):,}   (year {max_year})")
print(f"Training claims: {df_train['claim_count'].sum():,}")
print(f"Test claims:     {df_test['claim_count'].sum():,}")
```

### Frequency model

```python
X_train = df_train[FEATURE_COLS]
y_train = df_train["claim_count"].values
w_train = df_train["exposure"].values

X_test  = df_test[FEATURE_COLS]
y_test  = df_test["claim_count"].values
w_test  = df_test["exposure"].values

# Build Pools with the log-exposure offset (CRITICAL: baseline, not weight)
train_pool = Pool(
    X_train, y_train,
    baseline=np.log(np.clip(w_train, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
test_pool = Pool(
    X_test, y_test,
    baseline=np.log(np.clip(w_test, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

with mlflow.start_run(run_name="freq_model_m08") as freq_run:
    # Log all parameters that determined this model
    mlflow.log_params(best_freq_params)
    mlflow.log_params({
        "model_type":         "catboost_poisson",
        "raw_table":          TABLES["raw"],
        "raw_table_version":  int(raw_version),
        "feat_table":         TABLES["features"],
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "train_years":        str(sorted(df_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
        "offset":             "log_exposure",   # explicit: this model uses the correct offset
    })

    # Fit the final frequency model
    freq_model = CatBoostRegressor(**best_freq_params)
    freq_model.fit(train_pool, eval_set=test_pool)

    # Evaluate on test set
    freq_pred_test = freq_model.predict(test_pool)
    test_dev = poisson_deviance(y_test, freq_pred_test, w_test)

    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      mean_cv_deviance)
    mlflow.log_metric("n_training_rows",       len(df_train))
    mlflow.log_metric("n_test_rows",           len(df_test))

    # Log the model artefact to MLflow Model Registry
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = freq_run.info.run_id

print(f"Frequency model trained and logged.")
print(f"MLflow run ID: {freq_run_id}")
print(f"Test Poisson deviance: {test_dev:.5f}")
print(f"Mean CV deviance:      {mean_cv_deviance:.5f}")
print(f"Generalisation gap:    {test_dev - mean_cv_deviance:.5f} "
      f"({'within tolerance' if abs(test_dev - mean_cv_deviance) < 0.01 else 'REVIEW: may be overfitting'})")
```

**What is the generalisation gap?** The gap between the mean CV deviance and the test deviance tells you whether the model generalises from its validation period to the final test year. A small positive gap (test deviance slightly higher than CV deviance) is normal and expected -- the test year is a different period. A large gap (more than 0.01 for this dataset) suggests the model is tuned too aggressively to the validation period, or that the test year's distribution is materially different from training.

### Writing frequency predictions to Delta

```python
# Predict frequency for the full test set
freq_pred_all = freq_model.predict(test_pool)

# The model predicts claim counts (exposure * frequency).
# Divide by exposure to get frequency per policy-year.
freq_rate_all = freq_pred_all / np.clip(w_test, 1e-6, None)

freq_pred_df = pl.DataFrame({
    "policy_id":          df_test["policy_id"].tolist(),
    "accident_year":      df_test["accident_year"].tolist(),
    "exposure":           w_test.tolist(),
    "claim_count_actual": y_test.tolist(),
    "freq_pred_count":    freq_pred_all.tolist(),
    "freq_pred_rate":     freq_rate_all.tolist(),
    "mlflow_run_id":      [freq_run_id] * len(df_test),
    "run_date":           [RUN_DATE] * len(df_test),
})

spark.createDataFrame(freq_pred_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["freq_predictions"])

freq_pred_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['freq_predictions']} LIMIT 1"
).collect()[0]["version"]

print(f"Frequency predictions written: {len(freq_pred_df):,} rows")
print(f"Table: {TABLES['freq_predictions']} (version {freq_pred_version})")
```

### Severity model

```python
# -----------------------------------------------------------------------
# Severity model: Tweedie with variance_power=2 (Gamma distribution)
#
# Trains on policies with at least one claim only.
# Target: mean severity per claim (claim_amount / claim_count).
# Weight: claim count (more claims = more evidence about this risk's severity).
# No exposure offset -- we are modelling severity given a claim, not claim counts.
# -----------------------------------------------------------------------

df_sev_train = df_train[df_train["claim_count"] > 0].copy()
df_sev_test  = df_test[df_test["claim_count"]  > 0].copy()

df_sev_train["mean_sev"] = df_sev_train["claim_amount"] / df_sev_train["claim_count"]
df_sev_test["mean_sev"]  = df_sev_test["claim_amount"]  / df_sev_test["claim_count"]

X_sev_train = df_sev_train[FEATURE_COLS]
y_sev_train = df_sev_train["mean_sev"].values
w_sev_train = df_sev_train["claim_count"].values

X_sev_test  = df_sev_test[FEATURE_COLS]
y_sev_test  = df_sev_test["mean_sev"].values
w_sev_test  = df_sev_test["claim_count"].values

sev_train_pool = Pool(
    X_sev_train, y_sev_train,
    weight=w_sev_train,
    cat_features=CAT_FEATURES,
)
sev_test_pool = Pool(
    X_sev_test, y_sev_test,
    weight=w_sev_test,
    cat_features=CAT_FEATURES,
)

print(f"Severity training: {len(df_sev_train):,} policies with claims")
print(f"Severity test:     {len(df_sev_test):,} policies with claims")

with mlflow.start_run(run_name="sev_model_m08") as sev_run:
    mlflow.log_params(best_sev_params)
    mlflow.log_params({
        "model_type":         "catboost_gamma",
        "raw_table":          TABLES["raw"],
        "raw_table_version":  int(raw_version),
        "feat_table":         TABLES["features"],
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "train_years":        str(sorted(df_sev_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
        "severity_target":    "mean_sev_per_claim",
        "severity_weight":    "claim_count",
    })

    sev_model = CatBoostRegressor(**best_sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    sev_pred_test = sev_model.predict(sev_test_pool)
    sev_rmse = float(np.sqrt(np.mean((y_sev_test - sev_pred_test)**2)))

    mlflow.log_metric("test_severity_rmse",  sev_rmse)
    mlflow.log_metric("n_sev_training_rows", len(df_sev_train))
    mlflow.log_metric("n_sev_test_rows",     len(df_sev_test))
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = sev_run.info.run_id

print(f"Severity model trained and logged.")
print(f"MLflow run ID: {sev_run_id}")
print(f"Test severity RMSE: £{sev_rmse:,.0f}")
```

### Computing pure premiums for the full test set

The pure premium is frequency times severity. The critical step here is predicting severity for ALL test policies, not just those with claims. The severity model predicts expected severity given a claim occurs -- it is a valid prediction for any policy, regardless of whether it had a claim in the test year.

The earlier versions of this pipeline used `fillna(sev_pred.mean())` to fill in severity for zero-claim policies, where `sev_pred.mean()` was the mean over the claims-only prediction set. This is wrong: the mean severity for policies that had claims is biased upward relative to the population. Zero-claim policies tend to be lower-risk, so the correct imputed severity is lower than the claims-only mean.

The correct approach is to predict severity for all policies:

```python
# Predict severity for ALL test policies (not just those with claims)
# The severity model predicts: expected severity given a claim occurred
# This is a meaningful prediction for any policy

X_all_test = df_test[FEATURE_COLS]

# Pool for full test set (no labels needed for prediction)
all_test_pool_sev = Pool(X_all_test, cat_features=CAT_FEATURES)
sev_pred_all = sev_model.predict(all_test_pool_sev)

# Pure premium = predicted frequency rate * predicted severity
# This is the expected loss cost per policy-year for each risk
pure_premium = freq_rate_all * sev_pred_all

# Sanity checks
assert (sev_pred_all > 0).all(),  "Negative severity predictions (model error)"
assert (pure_premium  > 0).all(), "Negative pure premiums (frequency or severity error)"
assert np.isfinite(pure_premium).all(), "Non-finite pure premiums"

print(f"Pure premium statistics (test set):")
print(f"  Mean:   £{pure_premium.mean():,.2f}")
print(f"  Median: £{np.median(pure_premium):,.2f}")
print(f"  P95:    £{np.percentile(pure_premium, 95):,.2f}")
print(f"  Min:    £{pure_premium.min():,.2f}")
print(f"  Max:    £{pure_premium.max():,.2f}")

# Append to frequency predictions table
pred_full_df = pl.DataFrame({
    "policy_id":          df_test["policy_id"].tolist(),
    "accident_year":      df_test["accident_year"].tolist(),
    "exposure":           w_test.tolist(),
    "claim_count_actual": y_test.tolist(),
    "freq_pred_rate":     freq_rate_all.tolist(),
    "sev_pred":           sev_pred_all.tolist(),
    "pure_premium":       pure_premium.tolist(),
    "freq_run_id":        [freq_run_id] * len(df_test),
    "sev_run_id":         [sev_run_id]  * len(df_test),
    "run_date":           [RUN_DATE]    * len(df_test),
})

spark.createDataFrame(pred_full_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["freq_predictions"])

print(f"\nFull predictions with pure premium written to {TABLES['freq_predictions']}")
```