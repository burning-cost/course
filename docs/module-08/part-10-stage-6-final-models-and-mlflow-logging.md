## Part 10: Stage 6 — Final models and MLflow logging

The final models train on all data except the held-out test year. The test year is the most recent accident year — 2025 in our synthetic data. It is entirely separate from the CV folds: it was not used for tuning and it was not used to select Optuna parameters. Its deviance is the honest out-of-sample estimate of the model's future performance.

Add a markdown cell:

```python
%md
## Stage 6: Final models — frequency and severity
```

### Frequency model

```python
import mlflow
import mlflow.catboost

max_year = int(features_pd["accident_year"].max())
df_train = features_pd[features_pd["accident_year"] < max_year].copy()
df_test  = features_pd[features_pd["accident_year"] == max_year].copy()

print(f"Training years: {sorted(df_train['accident_year'].unique().tolist())}")
print(f"Test year:      {max_year}")
print(f"Training rows:  {len(df_train):,}")
print(f"Test rows:      {len(df_test):,}")
print(f"Training claims:{df_train['claim_count'].sum():,}")
print(f"Test claims:    {df_test['claim_count'].sum():,}")

X_train = df_train[FEATURE_COLS]
y_train = df_train["claim_count"].values
w_train = df_train["exposure"].values

X_test  = df_test[FEATURE_COLS]
y_test  = df_test["claim_count"].values
w_test  = df_test["exposure"].values

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
    mlflow.log_params(best_freq_params)
    mlflow.log_params({
        "model_type":         "catboost_poisson",
        "raw_table":          TABLES["raw"],
        "raw_table_version":  int(raw_version),
        "feat_table":         TABLES["features"],
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "offset":             "log_exposure",
        "train_years":        str(sorted(df_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
    })

    freq_model = CatBoostRegressor(**best_freq_params)
    freq_model.fit(train_pool, eval_set=test_pool)

    # Evaluate on test set
    freq_pred_test = freq_model.predict(test_pool)
    test_dev = poisson_deviance(y_test, freq_pred_test, w_test)

    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      mean_cv_deviance)
    mlflow.log_metric("generalisation_gap",    test_dev - mean_cv_deviance)
    mlflow.log_metric("n_training_rows",       len(df_train))
    mlflow.log_metric("n_test_rows",           len(df_test))

    mlflow.catboost.log_model(freq_model, "freq_model")

    # Log the feature spec alongside the model
    mlflow.log_artifact("/tmp/feature_spec.json", artifact_path="feature_spec")

    freq_run_id = freq_run.info.run_id

gen_gap = test_dev - mean_cv_deviance
print(f"Frequency model logged: {freq_run_id}")
print(f"Test Poisson deviance:  {test_dev:.5f}")
print(f"Mean CV deviance:       {mean_cv_deviance:.5f}")
print(f"Generalisation gap:     {gen_gap:.5f}  "
      f"({'OK' if abs(gen_gap) < 0.015 else 'REVIEW: large gap'})")
```

**The generalisation gap.** A gap of near zero means the tuned model performs as well on the test year as it did on the CV validation years. A positive gap (test deviance higher than CV deviance) is normal — the test year is a genuinely different period. A gap larger than 0.015 warrants investigation: either the model is overfitting to the validation periods, or the test year's distribution is materially different from training. Check the accident year frequency trends from Stage 2 first.

### Frequency predictions and pure premium

```python
# Frequency rate = predicted count / exposure
freq_rate_all = freq_pred_test / np.clip(w_test, 1e-6, None)
```

### Severity model

The severity model requires a three-way temporal split for valid conformal calibration (Part 12). We train the severity model on 2022-2023, calibrate conformal on 2024, and test on 2025. This ensures the calibration residuals are genuinely out-of-sample.

```python
# Severity model trains on claims-only data, excluding the calibration year
# so that Stage 8 can use the calibration year as a true holdout.
cal_year    = sorted(df_train["accident_year"].unique())[-1]   # 2024
df_sev_model = df_train[
    (df_train["accident_year"] < cal_year) & (df_train["claim_count"] > 0)
].copy()
df_sev_test = df_test[df_test["claim_count"] > 0].copy()

df_sev_model["mean_sev"] = (df_sev_model["incurred_loss"] / df_sev_model["claim_count"])
df_sev_test["mean_sev"]  = (df_sev_test["incurred_loss"]  / df_sev_test["claim_count"])

print(f"Severity training years: {sorted(df_sev_model['accident_year'].unique().tolist())}")
print(f"Conformal calibration year: {cal_year}")
print(f"Severity training n:   {len(df_sev_model):,} claims-only")
print(f"Severity test n:       {len(df_sev_test):,} claims-only")

sev_train_pool = Pool(
    df_sev_model[FEATURE_COLS],
    df_sev_model["mean_sev"].values,
    weight=df_sev_model["claim_count"].values,
    cat_features=CAT_FEATURES,
)
sev_test_pool = Pool(
    df_sev_test[FEATURE_COLS],
    df_sev_test["mean_sev"].values,
    weight=df_sev_test["claim_count"].values,
    cat_features=CAT_FEATURES,
)

with mlflow.start_run(run_name="sev_model_m08") as sev_run:
    mlflow.log_params(best_sev_params)
    mlflow.log_params({
        "model_type":       "catboost_gamma",
        "severity_target":  "mean_cost_per_claim",
        "severity_weight":  "claim_count",
        "train_years":      str(sorted(df_sev_model["accident_year"].unique().tolist())),
        "cal_year":         str(cal_year),
        "test_year":        str(max_year),
        "run_date":         RUN_DATE,
    })

    sev_model = CatBoostRegressor(**best_sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    sev_pred_test = sev_model.predict(sev_test_pool)
    sev_rmse = float(np.sqrt(np.mean((df_sev_test["mean_sev"].values - sev_pred_test) ** 2)))

    mlflow.log_metric("test_severity_rmse",  sev_rmse)
    mlflow.log_metric("n_sev_training",      len(df_sev_model))
    mlflow.log_metric("n_sev_test",          len(df_sev_test))
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = sev_run.info.run_id

print(f"Severity model logged: {sev_run_id}")
print(f"Test severity RMSE:    £{sev_rmse:,.0f}")
```

### Pure premium for all test policies

The severity model predicts expected cost given a claim occurs. This is a valid prediction for any policy — including zero-claim policies — because we are asking "if this risk has a claim, how much would it cost?" The pure premium is frequency times severity for every policy in the test set.

```python
# Predict severity for ALL test policies (claims and non-claims)
all_test_pool_sev = Pool(X_test, cat_features=CAT_FEATURES)
sev_pred_all = sev_model.predict(all_test_pool_sev)

pure_premium = freq_rate_all * sev_pred_all

assert (sev_pred_all > 0).all(),  "Severity model produced non-positive predictions"
assert (pure_premium  > 0).all(), "Pure premium contains zeros or negatives"
assert np.isfinite(pure_premium).all(), "Pure premium contains non-finite values"

print(f"Pure premium — test set (n={len(pure_premium):,}):")
print(f"  Mean:   £{pure_premium.mean():,.2f}")
print(f"  Median: £{np.median(pure_premium):,.2f}")
print(f"  P10:    £{np.percentile(pure_premium, 10):,.2f}")
print(f"  P90:    £{np.percentile(pure_premium, 90):,.2f}")
print(f"  P99:    £{np.percentile(pure_premium, 99):,.2f}")

# Write to Delta
pred_df = pl.DataFrame({
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

(
    spark.createDataFrame(pred_df.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["freq_predictions"])
)

spark.sql(f"""
    ALTER TABLE {TABLES['freq_predictions']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

print(f"\nPredictions written to {TABLES['freq_predictions']}")
```
