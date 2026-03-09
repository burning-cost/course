## Part 13: The severity model

The severity model predicts average cost per claim for policies that had at least one claim. Two design decisions need explaining before we fit it.

**No exposure offset.** The cost of a claim does not depend on how long the policy was in force. A repair costing £3,500 costs £3,500 whether the policy had been running for 3 months or 12 months at the time of the accident. Exposure enters the frequency model because frequency is a rate (claims per year). Severity is a conditional cost - we condition on having a claim, and that conditioning removes the exposure effect.

**Tweedie variance_power=2 as the Gamma equivalent.** CatBoost does not offer a `Gamma` loss function by name. The Tweedie distribution with variance power p=2 is mathematically the Gamma distribution with log link. Power=1 is Poisson. Power between 1 and 2 is compound Poisson-Gamma. Power=2 is Gamma. The relationship is exact.

### Prepare the severity data

Create a new cell:

```python
# Severity model: claims-only subset
# Average severity = total incurred / claim count
df_train_sev = df_train_final.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)
df_test_sev = df_test_final.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

print(f"Training claims: {len(df_train_sev):,} ({100*len(df_train_sev)/len(df_train_final):.1f}% of training policies)")
print(f"Test claims:     {len(df_test_sev):,} ({100*len(df_test_sev)/len(df_test_final):.1f}% of test policies)")
print(f"\nMean severity (training): £{df_train_sev['avg_severity'].mean():,.0f}")
print(f"Mean severity (test):     £{df_test_sev['avg_severity'].mean():,.0f}")
print(f"95th percentile (test):   £{df_test_sev['avg_severity'].quantile(0.95):,.0f}")
```

### Fit the severity model

```python
X_train_s = df_train_sev[FEATURES].to_pandas()
y_train_s = df_train_sev["avg_severity"].to_numpy()

X_test_s  = df_test_sev[FEATURES].to_pandas()
y_test_s  = df_test_sev["avg_severity"].to_numpy()

sev_params = {
    **best_params,
    "loss_function": "Tweedie:variance_power=2",   # Gamma equivalent
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       100,
}

# NOTE: No baseline parameter - severity has no exposure offset.
sev_train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

sev_model = CatBoostRegressor(**sev_params)
sev_model.fit(sev_train_pool, eval_set=sev_test_pool)
```

**A practical note on shared hyperparameters:** Here we are using the same tuned hyperparameters for the severity model as for the frequency model. This is a tutorial simplification. In production, tune severity hyperparameters separately. The optimal depth for a Poisson frequency model on 100,000 policies is not necessarily optimal for a Gamma severity model on the 7-10% of policies that had claims. The severity model is fitting a smaller, noisier dataset with a different response distribution.

### Evaluate and log the severity model

```python
y_pred_sev = sev_model.predict(sev_test_pool)
sev_rmse   = float(np.sqrt(np.mean((y_test_s - y_pred_sev) ** 2)))
sev_mae    = float(np.mean(np.abs(y_test_s - y_pred_sev)))
mean_bias  = float(np.mean(y_pred_sev) / np.mean(y_test_s) - 1)

print(f"Severity RMSE:       £{sev_rmse:,.0f}")
print(f"Severity MAE:        £{sev_mae:,.0f}")
print(f"Mean severity bias:  {mean_bias*100:+.1f}%")
```

We evaluate severity on RMSE rather than Poisson deviance. RMSE gives an error metric in the same units as the claim amounts (pounds). For the pricing committee, a severity model that is right on average is more important than one with low RMSE at the individual level - individual claim cost is inherently unpredictable, but the portfolio average must be right.

**What the mean severity bias tells you:** A bias above 5% in either direction signals a calibration problem. If the model is predicting £3,200 mean severity when the actual mean is £3,000, every pure premium computed from this model is overstated by 7%. This is not a minor error in a book rating hundreds of thousands of policies.

Now log the severity model:

```python
with mlflow.start_run(run_name="sev_catboost_tuned") as run_sev:
    mlflow.log_params({k: v for k, v in sev_params.items() if k != "verbose"})
    mlflow.log_param("train_years",  str(sorted(df_train_sev["accident_year"].unique().to_list())))
    mlflow.log_param("test_year",    str(max_year))
    mlflow.log_param("run_date",     str(date.today()))
    mlflow.log_metric("test_rmse",   sev_rmse)
    mlflow.log_metric("test_mae",    sev_mae)
    mlflow.log_metric("mean_bias",   mean_bias)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = run_sev.info.run_id

print(f"Severity model logged. Run ID: {sev_run_id}")
```