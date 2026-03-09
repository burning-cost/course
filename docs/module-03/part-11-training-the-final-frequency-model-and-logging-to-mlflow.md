## Part 11: Training the final frequency model and logging to MLflow

After tuning, we train the final model on all available data except the held-out test year. We use 2024 as the test year - it is the most recent accident year, making this the closest approximation to "predicting next year from this year's data."

### Build the final model parameters

Create a new cell:

```python
# Add the fixed parameters to the best tuned parameters
freq_params = {
    **best_params,
    "loss_function": "Poisson",
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,  # print progress every 100 iterations when training final model
}

max_year = df["accident_year"].max()
print(f"Test year (held out): {max_year}")
print(f"Training on accident years: {sorted(df.filter(pl.col('accident_year') < max_year)['accident_year'].unique().to_list())}")
```

### Prepare the final train and test sets

```python
df_train_final = df.filter(pl.col("accident_year") < max_year)
df_test_final  = df.filter(pl.col("accident_year") == max_year)

X_train_f = df_train_final[FEATURES].to_pandas()
y_train_f = df_train_final[FREQ_TARGET].to_numpy()
w_train_f = df_train_final[EXPOSURE_COL].to_numpy()

X_test_f = df_test_final[FEATURES].to_pandas()
y_test_f = df_test_final[FREQ_TARGET].to_numpy()
w_test_f = df_test_final[EXPOSURE_COL].to_numpy()

final_train_pool = Pool(X_train_f, y_train_f, baseline=np.log(w_train_f), cat_features=CAT_FEATURES)
final_test_pool  = Pool(X_test_f,  y_test_f,  baseline=np.log(w_test_f),  cat_features=CAT_FEATURES)
```

### Train and log to MLflow

MLflow runs are created with a context manager (`with mlflow.start_run(...)`). Everything inside the `with` block is recorded as part of this run. Create a new cell:

```python
with mlflow.start_run(run_name="freq_catboost_tuned") as run_freq:
    # Log what we did
    mlflow.log_params(freq_params)
    mlflow.log_param("cv_strategy",  "walk_forward_ibnr1")
    mlflow.log_param("n_cv_folds",   len(folds))
    mlflow.log_param("features",     json.dumps(FEATURES))
    mlflow.log_param("cat_features", json.dumps(CAT_FEATURES))
    mlflow.log_param("train_years",  str(sorted(df_train_final["accident_year"].unique().to_list())))
    mlflow.log_param("test_year",    str(max_year))
    mlflow.log_param("run_date",     str(date.today()))

    # Train
    freq_model = CatBoostRegressor(**freq_params)
    freq_model.fit(final_train_pool, eval_set=final_test_pool)

    # Evaluate on the test year
    y_pred_freq = freq_model.predict(final_test_pool)
    test_dev    = poisson_deviance(y_test_f, y_pred_freq, w_test_f)

    # Log metrics
    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      np.mean(cv_deviances))
    mlflow.log_metric("cv_deviance_std",       np.std(cv_deviances))

    # Log the model artefact
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = run_freq.info.run_id

print(f"\nTest year Poisson deviance: {test_dev:.4f}")
print(f"MLflow run ID: {freq_run_id}")
```

Run this cell. The model trains on 2019-2023 data (80,000 policies) and the training progress prints every 100 iterations. At the end you see the test deviance.

**What MLflow is recording:** The `log_params` call records all the hyperparameters. The `log_metric` calls record the evaluation results. The `log_model` call saves the entire CatBoost model object as an MLflow artefact. If someone needs to reproduce this model result in 18 months, they open the MLflow experiment, find this run by its ID, and load the model artefact. The training year range tells them which Delta table version to use for the data. That is the audit trail.

To view the MLflow run in the UI: in the Databricks left sidebar, click **Experiments**. Find the experiment for this notebook. Click on the run named "freq_catboost_tuned". You should see all the parameters and metrics logged.