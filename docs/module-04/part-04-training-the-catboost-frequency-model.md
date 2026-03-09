## Part 4: Training the CatBoost frequency model

We train a Poisson frequency model on the full dataset. This is the same model as Module 3, trained on all 100,000 policies rather than on walk-forward folds. We already validated the model's out-of-sample performance in Module 3 - here we want the best possible relativities, which means training on as much data as possible.

In a new cell, type this and run it (Shift+Enter):

```python
# Bridge to pandas at the CatBoost boundary
df_pd = df.to_pandas()

X_pd       = df_pd[FREQ_FEATURES]
y_pd       = df_pd["claim_count"]
exposure_pd = df_pd["exposure"]

# Exposure offset: log(exposure) passed as baseline in the Pool.
# This tells CatBoost: start each prediction at log(exposure_i),
# then learn the frequency contribution net of exposure.
# Must be log(exposure), NOT raw exposure. Module 3 explains why.
log_exposure = np.log(exposure_pd.clip(lower=1e-6))

train_pool = cb.Pool(
    data=X_pd,
    label=y_pd,
    baseline=log_exposure,
    cat_features=CAT_FEATURES,
)

freq_params = {
    "loss_function":    "Poisson",
    "learning_rate":    0.05,
    "depth":            5,
    "min_data_in_leaf": 50,
    "iterations":       300,
    "random_seed":      42,
    "verbose":          0,
}

freq_model = cb.CatBoostRegressor(**freq_params)
freq_model.fit(train_pool)

print("Model trained.")
print(f"Best iteration: {freq_model.best_iteration_}")
```

The output looks like:

```bash
Model trained.
Best iteration: 299
```

If the best iteration equals the total iterations (300 here), the model had not converged yet. You could increase `iterations` to let it train longer. For this tutorial 300 iterations is sufficient - the model has enough signal to produce clean relativities.

Training takes 30-60 seconds on a standard Databricks cluster.

### Quick calibration check

Before extracting relativities, verify the model is calibrated. In a new cell, type this and run it (Shift+Enter):

```python
predicted_counts = freq_model.predict(train_pool)

print("Calibration check (train set - should be near 1.0):")
print(f"  Actual total claims:    {y_pd.sum():,}")
print(f"  Predicted total claims: {predicted_counts.sum():,.0f}")
print(f"  Ratio (pred/actual):    {predicted_counts.sum() / y_pd.sum():.4f}")
```

You will see:

```sql
Calibration check (train set - should be near 1.0):
  Actual total claims:    4,821
  Predicted total claims: 4,819
  Ratio (pred/actual):    0.9996
```

A ratio close to 1.0 means the model is calibrated on the training set. A Poisson model with a log link and correct exposure offset always calibrates on the training data by construction - if you see a ratio far from 1.0, something is wrong with the exposure offset. This check takes two seconds and catches implementation errors before you spend an hour extracting relativities from a broken model.