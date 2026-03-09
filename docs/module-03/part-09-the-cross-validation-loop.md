## Part 9: The cross-validation loop

Now we put it together. The CV loop trains a fresh CatBoost model on each fold and evaluates it on the held-out year. We track the deviance across folds to get a mean and variance.

Create a new cell:

```python
cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_train = df_pd.iloc[train_idx]
    df_val   = df_pd.iloc[val_idx]

    X_train = df_train[FEATURES]
    y_train = df_train[FREQ_TARGET].values
    w_train = df_train[EXPOSURE_COL].values

    X_val = df_val[FEATURES]
    y_val = df_val[FREQ_TARGET].values
    w_val = df_val[EXPOSURE_COL].values

    train_pool = Pool(X_train, y_train, baseline=np.log(w_train), cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val,   baseline=np.log(w_val),   cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        verbose=0,     # 0 means no training progress output per fold
    )
    model.fit(train_pool, eval_set=val_pool)

    y_pred = model.predict(val_pool)
    fold_dev = poisson_deviance(y_val, y_pred, w_val)
    cv_deviances.append(fold_dev)

    val_year = sorted(df_pd.iloc[val_idx]["accident_year"].unique().tolist())
    print(f"Fold {fold_idx+1} (validate {val_year}): Poisson deviance = {fold_dev:.4f}")

print(f"\nMean CV deviance: {np.mean(cv_deviances):.4f}")
print(f"Std CV deviance:  {np.std(cv_deviances):.4f}")
```

Run this cell. It trains three CatBoost models. On Free Edition this takes 2-4 minutes. The output should look like:

```sql
Fold 1 (validate [2022]): Poisson deviance = 0.1923
Fold 2 (validate [2023]): Poisson deviance = 0.1887
Fold 3 (validate [2024]): Poisson deviance = 0.1841
```

```sql
Mean CV deviance: 0.1884
Std CV deviance:  0.0034
```

The exact numbers will vary slightly. What you are looking for is that the three folds produce similar deviances - if one fold is substantially worse than the others, it may indicate a data quality issue in that year or a genuine model instability.

**A note on what the standard deviation means:** Three folds gives a crude estimate of variance, not a proper confidence interval. Do not report a confidence interval computed from three folds as if it is statistically rigorous. What you can say is: "CV deviance across three temporal folds ranged from X to Y, with a mean of Z."

**What `verbose=0` does:** By default, CatBoost prints training progress on every iteration. On 500 iterations across three folds, that is 1,500 lines of output. Setting `verbose=0` suppresses this. If you want to see progress (useful when debugging), change it to `verbose=100` to print every 100 iterations.