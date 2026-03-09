## Part 10: Hyperparameter tuning with Optuna

The default parameters we used in Part 9 (depth=6, learning_rate=0.05, iterations=500) are reasonable starting points but not optimised. Optuna searches the parameter space to find better values.

We tune on the last fold only - train on 2019-2022, validate on 2023. Tuning on all folds is more rigorous but multiplies compute time by the number of folds. For a 100,000-policy book, 40 trials on a single fold takes 15-20 minutes on a standard cluster.

First, extract the last fold data. Create a new cell:

```python
# Use the last fold for tuning: train 2019-2022, validate 2023
# (fold index 2, since folds is 0-indexed)
train_idx_t, val_idx_t = folds[-1]

df_train_t = df_pd.iloc[train_idx_t]
df_val_t   = df_pd.iloc[val_idx_t]

X_train_t = df_train_t[FEATURES]
y_train_t = df_train_t[FREQ_TARGET].values
w_train_t = df_train_t[EXPOSURE_COL].values

X_val_t = df_val_t[FEATURES]
y_val_t = df_val_t[FREQ_TARGET].values
w_val_t = df_val_t[EXPOSURE_COL].values

# Build Pool objects ONCE outside the objective function.
# CatBoost re-encodes categoricals at Pool construction time.
# If you construct inside the objective, this encoding work happens
# on every trial - wasted effort across 40 trials.
train_pool_t = Pool(X_train_t, y_train_t, baseline=np.log(w_train_t), cat_features=CAT_FEATURES)
val_pool_t   = Pool(X_val_t,   y_val_t,   baseline=np.log(w_val_t),   cat_features=CAT_FEATURES)

print(f"Tuning on: {sorted(df_train_t['accident_year'].unique().tolist())} -> validate {sorted(df_val_t['accident_year'].unique().tolist())}")
```

Now define the Optuna objective function. Create a new cell:

```python
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna's own verbose output

def objective(trial: optuna.Trial) -> float:
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 1000),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson",
        "eval_metric":   "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool_t, eval_set=val_pool_t)
    pred = model.predict(val_pool_t)
    return poisson_deviance(y_val_t, pred, w_val_t)
```

**What each parameter does:**

- **depth**: the maximum depth of each tree. Controls how many features can interact. Depth 4 means at most 4-way interactions. For motor data with 5-8 features, depth 4-6 is usually optimal. Depth 7+ overfits without improving validation deviance.
- **learning_rate**: how large a step each tree takes. Lower rates require more iterations but generalise better. We search on a log scale (0.02 to 0.15) because the effect is multiplicative.
- **l2_leaf_reg**: L2 regularisation on leaf values. Increase this if training deviance is much lower than validation deviance - it is the standard overfitting signal.
- **iterations**: the number of trees. Interacts with learning_rate - a low rate needs more iterations to converge. Optuna handles this by exploring the joint space.

Now run the study. Create a new cell:

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest Poisson deviance (40 trials): {study.best_value:.5f}")
print("\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
```

Run this. Optuna runs 40 trials. On Free Edition this takes 10-20 minutes. The progress bar shows completed trials. Let it run.

**What Optuna is doing internally:** The first 10-15 trials are essentially random exploration. After that, Optuna uses a Tree-structured Parzen Estimator (TPE) to concentrate subsequent trials on the most promising regions of the parameter space. The marginal improvement from trials 30-40 is typically small compared to trials 1-20 - we use 40 to be thorough.

After it finishes, run this in the next cell to see which parameters drove most of the variation across trials:

```python
importances = optuna.importance.get_param_importances(study)
print("Parameter importances (what drove trial-to-trial variation):")
for param, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {imp:.3f}")
```

On UK motor data with 5-8 features, typical results: `depth` accounts for 40-55% of trial variance, `learning_rate` 25-35%, `l2_leaf_reg` 10-20%, `iterations` 5-15%. This tells you that depth is the parameter worth tuning most carefully. If compute time is limited, fixing depth=5 and tuning only learning_rate and iterations in 20 trials will get you within 0.001-0.002 deviance of the full search.