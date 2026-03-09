## Part 9: Stage 5 -- Hyperparameter tuning with Optuna

We tune on the last fold -- the most recent out-of-time period, and therefore the most realistic validation scenario.

Add a markdown cell:

```python
%md
## Stage 5: Hyperparameter tuning with Optuna
```

### Setting up the tuning fold

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Get the last fold
folds = list(cv.split(features_pd))
last_train_idx, last_val_idx = folds[-1]

df_tune_tr = features_pd.loc[last_train_idx]
df_tune_va = features_pd.loc[last_val_idx]

X_tune_tr = df_tune_tr[FEATURE_COLS]
y_tune_tr = df_tune_tr["claim_count"].values
w_tune_tr = df_tune_tr["exposure"].values

X_tune_va = df_tune_va[FEATURE_COLS]
y_tune_va = df_tune_va["claim_count"].values
w_tune_va = df_tune_va["exposure"].values

# Pre-build the Pools (they do not change between trials)
tune_train_pool = Pool(
    X_tune_tr, y_tune_tr,
    baseline=np.log(np.clip(w_tune_tr, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
tune_val_pool = Pool(
    X_tune_va, y_tune_va,
    baseline=np.log(np.clip(w_tune_va, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

print(f"Tuning fold: train n={len(df_tune_tr):,}, val n={len(df_tune_va):,}")
```

### The frequency model Optuna objective

```python
def freq_objective(trial: optuna.Trial) -> float:
    """
    Each call to this function represents one Optuna trial.
    Optuna calls it N_OPTUNA_TRIALS times, each time with different
    parameter suggestions. It returns the Poisson deviance on the
    validation fold -- Optuna minimises this.

    Parameter notes:
    - iterations: tree count. More trees improve fit but increase training time.
      [200, 600] covers the useful range for this dataset size.
    - depth: max tree depth. Insurance data rarely benefits from depth > 6.
      Depth 4-5 is usually optimal for personal lines.
    - learning_rate: step size for gradient descent. Log-uniform because
      the impact is multiplicative: 0.02 vs 0.05 matters more than 0.10 vs 0.13.
    - l2_leaf_reg: L2 regularisation on leaf weights. Prevents overfitting
      in thin cells. Equivalent to lambda in LightGBM.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(tune_train_pool, eval_set=tune_val_pool)
    pred  = model.predict(tune_val_pool)
    return poisson_deviance(y_tune_va, pred, w_tune_va)

freq_study = optuna.create_study(direction="minimize")
freq_study.optimize(freq_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_freq_params = freq_study.best_params
best_freq_params["loss_function"] = "Poisson"
best_freq_params["random_seed"]   = 42
best_freq_params["verbose"]       = 0

print(f"\nBest frequency model parameters (trial {freq_study.best_trial.number}):")
for k, v in best_freq_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest tuning deviance: {freq_study.best_value:.5f}")
print(f"Improvement over default: "
      f"{cv_deviances[-1] - freq_study.best_value:.5f}")
```

### The severity model Optuna objective

The severity model requires its own tuning study. It trains only on policies with at least one claim, uses a Gamma loss function, and does not use an exposure offset. We run it here with the same structure as the frequency study.

```python
# -----------------------------------------------------------------------
# Build the claims-only training set for severity tuning
# -----------------------------------------------------------------------
# The severity model trains on policies with at least one claim.
# Zero-claim policies have no observed severity and cannot contribute
# to the severity likelihood.
#
# We partition the last fold's training set into claims-only for severity.
# The validation set is also claims-only for fair evaluation.
# -----------------------------------------------------------------------

df_sev_tune_tr = df_tune_tr[df_tune_tr["claim_count"] > 0].copy()
df_sev_tune_va = df_tune_va[df_tune_va["claim_count"] > 0].copy()

# Severity target: incurred loss per claim
# We use mean severity per claim, not total incurred per policy,
# because a policy with 3 claims should not be 3x more influential
# than a policy with 1 claim at the same severity.
df_sev_tune_tr["mean_sev"] = (
    df_sev_tune_tr["claim_amount"] / df_sev_tune_tr["claim_count"]
)
df_sev_tune_va["mean_sev"] = (
    df_sev_tune_va["claim_amount"] / df_sev_tune_va["claim_count"]
)

X_sev_tr = df_sev_tune_tr[FEATURE_COLS]
y_sev_tr = df_sev_tune_tr["mean_sev"].values
w_sev_tr = df_sev_tune_tr["claim_count"].values  # weight by claim count

X_sev_va = df_sev_tune_va[FEATURE_COLS]
y_sev_va = df_sev_tune_va["mean_sev"].values
w_sev_va = df_sev_tune_va["claim_count"].values

# Build Pools -- note: no baseline for severity (no exposure offset)
sev_tune_train_pool = Pool(
    X_sev_tr, y_sev_tr,
    weight=w_sev_tr,       # weight by number of claims (more claims = more evidence)
    cat_features=CAT_FEATURES,
)
sev_tune_val_pool = Pool(
    X_sev_va, y_sev_va,
    weight=w_sev_va,
    cat_features=CAT_FEATURES,
)

print(f"Severity tuning fold: train n={len(df_sev_tune_tr):,} (claims-only), "
      f"val n={len(df_sev_tune_va):,}")

def sev_objective(trial: optuna.Trial) -> float:
    """
    Severity model objective: minimise RMSE on held-out claims.
    Tweedie with variance_power=2 is a Gamma distribution.
    Gamma is appropriate for positive-only severity with right skew.

    We use RMSE as the evaluation metric because it is interpretable
    in the units of the severity (pounds). An RMSE of 800 means the model
    is off by roughly £800 on average for held-out claims.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Tweedie:variance_power=2",  # Gamma
        "eval_metric":   "RMSE",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(sev_tune_train_pool, eval_set=sev_tune_val_pool)
    pred = model.predict(sev_tune_val_pool)
    # RMSE in pounds
    return float(np.sqrt(np.mean((y_sev_va - pred)**2)))

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_sev_params = sev_study.best_params
best_sev_params["loss_function"] = "Tweedie:variance_power=2"
best_sev_params["eval_metric"]   = "RMSE"
best_sev_params["random_seed"]   = 42
best_sev_params["verbose"]       = 0

print(f"\nBest severity model parameters (trial {sev_study.best_trial.number}):")
for k, v in best_sev_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest severity RMSE:  £{sev_study.best_value:,.0f}")
```