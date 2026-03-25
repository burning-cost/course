## Part 9: Stage 5 — Hyperparameter tuning with Optuna

We tune on the last fold — the most recent out-of-time period. Tuning on any earlier fold would optimise parameters for a period that the final model does not need to predict.

We run separate tuning studies for frequency and severity. The frequency model is a Poisson regression on claim counts with a log-exposure offset. The severity model is a Gamma regression on mean claim cost, conditioned on claims occurring. The optimal tree depth for a 200,000-row Poisson problem is not the same as for a 14,000-row Gamma problem. Sharing hyperparameters between them is a tutorial simplification that costs several points of severity RMSE.

Add a markdown cell:

```python
%md
## Stage 5: Hyperparameter tuning — Optuna
```

### Setting up the tuning fold

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Last fold from Stage 4
last_train_idx, last_val_idx = folds[-1]
df_tune_tr = features_pd.loc[last_train_idx]
df_tune_va = features_pd.loc[last_val_idx]

tune_train_pool = Pool(
    df_tune_tr[FEATURE_COLS],
    df_tune_tr["claim_count"].values,
    baseline=np.log(np.clip(df_tune_tr["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
tune_val_pool = Pool(
    df_tune_va[FEATURE_COLS],
    df_tune_va["claim_count"].values,
    baseline=np.log(np.clip(df_tune_va["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

print(f"Tuning fold: train n={len(df_tune_tr):,}, val n={len(df_tune_va):,}")
print(f"Train claims:  {df_tune_tr['claim_count'].sum():,}")
print(f"Val claims:    {df_tune_va['claim_count'].sum():,}")
```

### Frequency model tuning

```python
def freq_objective(trial: optuna.Trial) -> float:
    """
    Minimise Poisson deviance on the last validation fold.

    Parameter choices:
    - iterations [200, 600]: tree count. 200-400 is usually optimal for
      personal lines with 150,000 training rows. More trees past the
      elbow improve CV deviance by less than 0.001 per 100 trees.
    - depth [4, 7]: max tree depth. Insurance data rarely benefits from
      depth > 6. Depth 4-5 is optimal for most personal lines.
    - learning_rate [0.02, 0.15]: log-uniform because the improvement
      from 0.02 to 0.05 is similar in scale to 0.05 to 0.13.
    - l2_leaf_reg [1.0, 10.0]: leaf weight regularisation. Prevents
      overfitting in thin interaction cells.
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
    m    = CatBoostRegressor(**params)
    m.fit(tune_train_pool, eval_set=tune_val_pool)
    pred = m.predict(tune_val_pool)
    return poisson_deviance(df_tune_va["claim_count"].values, pred,
                            df_tune_va["exposure"].values)

freq_study = optuna.create_study(direction="minimize")
freq_study.optimize(freq_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_freq_params = {
    **freq_study.best_params,
    "loss_function": "Poisson",
    "random_seed":   42,
    "verbose":       0,
}

print(f"\nBest frequency model (trial {freq_study.best_trial.number}):")
for k, v in best_freq_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest deviance:     {freq_study.best_value:.5f}")
print(f"Default deviance:  {cv_deviances[-1]:.5f}")
print(f"Improvement:       {cv_deviances[-1] - freq_study.best_value:.5f}")
```

### Severity model tuning

The severity model trains on policies with at least one claim. The target is mean cost per claim (`incurred_loss / claim_count`). Policies are weighted by claim count — a policy with three claims has three times more evidence about severity than a policy with one claim.

```python
df_sev_tune_tr = df_tune_tr[df_tune_tr["claim_count"] > 0].copy()
df_sev_tune_va = df_tune_va[df_tune_va["claim_count"] > 0].copy()

df_sev_tune_tr = df_sev_tune_tr.assign(
    mean_sev=df_sev_tune_tr["incurred_loss"] / df_sev_tune_tr["claim_count"]
)
df_sev_tune_va = df_sev_tune_va.assign(
    mean_sev=df_sev_tune_va["incurred_loss"] / df_sev_tune_va["claim_count"]
)

sev_tune_train_pool = Pool(
    df_sev_tune_tr[FEATURE_COLS],
    df_sev_tune_tr["mean_sev"].values,
    weight=df_sev_tune_tr["claim_count"].values,  # weight by claim count, NOT baseline
    cat_features=CAT_FEATURES,
)
sev_tune_val_pool = Pool(
    df_sev_tune_va[FEATURE_COLS],
    df_sev_tune_va["mean_sev"].values,
    weight=df_sev_tune_va["claim_count"].values,
    cat_features=CAT_FEATURES,
)

print(f"Severity tuning: train n={len(df_sev_tune_tr):,} claims-only, "
      f"val n={len(df_sev_tune_va):,}")


def sev_objective(trial: optuna.Trial) -> float:
    """
    Minimise RMSE on held-out mean severity.

    Tweedie with variance_power=2 is the Gamma distribution.
    Gamma is appropriate for positive-only severity with right skew.
    RMSE is used as the eval metric because it is interpretable in £.
    An RMSE of £800 means the model is off by roughly £800 on average
    for held-out claims — a number the underwriting committee can assess.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Tweedie:variance_power=2",
        "eval_metric":   "RMSE",
        "random_seed":   42,
        "verbose":       0,
    }
    m    = CatBoostRegressor(**params)
    m.fit(sev_tune_train_pool, eval_set=sev_tune_val_pool)
    pred = m.predict(sev_tune_val_pool)
    return float(np.sqrt(np.mean((df_sev_tune_va["mean_sev"].values - pred) ** 2)))

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_sev_params = {
    **sev_study.best_params,
    "loss_function": "Tweedie:variance_power=2",
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       0,
}

print(f"\nBest severity model (trial {sev_study.best_trial.number}):")
for k, v in best_sev_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest severity RMSE: £{sev_study.best_value:,.0f}")
```

**What you should see:** Both studies complete without error. The frequency best deviance should be lower than the fold 3 CV deviance from Stage 4. The severity RMSE should be in the range £500-£1,200 depending on the synthetic data random seed.

**Tuning on the last fold only.** We do not average Optuna performance across all folds because that would optimise for historical accuracy — what parameters work well on periods 2022-2024. We want parameters that work well on the most recent period, 2025, which is the closest proxy for the prospective portfolio. Fold 3 (train on 2022-2024, validate on 2025) is that proxy.

**The 20-trial default.** Twenty trials is adequate for this tutorial. In production, 40-60 trials on the frequency model and 30-40 on severity adds roughly one percentage point of improvement in validation deviance for a 4-6x increase in compute time. Whether that is worth the cost depends on your cluster configuration and review cycle timeline.
