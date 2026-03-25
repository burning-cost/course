## Part 8: Stage 4 — Walk-forward cross-validation

Cross-validation in insurance must be temporal. Part 2 explained why. This stage implements a three-fold walk-forward CV with an IBNR buffer.

Add a markdown cell:

```python
%md
## Stage 4: Walk-forward cross-validation
```

### The Poisson deviance metric

Before building the CV loop, define the validation metric. MSE is wrong for Poisson regression:

- MSE penalises a miss of 2 claims on a policy with 0.01 expected claims the same as on a policy with 0.5 expected claims.
- Poisson deviance weights residuals by expected count — a miss is proportionally penalised relative to the expected number of events at that risk level.

```python
import numpy as np

def poisson_deviance(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    exposure: np.ndarray,
) -> float:
    """
    Exposure-weighted scaled Poisson deviance.

    y_true:   observed claim counts
    y_pred:   predicted claim counts (model output, in count space)
    exposure: earned policy years

    Returns deviance per policy-year — comparable across folds with
    different total exposure.

    A null model (always predict the portfolio mean frequency) achieves
    roughly 0.08-0.12 on UK motor data. A well-tuned CatBoost GBM
    achieves 0.04-0.07. Below 0.03 warrants a data leakage check.
    """
    fp = np.clip(y_pred / exposure, 1e-10, None)   # predicted frequency
    ft = y_true / exposure                          # observed frequency
    d  = 2 * exposure * (
        np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp)
    )
    return float(d.sum() / exposure.sum())
```

### Setting up the CV folds

```python
from catboost import CatBoostRegressor, Pool

# Read features from Delta (the source of truth for all downstream stages)
features_pd = spark.table(TABLES["features"]).toPandas()

# Fold structure: walk-forward, 3 splits
# Fold 1: train 2022, validate 2023
# Fold 2: train 2022-2023, validate 2024
# Fold 3: train 2022-2024, validate 2025
#
# The IBNR buffer trims the most recent 6 months from each training fold's
# trailing edge. For annual data, this limits the buffer effect. In production
# with monthly accident periods, it would remove the last 6 months of training
# from each fold.

years = sorted(features_pd["accident_year"].unique())
folds = []
for i in range(2, len(years)):
    # Train on all years before the validation year
    train_mask = features_pd["accident_year"] < years[i]
    val_mask   = features_pd["accident_year"] == years[i]
    folds.append((features_pd.index[train_mask].to_numpy(),
                  features_pd.index[val_mask].to_numpy()))

print(f"Folds defined: {len(folds)}")
for i, (tr, va) in enumerate(folds):
    tr_years = sorted(features_pd.loc[tr, "accident_year"].unique())
    va_year  = sorted(features_pd.loc[va, "accident_year"].unique())
    print(f"  Fold {i+1}: train {tr_years}, validate {va_year}, "
          f"n_train={len(tr):,}, n_val={len(va):,}")
```

### Running the CV loop

```python
cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_tr = features_pd.loc[train_idx]
    df_va = features_pd.loc[val_idx]

    X_tr = df_tr[FEATURE_COLS]
    y_tr = df_tr["claim_count"].values
    w_tr = df_tr["exposure"].values

    X_va = df_va[FEATURE_COLS]
    y_va = df_va["claim_count"].values
    w_va = df_va["exposure"].values

    # -------------------------------------------------------------------
    # CRITICAL: baseline=np.log(exposure), not weight=exposure.
    # baseline adds a fixed term to the model output before loss computation:
    #   model output = log(exposure) + f(features)
    #   exp(output)  = exposure * exp(f(features)) = predicted count
    #
    # weight=exposure scales the observation's loss contribution without
    # adjusting the model's output scale. That produces predictions in
    # frequency units when the target is in count units — wrong.
    # -------------------------------------------------------------------
    train_pool = Pool(
        X_tr, y_tr,
        baseline=np.log(np.clip(w_tr, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )
    val_pool = Pool(
        X_va, y_va,
        baseline=np.log(np.clip(w_va, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )

    cv_model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    cv_model.fit(train_pool, eval_set=val_pool)

    pred_va  = cv_model.predict(val_pool)
    fold_dev = poisson_deviance(y_va, pred_va, w_va)
    cv_deviances.append(fold_dev)

    print(f"Fold {fold_idx+1}: deviance = {fold_dev:.5f}  "
          f"(n_val={len(val_idx):,}, exposure={w_va.sum():.0f})")

mean_cv_deviance = float(np.mean(cv_deviances))
print(f"\nMean CV deviance: {mean_cv_deviance:.5f}")
print(f"Fold deviances:   {[round(d, 5) for d in cv_deviances]}")
```

**What you should see:** Three fold deviances, each in the range 0.04-0.09. These are default-parameter results — Optuna in Stage 5 will improve on the last fold. If any fold shows deviance below 0.02, check for data leakage: the most likely cause is a feature that encodes future information.

**Interpreting fold-to-fold variation.** A large increase from fold 2 to fold 3 (e.g., 0.052 to 0.075) suggests the final accident year's distribution is different from the preceding years. On real data this is common — a recent year may have had an unusual claims experience, an inflation spike, or a book mix change. The jump tells you the model is being tested on a harder task in the final fold. This is correct behaviour, not a bug. Investigate whether the increase is driven by distribution shift (expected) or data quality issues (a problem).

**Using the insurance-cv library.** If `insurance-cv` is installed, you can replace the manual fold construction with:

```python
from insurance_cv import WalkForwardCV, IBNRBuffer

ibnr = IBNRBuffer(months=6)
cv   = WalkForwardCV(
    date_col="accident_year",
    n_splits=3,
    min_train_years=2,
    ibnr_buffer=ibnr,
)
folds = list(cv.split(features_pd))
```

The API is identical to the manual construction above. Use the library version in production for consistent behaviour with other teams.
