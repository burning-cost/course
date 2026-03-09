## Part 8: Stage 4 -- Walk-forward cross-validation

Cross-validation in insurance must be temporal. We covered the reasons in Part 2. This stage implements a walk-forward CV with an IBNR buffer.

Add a markdown cell:

```python
%md
## Stage 4: Walk-forward cross-validation
```

### Setting up the folds

```python
from insurance_cv import WalkForwardCV, IBNRBuffer

# -----------------------------------------------------------------------
# Fold setup: three temporal folds.
# Each fold trains on progressively more data and validates on the
# next accident year.
#
# Fold 1: train on 2019-2021, validate on 2022
# Fold 2: train on 2019-2022, validate on 2023
# Fold 3: train on 2019-2022 (with IBNR buffer), validate on 2023
#
# The IBNR buffer removes the most recent 6 months from each training fold.
# For accident year data (not monthly), this trims the 2022 H2 data from
# Fold 2's training set. In production with monthly data, it removes the
# six most recent months from the training window.
# -----------------------------------------------------------------------

ibnr_buffer = IBNRBuffer(months=6)
cv = WalkForwardCV(
    date_col="accident_year",
    n_splits=3,
    min_train_years=2,
    ibnr_buffer=ibnr_buffer,
)

# Convert features to pandas for the CV loop (CatBoost Pool requires pandas)
features_pd = features_pl.to_pandas()

print("Cross-validation folds:")
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features_pd)):
    train_years = sorted(features_pd.loc[train_idx, "accident_year"].unique())
    val_years   = sorted(features_pd.loc[val_idx,   "accident_year"].unique())
    print(f"  Fold {fold_idx+1}: train years={train_years}, val years={val_years}, "
          f"n_train={len(train_idx):,}, n_val={len(val_idx):,}")
```

### The Poisson deviance metric

```python
def poisson_deviance(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     exposure: np.ndarray) -> float:
    """
    Scaled Poisson deviance for frequency models.

    y_true:   observed claim counts (not frequencies)
    y_pred:   predicted claim counts (exposure * predicted frequency)
    exposure: policy exposure in years

    The scaling by total exposure gives a per-policy-year deviance,
    making it comparable across folds with different exposure totals.

    MSE is the wrong metric for Poisson regression:
      - MSE penalises a miss of 2 claims the same whether the policy has
        0.01 expected claims or 0.5 expected claims
      - Poisson deviance weights residuals correctly by the expected count

    A null model (always predict the portfolio mean frequency) achieves
    a Poisson deviance around 0.08-0.12 on UK motor data. A good GBM
    achieves 0.04-0.07. Anything below 0.04 should be investigated for
    data leakage.
    """
    fp = np.clip(y_pred / exposure, 1e-10, None)   # predicted frequency
    ft = y_true / exposure                          # observed frequency
    d  = 2 * exposure * (
        np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp)
    )
    return float(d.sum() / exposure.sum())
```

### Running the CV loop

```python
cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features_pd)):
    df_tr = features_pd.loc[train_idx]
    df_va = features_pd.loc[val_idx]

    X_tr = df_tr[FEATURE_COLS]
    y_tr = df_tr["claim_count"].values
    w_tr = df_tr["exposure"].values

    X_va = df_va[FEATURE_COLS]
    y_va = df_va["claim_count"].values
    w_va = df_va["exposure"].values

    # ----------------------------------------------------------------
    # IMPORTANT: the Poisson offset is log(exposure), not exposure.
    # baseline= adds a fixed term to the model output before the loss.
    # The model then predicts: exp(log(exposure) + f(features))
    #                        = exposure * exp(f(features))
    # which is the expected claim count for a policy with this exposure.
    # ----------------------------------------------------------------
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

    # Default params for CV: these are not tuned yet
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

    pred_va   = cv_model.predict(val_pool)
    fold_dev  = poisson_deviance(y_va, pred_va, w_va)
    cv_deviances.append(fold_dev)

    print(f"Fold {fold_idx+1}: Poisson deviance = {fold_dev:.5f}  "
          f"(n_val={len(val_idx):,}, exposure={w_va.sum():.0f} years)")

mean_cv_deviance = float(np.mean(cv_deviances))
print(f"\nMean CV deviance: {mean_cv_deviance:.5f}")
print(f"Fold deviances:   {[round(d, 5) for d in cv_deviances]}")
```

**What you should see:** Three fold deviances, each between roughly 0.04 and 0.09 for the synthetic data. The last fold may be higher if accident year 2023 is partially immature in the synthetic data. If any fold deviance is below 0.01, check for data leakage -- you may have future information in the training features.

**What the IBNR buffer does in practice:** If your data is monthly (e.g., accident month rather than accident year), the six-month buffer removes the six most recent months from each fold's training set. A policy written in June 2024 would not appear in a fold that trains on data through December 2024. This forces the model to learn from claims that have had at least six months to develop, reducing IBNR contamination.

For this notebook, we use annual data so the buffer effect is limited. In production, move to monthly accident periods and set the buffer to 12 months for motor property damage, 24 months for motor bodily injury.