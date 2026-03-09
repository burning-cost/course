# Module 3 Exercises: GBMs for Insurance Pricing

Five exercises. Work through them in order - each builds on the previous. All of them assume you have run the tutorial notebook through to Part 14.

**Before you start:** Open your `module-03-gbm-catboost` notebook and make sure the cluster is running. Check that the following variables are in scope from the tutorial:

- `df` - the 100,000-policy Polars DataFrame
- `df_pd` - the pandas version of `df`
- `folds` - the walk-forward CV splits from Part 6
- `freq_model` - the fitted CatBoost frequency model from Part 11
- `sev_model` - the fitted CatBoost severity model from Part 13
- `FEATURES`, `CAT_FEATURES`, `FREQ_TARGET`, `EXPOSURE_COL` - the feature definitions from Part 5
- `poisson_deviance` - the function from Part 8

If any of these are missing (e.g. the cluster restarted), re-run the cells from the top of the tutorial notebook in order before starting the exercises.

---

## Exercise 1: The offset trap - seeing the error quantified

**What this covers:** Understanding why `baseline=np.log(exposure)` is required rather than `baseline=exposure`, and how the error manifests in the predictions.

**The setup:** Create a new notebook in Databricks called `module-03-exercise-01`. Install the required libraries in the first cell and restart Python:

```python
%pip install catboost mlflow polars
```

```python
dbutils.library.restartPython()
```

Then import everything:

```python
import numpy as np
import polars as pl
import mlflow
from catboost import CatBoostRegressor, Pool
```

Now generate a small synthetic dataset. In a new cell, paste this:

```python
rng = np.random.default_rng(77)
n = 20_000

area          = rng.choice(["A", "B", "C", "D"], size=n, p=[0.25, 0.30, 0.25, 0.20])
ncd           = rng.integers(0, 6, size=n)
vehicle_group = rng.integers(1, 51, size=n)
driver_age    = rng.integers(17, 85, size=n)
exposure      = rng.uniform(0.25, 1.0, size=n)

area_eff = {"A": 0.0, "B": 0.10, "C": 0.25, "D": 0.40}
log_mu = (
    -3.0
    + np.array([area_eff[a] for a in area])
    + (-0.12) * ncd
    + 0.008   * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.50, 0.0)
)
claim_count = rng.poisson(np.exp(log_mu) * exposure)

df = pl.DataFrame({
    "area":         area,
    "ncd":          ncd,
    "vehicle_group": vehicle_group.astype(np.int32),
    "driver_age":   driver_age.astype(np.int32),
    "exposure":     exposure,
    "claim_count":  claim_count.astype(np.int32),
})

train = df[:16_000]
test  = df[16_000:]

FEATURES     = ["area", "vehicle_group", "ncd", "driver_age"]
CAT_FEATURES = ["area"]
```

Note the true young driver uplift: `exp(0.50) = 1.65`. The model with the correct offset should recover approximately this multiplier when comparing predicted frequencies for young versus older drivers.

**Task 1:** Copy the Poisson deviance function from the tutorial (Part 8) into a new cell. Then fit the same CatBoost Poisson model in three ways:

- Version A: `baseline=np.log(exposure)` (correct)
- Version B: `baseline=exposure` (wrong - missing the log transform)
- Version C: no baseline at all (assumes all exposure = 1.0)

For each version, compute the mean predicted frequency on the test set: `sum(predicted_counts) / sum(exposure)`. Compare each to the true frequency: `sum(claim_counts) / sum(exposure)`.

Use these fixed parameters for all three versions (so you isolate the offset difference):

```python
base_params = dict(
    loss_function="Poisson",
    eval_metric="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
    verbose=0,
)
```

**Task 2:** For Version A (correct), compute the mean predicted frequency separately for young drivers (age < 25) and the rest. Does the model recover the true uplift of approximately `exp(0.50) = 1.65`?

```python
# Hint: after predicting, divide by exposure to get frequency
# then mask by driver_age
```

**Task 3:** Log the test Poisson deviance for all three versions using MLflow. Set up an experiment first:

```python
mlflow.set_experiment("/module_03_exercise_01")
```

Then for each version, use `with mlflow.start_run(run_name=version_name):` and call `mlflow.log_metric("test_poisson_deviance", dev)` and `mlflow.log_param("baseline", "log_exposure or raw_exposure or none")`. After running, open the Experiments panel in Databricks and compare the three runs. Which version has the lowest test deviance?

<details>
<summary>Hint for Task 1</summary>

CatBoost's `baseline` is added directly to the linear predictor before the log link is applied. With a log link:

- Correct: `exp(f(x) + log(exposure)) = exp(f(x)) * exposure`
- Wrong: `exp(f(x) + exposure)` = predicts exposure 0.5 as `exp(0.5) = 1.65x` too high
- No offset: `exp(f(x))` = assumes exposure = 1.0 for all policies

The error in Version B is exposure-dependent, so it is worse for policies with extreme exposures (very short or full-year). It correlates with writing month and cancellation patterns, not with risk.

</details>

<details>
<summary>Solution - Exercise 1</summary>

```python
import numpy as np
import polars as pl
import mlflow
from catboost import CatBoostRegressor, Pool

def poisson_deviance(y_true, y_pred, exposure):
    freq_true = y_true / exposure
    freq_pred = np.clip(y_pred / exposure, 1e-10, None)
    d = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true / freq_pred), 0.0)
        - (freq_true - freq_pred)
    )
    return d.sum() / exposure.sum()

X_train = train[FEATURES].to_pandas()
y_train = train["claim_count"].to_numpy()
w_train = train["exposure"].to_numpy()
X_test  = test[FEATURES].to_pandas()
y_test  = test["claim_count"].to_numpy()
w_test  = test["exposure"].to_numpy()

true_freq = y_test.sum() / w_test.sum()
print(f"True frequency (test): {true_freq:.4f}")

mlflow.set_experiment("/module_03_exercise_01")

results = {}
for version, bl_train, bl_test in [
    ("correct_log",   np.log(w_train), np.log(w_test)),
    ("wrong_no_log",  w_train,         w_test),
    ("no_offset",     None,            None),
]:
    tp = Pool(X_train, y_train, baseline=bl_train, cat_features=CAT_FEATURES)
    vp = Pool(X_test,  y_test,  baseline=bl_test,  cat_features=CAT_FEATURES)
    m  = CatBoostRegressor(**base_params)
    m.fit(tp, eval_set=vp)
    preds     = m.predict(vp)
    mean_freq = preds.sum() / w_test.sum()
    dev       = poisson_deviance(y_test, preds, w_test)
    results[version] = {"mean_freq": mean_freq, "deviance": dev, "model": m}

    with mlflow.start_run(run_name=f"offset_{version}"):
        mlflow.log_metric("test_poisson_deviance",    dev)
        mlflow.log_metric("mean_predicted_frequency", mean_freq)
        mlflow.log_param("baseline", version)

    print(f"{version:20s}: mean_freq={mean_freq:.4f}  deviance={dev:.4f}")

# Task 2: young driver uplift check
m_correct = results["correct_log"]["model"]
test_pd   = test.to_pandas()
test_pd["pred_freq"] = (
    m_correct.predict(Pool(X_test, baseline=np.log(w_test), cat_features=CAT_FEATURES))
    / w_test
)
young = test_pd[test_pd["driver_age"] < 25]["pred_freq"].mean()
other = test_pd[test_pd["driver_age"] >= 25]["pred_freq"].mean()
print(f"\nTask 2 - Young driver uplift")
print(f"  Mean freq (age < 25):  {young:.4f}")
print(f"  Mean freq (age >= 25): {other:.4f}")
print(f"  Recovered uplift:      {young / other:.3f}  (true: {np.exp(0.50):.3f})")
```

**Discussion:** Version B (baseline = raw exposure) produces calibration errors that vary by exposure level. The model trains fine - CatBoost has no way of knowing the baseline is wrong - but the predictions are systematically off in a way that correlates with policy characteristics rather than risk. Version C (no baseline) overestimates predicted counts uniformly. Both errors look like model problems on a calibration plot, but the root cause is an implementation choice. The correct version (A) recovers the true young driver uplift of approximately 1.65.

</details>

---

## Exercise 2: Walk-forward CV vs random split - quantifying the optimism gap

**What this covers:** Measuring how much a random split overstates out-of-sample performance compared to a proper temporal split on insurance data.

**The setup:** You can do this in your main `module-03-gbm-catboost` notebook. `df` and `df_pd` are already in scope from the tutorial. You also need `WalkForwardCV` - check that `from insurance_cv import WalkForwardCV` is imported (it should be from the tutorial).

Create a new section at the bottom of your notebook with a markdown cell:

```python
%md
## Exercise 2: Walk-forward CV vs random split
```

You will also need sklearn's train_test_split:

```python
from sklearn.model_selection import train_test_split
```

**Task 1:** Split the data two ways and fit the same CatBoost model (depth=5, learning_rate=0.05, iterations=500, Poisson loss) on each:

- Random split: 80% train, 20% test using `train_test_split(np.arange(len(df_pd)), test_size=0.20, random_state=42)`. This ignores accident year entirely.
- Temporal split: train on `accident_year <= 2022`, test on `accident_year >= 2023`.

Evaluate Poisson deviance on each test set. Report both deviances. Which is lower? The difference between them is the optimism in the random split - the extent to which a random 80/20 split overstates performance.

Write a helper function first to avoid repeating yourself:

```python
def fit_evaluate(train_idx, test_idx, label=""):
    Xtr = df_pd.iloc[train_idx][FEATURES]
    ytr = df_pd.iloc[train_idx][FREQ_TARGET].values
    wtr = df_pd.iloc[train_idx][EXPOSURE_COL].values
    Xte = df_pd.iloc[test_idx][FEATURES]
    yte = df_pd.iloc[test_idx][FREQ_TARGET].values
    wte = df_pd.iloc[test_idx][EXPOSURE_COL].values
    tp  = Pool(Xtr, ytr, baseline=np.log(wtr), cat_features=CAT_FEATURES)
    vp  = Pool(Xte, yte, baseline=np.log(wte), cat_features=CAT_FEATURES)
    m   = CatBoostRegressor(
              loss_function="Poisson", eval_metric="Poisson",
              iterations=500, learning_rate=0.05, depth=5,
              random_seed=42, verbose=0)
    m.fit(tp, eval_set=vp)
    dev = poisson_deviance(yte, m.predict(vp), wte)
    if label:
        print(f"{label}: deviance = {dev:.4f}")
    return dev, m
```


**Task 2:** Using the `folds` variable from Part 6 of the tutorial, compute the mean CV deviance across all three folds. Compare it to the temporal split deviance from Task 1. The IBNR buffer should make the CV deviance slightly worse (more conservative) than a simple temporal split with no buffer, because the training set for each fold excludes the year immediately before the validation year.

**Task 3:** A colleague says: "Our data only goes back to 2019. Walk-forward CV with an IBNR buffer gives us only three usable folds. That is not enough to estimate variance reliably. We should use 5-fold random CV." Write a 3-sentence response explaining why this argument is wrong for insurance pricing models.

<details>
<summary>Hint for Task 1</summary>

Random splits mix policies from all accident years in both training and test. A model trained on 80% of 2019-2024 data sees the full range of accident years in training. When it predicts the remaining 20%, it is not being asked to generalise to new time periods - it is filling in gaps in a history it has already seen. The test set shares the same temporal patterns as the training set, so the deviance looks better than it will be in production.

</details>

<details>
<summary>Solution - Exercise 2</summary>

```python
from sklearn.model_selection import train_test_split

# Task 1
idx_all   = np.arange(len(df_pd))
rnd_train, rnd_test = train_test_split(idx_all, test_size=0.20, random_state=42)

temp_train = df_pd[df_pd["accident_year"] <= 2022].index.to_numpy()
temp_test  = df_pd[df_pd["accident_year"] >= 2023].index.to_numpy()

dev_random,   _ = fit_evaluate(rnd_train,  rnd_test,  "Random split")
dev_temporal, _ = fit_evaluate(temp_train, temp_test, "Temporal split")

print(f"\nOptimism gap: {dev_temporal - dev_random:.4f} ({(dev_temporal/dev_random - 1)*100:.1f}% higher on temporal split)")

# Task 2: CV mean deviance
cv_devs = []
for train_idx, val_idx in folds:
    dev, _ = fit_evaluate(train_idx, val_idx)
    cv_devs.append(dev)
    print(f"  CV fold deviance: {dev:.4f}")
print(f"\nCV mean: {np.mean(cv_devs):.4f}")
print(f"Temporal split: {dev_temporal:.4f}")
```

**Task 3 - response to the colleague:**

Random CV measures how well the model fills in gaps in a history it has already seen. That is not the deployment scenario: in deployment, the model predicts next year from this year's data. Walk-forward CV measures exactly that scenario, and three valid folds are worth more than five invalid ones. The concern about variance is valid - three folds gives a wide interval on the CV estimate - but that interval brackets the true out-of-time performance. Five random folds give a precise estimate of the wrong quantity.

</details>

---

## Exercise 3: CatBoost Gamma severity model and the Tweedie power link

**What this covers:** Fitting a severity model, understanding why exposure is excluded, and comparing loss function choices.

**The setup:** You can continue in your main `module-03-gbm-catboost` notebook. The severity data is already prepared in Part 13 of the tutorial. You need `df_train_sev`, `df_test_sev`, `X_train_s`, `y_train_s`, `X_test_s`, `y_test_s` in scope. If they are not, re-run Part 13.

Add a markdown cell:

```python
%md
## Exercise 3: Severity model and Tweedie power link
```

**Task 1:** Fit three CatBoost severity models using different loss functions:

- `Tweedie:variance_power=2` (Gamma equivalent - what the tutorial uses)
- `Tweedie:variance_power=1.5` (compound Poisson-Gamma - between frequency and severity)
- `RMSE` (no distributional assumption - treats severity as a standard regression)

Use the same parameters for all three (so the comparison is fair):

```python
shared_params = dict(
    iterations=400,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
    verbose=0,
)
```

For each, compute test RMSE and MAE. Which objective produces the best test-set RMSE? You may be surprised.

**Task 2:** Extract feature importances from all three severity models using `model.get_feature_importance()`. Compare to the frequency model's importances from Part 12 of the tutorial. Which features are important for frequency but not severity? Which matter for both? Does this match your intuition about motor insurance?

**Task 3:** Combine the frequency and severity models to produce a pure premium estimate for the test set. The formula is:

```sql
pure_premium = (predicted_claim_count / exposure) * predicted_avg_severity
            = freq_rate * severity
```

Compute: `sum(pure_premium * exposure)` and compare to `sum(actual_incurred)`. What is the ratio? A ratio above 1.05 or below 0.95 at the portfolio level would indicate a calibration problem.

**Task 4:** The tutorial note says severity hyperparameters should be tuned separately from frequency hyperparameters. Run a quick Optuna study (10 trials - keeping it fast) to find the optimal depth for the severity model specifically. Does the optimal depth differ from the frequency model's optimal depth? The severity dataset is smaller (7-10% of all policies) - does this affect what depth is optimal?

<details>
<summary>Hint for Task 1</summary>

The Tweedie power=2 objective weights residuals by the square of the prediction, which gives more relative weight to accurately predicting lower-severity claims than the RMSE objective does. For right-skewed claim severity distributions, this often produces better RMSE despite not optimising RMSE directly. The reason is that RMSE minimisation concentrates on reducing large errors in the tail, which can come at the expense of accuracy in the body of the distribution where most observations lie.

</details>

<details>
<summary>Solution - Exercise 3</summary>

```python
import numpy as np
from catboost import CatBoostRegressor, Pool

shared_params = dict(iterations=400, learning_rate=0.05, depth=5, random_seed=42, verbose=0)

sev_models = {}
for loss_fn, label in [
    ("Tweedie:variance_power=2",   "gamma_equiv"),
    ("Tweedie:variance_power=1.5", "tweedie_1.5"),
    ("RMSE",                       "rmse"),
]:
    tp = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
    vp = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)
    params = {**shared_params, "loss_function": loss_fn}
    m = CatBoostRegressor(**params)
    m.fit(tp, eval_set=vp)
    preds = m.predict(vp)
    rmse  = np.sqrt(np.mean((y_test_s - preds) ** 2))
    mae   = np.mean(np.abs(y_test_s - preds))
    print(f"{label:18s}: RMSE=£{rmse:,.0f}  MAE=£{mae:,.0f}")
    sev_models[label] = m

# Task 2: Feature importances
print("\nFeature importances comparison:")
print(f"{'Feature':<22}", end="")
for label in sev_models:
    print(f" {label:>14}", end="")
print()
for i, feat in enumerate(FEATURES):
    print(f"{feat:<22}", end="")
    for label, m in sev_models.items():
        imp = m.get_feature_importance()[i]
        print(f" {imp:>14.2f}", end="")
    print()

# Task 3: Pure premium calibration
freq_pred_test = freq_model.predict(
    Pool(df_test_final[FEATURES].to_pandas(), baseline=np.log(w_test_f), cat_features=CAT_FEATURES)
)
sev_pred_test  = sev_models["gamma_equiv"].predict(
    Pool(df_test_final[FEATURES].to_pandas(), cat_features=CAT_FEATURES)
)
freq_rate     = freq_pred_test / w_test_f
pure_premium  = freq_rate * sev_pred_test
pred_incurred = (pure_premium * w_test_f).sum()
actual_incurred = df_test_final["incurred"].sum()
print(f"\nTask 3 - Portfolio calibration")
print(f"Actual total incurred:    £{actual_incurred:,.0f}")
print(f"Predicted total incurred: £{pred_incurred:,.0f}")
print(f"Ratio (pred/actual):      {pred_incurred / actual_incurred:.4f}")

# Task 4: Quick Optuna for severity depth
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sev_tp = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_vp = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

def sev_objective(trial):
    p = {
        "depth":         trial.suggest_int("depth", 3, 7),
        "learning_rate": 0.05,
        "iterations":    400,
        "loss_function": "Tweedie:variance_power=2",
        "random_seed":   42,
        "verbose":       0,
    }
    m = CatBoostRegressor(**p)
    m.fit(sev_tp, eval_set=sev_vp)
    preds = m.predict(sev_vp)
    return float(np.sqrt(np.mean((y_test_s - preds) ** 2)))

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=10, show_progress_bar=True)
print(f"\nBest depth for severity: {sev_study.best_params['depth']}")
print(f"Best severity RMSE: £{sev_study.best_value:,.0f}")
```

**Discussion:** The Gamma-equivalent (Tweedie power=2) and RMSE objectives often produce similar test RMSE on hold-out data for motor severity. The Tweedie objective weights errors by the predicted value squared, concentrating on relative accuracy across the distribution rather than absolute error in the tail. NCD years typically appears important for frequency but weak for severity - higher NCD accumulates because the driver does not have accidents, not because their accidents are cheaper when they do occur. Vehicle group tends to matter for both, because expensive vehicles both increase frequency (riskier drivers) and severity (expensive to repair or replace).

</details>

---

## Exercise 4: Hyperparameter sensitivity - what actually matters?

**What this covers:** Understanding which hyperparameters drive performance for insurance data specifically, so you can design efficient tuning searches.

**The setup:** You can add this to the main notebook. `folds` and the feature definitions are in scope from the tutorial. Set up the last fold data again:

```python
%md
## Exercise 4: Hyperparameter sensitivity
```

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Last fold: train 2019-2022, validate 2023
train_idx_t, val_idx_t = folds[-1]
df_tr4 = df_pd.iloc[train_idx_t]
df_va4 = df_pd.iloc[val_idx_t]

X_tr4 = df_tr4[FEATURES]
y_tr4 = df_tr4[FREQ_TARGET].values
w_tr4 = df_tr4[EXPOSURE_COL].values
X_va4 = df_va4[FEATURES]
y_va4 = df_va4[FREQ_TARGET].values
w_va4 = df_va4[EXPOSURE_COL].values

# Build Pools ONCE
tp4 = Pool(X_tr4, y_tr4, baseline=np.log(w_tr4), cat_features=CAT_FEATURES)
vp4 = Pool(X_va4, y_va4, baseline=np.log(w_va4), cat_features=CAT_FEATURES)
```

**Task 1:** Run an Optuna study with 30 trials searching over:
- `depth`: 3-7
- `learning_rate`: 0.01-0.20 (log scale)
- `l2_leaf_reg`: 1.0-15.0
- `iterations`: 200-800

Report the best parameters and best deviance. Use `n_jobs=1` - CatBoost handles its own threading internally, and setting higher n_jobs here causes conflicts.

**Task 2:** After the study, run:

```python
importances = optuna.importance.get_param_importances(study_full)
for param, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {imp:.3f}")
```

Which parameter accounts for the most trial-to-trial variation? Which accounts for the least? Does `iterations` matter as much as `learning_rate`?

**Task 3:** Run a second study with 30 trials, but this time fix `depth=5` and `l2_leaf_reg=3.0`. Only tune `learning_rate` and `iterations`. How much deviance does the restricted search lose compared to the full search?

**Task 4:** Based on your results, write a 4-sentence recommendation: for a book of 100,000 motor policies, what is the minimum viable Optuna search? How many trials and which parameter space? What would you tell a colleague who wants to save compute time?

<details>
<summary>Hint for Task 2</summary>

Optuna's parameter importances use fANOVA - a method that decomposes trial-to-trial variance by parameter. A high importance score means changing that parameter moves the objective substantially. A low score means the objective is relatively flat over that parameter's range. For insurance motor data, depth typically dominates because it controls the complexity of feature interactions, and interaction complexity matters more than fine-tuning the learning schedule.

</details>

<details>
<summary>Solution - Exercise 4</summary>

```python
def full_objective(trial):
    params = {
        "depth":         trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "iterations":    trial.suggest_int("iterations", 200, 800),
        "loss_function": "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    m = CatBoostRegressor(**params)
    m.fit(tp4, eval_set=vp4)
    return poisson_deviance(y_va4, m.predict(vp4), w_va4)

study_full = optuna.create_study(direction="minimize")
study_full.optimize(full_objective, n_trials=30, show_progress_bar=True)
print(f"Full search best:   {study_full.best_value:.5f}")
for k, v in study_full.best_params.items():
    print(f"  {k}: {v}")

# Task 2
importances = optuna.importance.get_param_importances(study_full)
print("\nParameter importances:")
for param, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {imp:.3f}")

# Task 3
def restricted_objective(trial):
    params = {
        "depth":         5,
        "l2_leaf_reg":   3.0,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "iterations":    trial.suggest_int("iterations", 200, 800),
        "loss_function": "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    m = CatBoostRegressor(**params)
    m.fit(tp4, eval_set=vp4)
    return poisson_deviance(y_va4, m.predict(vp4), w_va4)

study_restricted = optuna.create_study(direction="minimize")
study_restricted.optimize(restricted_objective, n_trials=30, show_progress_bar=True)
print(f"\nFull search best:       {study_full.best_value:.5f}")
print(f"Restricted search best: {study_restricted.best_value:.5f}")
print(f"Deviance gap:           {study_restricted.best_value - study_full.best_value:.5f}")
```

**Task 4 - minimum viable tuning recommendation:**

For a 100,000-policy motor book with 5-8 rating factors: 20 trials searching `learning_rate` on a log scale (0.02-0.15) and `iterations` (250-750), with `depth` fixed at 5 and `l2_leaf_reg` fixed at 3. The reason for fixing depth: it rarely moves outside 4-6 on insurance data with this feature count, and fixing it saves roughly 40% of compute versus searching over it. Learning rate and iterations interact tightly and both need tuning. 20 trials gets within 0.001 deviance of the 40-trial optimum in our experience on synthetic motor data. If you have budget for more, run 40 trials and add `l2_leaf_reg` to the search - but the marginal benefit over 20 trials is usually small.

</details>

---

## Exercise 5: The pricing committee presentation - Gini lift and the double lift chart

**What this covers:** Building the complete GBM-vs-GLM comparison you would present to a pricing committee, and writing the accompanying narrative.

**The setup:** You need everything from Part 14 of the tutorial in scope: `freq_model`, `y_pred_freq`, `w_test_f`, `y_test_f`, `glm_freq_pred`, `gbm_freq_pred`, `gini_gbm`, `gini_glm`, and the double lift chart variables. If any are missing, re-run Part 14 of the tutorial.

Add a markdown cell:

```python
%md
## Exercise 5: Pricing committee presentation
```

**Task 1:** Reproduce the Gini calculation from Part 14 in a clean cell. Then add a third comparison: fit a naive benchmark - a model that predicts the same claim frequency for every policy (the portfolio mean). What is the Gini of the naive model? How does this contextualise the GLM and GBM Ginis?

```python
# Hint: the naive model assigns the same score to all policies
# roc_auc_score requires variation in scores, so add a tiny amount of noise:
naive_scores = np.full_like(gbm_freq_pred, actual_freq.mean()) + np.random.default_rng(0).normal(0, 1e-10, len(gbm_freq_pred))
```

The naive Gini should be near zero. The GLM's Gini minus zero gives you the absolute discrimination of the GLM. The GBM's Gini minus the GLM's Gini gives you the incremental discrimination. This framing makes the committee discussion more concrete.

**Task 2:** Re-generate the double lift chart from Part 14. Add a second panel showing the number of policies in each decile (the `n_obs` list from the tutorial). This matters for the committee: a top decile with 50 policies is not statistically meaningful; a top decile with 2,000 policies is.

**Task 3:** Look at the top two deciles of the double lift chart. What do the policies in these deciles have in common? Run the risk profile analysis from Part 14 (the cells after the chart) and summarise: what is the mean driver age, mean vehicle group, and percentage of young drivers (under 25) in the top decile vs the portfolio?

**Task 4:** Write a 4-sentence pricing committee summary. It must include: (a) the Gini improvement with both absolute values, (b) what risk cohort the GBM is repricing relative to the GLM, (c) the threshold below which you would not recommend deployment and why, and (d) your recommendation given the actual results.

<details>
<summary>Hint for Task 2</summary>

A two-panel matplotlib figure uses `fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))`. Plot the double lift line chart on `ax1` and a bar chart of policy counts on `ax2`. The two panels sharing the x-axis (decile) makes it easy to see where the discrimination is concentrated and how many policies are in each bucket.

</details>

<details>
<summary>Solution - Exercise 5</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

y_binary    = (y_test_f > 0).astype(int)
actual_freq = y_test_f / w_test_f

def gini(y_bin, scores):
    return 2 * roc_auc_score(y_bin, scores) - 1

# Task 1: Naive benchmark
rng_noise   = np.random.default_rng(0)
naive_scores = np.full_like(gbm_freq_pred, actual_freq.mean()) + rng_noise.normal(0, 1e-10, len(gbm_freq_pred))
gini_naive  = gini(y_binary, naive_scores)

print(f"Gini - Naive:    {gini_naive:.3f}  (random guess baseline)")
print(f"Gini - GLM:      {gini_glm:.3f}  (absolute discrimination: {gini_glm - gini_naive:+.3f} over naive)")
print(f"Gini - GBM:      {gini_gbm:.3f}  (absolute discrimination: {gini_gbm - gini_naive:+.3f} over naive)")
print(f"GBM incremental: {gini_gbm - gini_glm:+.3f}")

# Task 2: Double lift chart with policy counts panel
ratio     = gbm_freq_pred / (glm_freq_pred + 1e-10)
n_bins    = 10
bin_edges = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
bin_idx   = np.digitize(ratio, bin_edges[1:-1])

ratio_means, actual_means, n_obs = [], [], []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    ratio_means.append(ratio[mask].mean())
    actual_means.append(actual_freq[mask].mean())
    n_obs.append(int(mask.sum()))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

ax1.plot(range(1, len(ratio_means)+1), actual_means, "o-", color="steelblue", linewidth=2)
ax1.axhline(actual_freq.mean(), linestyle="--", color="grey", label=f"Portfolio mean ({actual_freq.mean():.4f})")
ax1.set_ylabel("Actual observed frequency")
ax1.set_title("Double Lift Chart: CatBoost vs GLM - Motor Frequency")
ax1.legend()

ax2.bar(range(1, len(n_obs)+1), n_obs, color="steelblue", alpha=0.7)
ax2.set_xlabel("Decile (GBM/GLM ratio, low to high)")
ax2.set_ylabel("Number of policies")
ax2.set_title("Policy count per decile")

plt.tight_layout()
plt.show()

# Task 3: Top decile risk profile
top_mask = bin_idx == (n_bins - 1)
df_top   = df_test_final.to_pandas().loc[top_mask]
df_all   = df_test_final.to_pandas()

print(f"\nTop decile ({top_mask.sum():,} policies)")
print(f"  Mean driver age:     {df_top['driver_age'].mean():.1f}  (portfolio: {df_all['driver_age'].mean():.1f})")
print(f"  Mean vehicle group:  {df_top['vehicle_group'].mean():.1f}  (portfolio: {df_all['vehicle_group'].mean():.1f})")
print(f"  Pct driver_age < 25: {(df_top['driver_age'] < 25).mean():.1%}  (portfolio: {(df_all['driver_age'] < 25).mean():.1%})")
print(f"  Actual frequency:    {actual_freq[top_mask].mean():.4f}  (portfolio: {actual_freq.mean():.4f})")
print(f"  Actual / portfolio:  {actual_freq[top_mask].mean() / actual_freq.mean():.2f}x")
```

**Task 4 - pricing committee summary (fill in actual numbers from your run):**

The CatBoost model scores a Gini coefficient of [gini_gbm] compared to [gini_glm] for the GLM, a lift of [gini_gbm - gini_glm] Gini points on a held-out test year (2024). The additional discrimination is concentrated in policies combining young drivers (under 25) with high vehicle groups (above 35), where the GBM is predicting materially higher frequency than the GLM's multiplicative structure allows - and the double lift chart confirms that actual frequencies in the top decile are two to three times the portfolio mean, validating the GBM's additional signal as real rather than spurious. We would not recommend deployment for a Gini lift below 0.03, because below that threshold the improvement is small enough to fall within estimation noise from the single test year, and the governance overhead of maintaining a GBM in production is only justified by a demonstrable and reproducible improvement. Based on the results above, we recommend proceeding to Module 4 (SHAP relativity extraction) for the governance committee review before any production deployment decision.

</details>
