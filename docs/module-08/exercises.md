# Module 8 Exercises: End-to-End Pricing Pipeline

Six exercises. Each builds on the pipeline concepts from the tutorial. Work through them in order — the first four build on each other. Exercises 5 and 6 are independent.

Complete worked solutions are provided inside collapsed `<details>` blocks at the end of each exercise. Read the question and attempt a solution before looking.

---

## Exercise 1: The feature engineering trap

**What this covers:** The NCB encoding incident from Part 1, reproduced as a controlled experiment. You will cause the failure, observe its consequences, and implement the correct fix.

**Setup — run this cell first:**

```python
import polars as pl
import numpy as np
from catboost import CatBoostRegressor, Pool

rng = np.random.default_rng(seed=42)
n   = 20_000

df = pl.DataFrame({
    "ncb_years":     rng.choice([0,1,2,3,4,5], n, p=[0.08,0.07,0.10,0.15,0.20,0.40]).tolist(),
    "vehicle_group": rng.integers(1, 51, n).tolist(),
    "region":        rng.choice(["North","Midlands","London","SouthEast","SouthWest"], n).tolist(),
    "driver_age":    rng.integers(17, 85, n).tolist(),
    "exposure":      np.clip(rng.beta(8, 2, n), 0.05, 1.0).tolist(),
    "accident_year": rng.choice([2022,2023,2024,2025], n).tolist(),
})

# Frequency DGP: NCD drives frequency — NCD 5 = much lower risk
ncb_arr = np.array(df["ncb_years"].to_list())
age_arr = np.array(df["driver_age"].to_list())
freq    = (
    0.08
    * np.array([{0:2.3, 1:1.8, 2:1.4, 3:1.1, 4:0.85, 5:0.62}[x] for x in ncb_arr])
    * (0.95 + 0.003 * np.maximum(0, 25 - age_arr))   # young driver uplift
)
df = df.with_columns(
    pl.Series("claim_count", rng.poisson(freq))
)
print(f"Portfolio freq: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
print(f"NCD 0 freq:  {df.filter(pl.col('ncb_years')==0)['claim_count'].sum() / df.filter(pl.col('ncb_years')==0)['exposure'].sum():.4f}")
print(f"NCD 5 freq:  {df.filter(pl.col('ncb_years')==5)['claim_count'].sum() / df.filter(pl.col('ncb_years')==5)['exposure'].sum():.4f}")
```

**Part A.** The TRAINING pipeline encodes NCB as a string. The SCORING pipeline passes NCB as an integer. Write both, train the model on 2022-2024, score on 2025, and compare the mean predicted frequency for NCD 0 vs NCD 5 under both encodings. What do you observe?

```python
# Training pipeline (correct encoding — CatBoost sees NCB as categorical)
def training_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ncb_years").cast(pl.Utf8).alias("ncb_encoded")
    )

# Scoring pipeline (wrong encoding — CatBoost sees NCB as continuous integer)
def scoring_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ncb_years").cast(pl.Int32).alias("ncb_encoded")
    )

FEATURE_COLS = ["ncb_encoded", "vehicle_group", "driver_age"]
CAT_FEAT_TRAINING = ["ncb_encoded"]
CAT_FEAT_SCORING  = []   # integer — no categorical features

df_pd = df.to_pandas()
train_pd = df_pd[df_pd["accident_year"] < 2025]
test_pd  = df_pd[df_pd["accident_year"] == 2025]

# TODO: complete the training and scoring, compute mean freq by NCD group, compare
```

**Part B.** Implement the fix: a single `apply_features()` function called identically in both pipelines. Verify that the NCD-0 and NCD-5 predictions are now consistent between training-time scoring and scoring-time scoring.

**Part C.** Add a `FeatureSpec` that would have caught the integer-vs-string divergence automatically. Write the validation code and show the error message it produces when applied to integer-encoded NCB data.

<details>
<summary>Solution</summary>

**Part A:**

```python
# Training pipeline
train_feats = training_features(pl.from_pandas(train_pd)).to_pandas()
test_feats_wrong = scoring_features(pl.from_pandas(test_pd)).to_pandas()

train_pool = Pool(
    train_feats[FEATURE_COLS], train_feats["claim_count"].values,
    baseline=np.log(np.clip(train_feats["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEAT_TRAINING,
)
model_trained = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4,
    learning_rate=0.08, random_seed=42, verbose=0,
)
model_trained.fit(train_pool)

# Scoring with wrong encoding (integer, no cat features)
wrong_pool = Pool(
    test_feats_wrong[FEATURE_COLS],
    baseline=np.log(np.clip(test_pd["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEAT_SCORING,  # empty — integer encoding
)
pred_wrong = model_trained.predict(wrong_pool) / test_pd["exposure"].values

# Scoring with correct encoding (string, cat feature)
test_feats_correct = training_features(pl.from_pandas(test_pd)).to_pandas()
correct_pool = Pool(
    test_feats_correct[FEATURE_COLS],
    baseline=np.log(np.clip(test_pd["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEAT_TRAINING,
)
pred_correct = model_trained.predict(correct_pool) / test_pd["exposure"].values

# Compare NCD 0 vs NCD 5 under both encodings
test_ncb = test_pd["ncb_years"].values
for ncd in [0, 5]:
    mask = test_ncb == ncd
    print(f"NCD {ncd} | Wrong encoding: {pred_wrong[mask].mean():.4f} | "
          f"Correct encoding: {pred_correct[mask].mean():.4f}")

# You will see: wrong encoding collapses the NCD 0/5 distinction.
# CatBoost treats integer NCB as a continuous variable and the model
# cannot apply the learned categorical structure to unseen integer inputs.
# The discrimination between NCD 0 and NCD 5 is severely degraded.
```

**Part B:**

```python
def apply_features(df: pl.DataFrame) -> pl.DataFrame:
    """Single function. Call identically at training and scoring time."""
    return df.with_columns(
        pl.col("ncb_years").cast(pl.Utf8).alias("ncb_encoded")
    )

FEATURE_COLS_SHARED = ["ncb_encoded", "vehicle_group", "driver_age"]
CAT_FEATURES_SHARED = ["ncb_encoded"]

# Training
train_shared = apply_features(pl.from_pandas(train_pd)).to_pandas()
train_pool_shared = Pool(
    train_shared[FEATURE_COLS_SHARED], train_shared["claim_count"].values,
    baseline=np.log(np.clip(train_shared["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES_SHARED,
)
model_shared = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4,
    learning_rate=0.08, random_seed=42, verbose=0,
)
model_shared.fit(train_pool_shared)

# Scoring — SAME FUNCTION
test_shared = apply_features(pl.from_pandas(test_pd)).to_pandas()
score_pool_shared = Pool(
    test_shared[FEATURE_COLS_SHARED],
    baseline=np.log(np.clip(test_pd["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES_SHARED,
)
pred_shared = model_shared.predict(score_pool_shared) / test_pd["exposure"].values

for ncd in [0, 5]:
    mask = test_ncb == ncd
    print(f"NCD {ncd} | Shared function: {pred_shared[mask].mean():.4f}")
# NCD 0 should be ~2x NCD 5 — matching the DGP relativities.
```

**Part C:**

```python
class SimpleFeatureSpec:
    def __init__(self):
        self.spec = {}

    def record(self, df, cat_features):
        for col in df.columns:
            s = df[col]
            if col in cat_features or s.dtype == pl.Utf8:
                self.spec[col] = {"dtype": "categorical",
                                  "values": sorted(s.drop_nulls().unique().to_list())}
            else:
                self.spec[col] = {"dtype": "numeric",
                                  "min": float(s.min()), "max": float(s.max())}

    def validate(self, df):
        errors = []
        for col, spec in self.spec.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
                continue
            if spec["dtype"] == "categorical" and df[col].dtype not in (pl.Utf8, pl.Categorical):
                errors.append(
                    f"{col}: expected categorical (Utf8), got {df[col].dtype}. "
                    f"Likely an encoding change between training and scoring. "
                    f"Check apply_features() is called before scoring."
                )
        return errors

spec = SimpleFeatureSpec()
spec.record(
    pl.from_pandas(train_shared[FEATURE_COLS_SHARED]),
    cat_features=CAT_FEATURES_SHARED,
)

# Validate the wrong-encoded scoring data
errors = spec.validate(pl.from_pandas(test_feats_wrong[FEATURE_COLS_SHARED]))
for e in errors:
    print(f"VALIDATION ERROR: {e}")
# Prints: ncb_encoded: expected categorical (Utf8), got Int32.
```

</details>

---

## Exercise 2: Walk-forward CV — understanding the time boundary

**What this covers:** The temporal split principle, and why random splits produce optimistic metrics for insurance data.

**Setup — use the dataset from Exercise 1.**

**Part A.** Implement both splits and compare:

1. **Random split**: 80% train, 20% test, random.
2. **Temporal split**: train on 2022-2024, test on 2025.

Train the same default CatBoost Poisson model on each split and compute Poisson deviance on the respective test sets. Which split produces a lower (better) test deviance? Which is the honest estimate of future performance?

**Part B.** The IBNR problem: the 2025 accident year's claims have only partially developed at the time the data was extracted. Create a version of the fold where 2025 policies are included in training but their claim counts are capped at 50% of their actual value (simulating partial development). Train on 2022-2024 plus this "truncated 2025", test on the real 2025 data. Compare the test deviance to the clean temporal split. What does this tell you about the importance of the IBNR buffer?

<details>
<summary>Solution</summary>

**Part A:**

```python
from sklearn.model_selection import train_test_split

df_pd_ex2 = df.to_pandas()

# Random split
X_all   = df_pd_ex2[["ncb_years", "vehicle_group", "driver_age"]]
y_all   = df_pd_ex2["claim_count"].values
w_all   = df_pd_ex2["exposure"].values

X_tr_r, X_te_r, y_tr_r, y_te_r, w_tr_r, w_te_r = train_test_split(
    X_all, y_all, w_all, test_size=0.2, random_state=42
)
m_rand = CatBoostRegressor(loss_function="Poisson", iterations=200,
                           depth=4, learning_rate=0.08, random_seed=42, verbose=0)
m_rand.fit(Pool(X_tr_r, y_tr_r, baseline=np.log(np.clip(w_tr_r, 1e-6, None))))
pred_r = m_rand.predict(Pool(X_te_r, baseline=np.log(np.clip(w_te_r, 1e-6, None))))

def pd_deviance(y_true, y_pred, w):
    fp = np.clip(y_pred / w, 1e-10, None)
    ft = y_true / w
    return float((2 * w * (np.where(ft>0, ft*np.log(ft/fp), 0.) - (ft-fp))).sum() / w.sum())

dev_random = pd_deviance(y_te_r, pred_r, w_te_r)

# Temporal split
df_tr_t = df_pd_ex2[df_pd_ex2["accident_year"] < 2025]
df_te_t = df_pd_ex2[df_pd_ex2["accident_year"] == 2025]

m_temp = CatBoostRegressor(loss_function="Poisson", iterations=200,
                           depth=4, learning_rate=0.08, random_seed=42, verbose=0)
m_temp.fit(Pool(
    df_tr_t[["ncb_years","vehicle_group","driver_age"]],
    df_tr_t["claim_count"].values,
    baseline=np.log(np.clip(df_tr_t["exposure"].values, 1e-6, None)),
))
pred_t = m_temp.predict(Pool(
    df_te_t[["ncb_years","vehicle_group","driver_age"]],
    baseline=np.log(np.clip(df_te_t["exposure"].values, 1e-6, None)),
))
dev_temporal = pd_deviance(df_te_t["claim_count"].values, pred_t, df_te_t["exposure"].values)

print(f"Random split deviance:   {dev_random:.5f}  (optimistic — future data in training)")
print(f"Temporal split deviance: {dev_temporal:.5f}  (honest — training uses only past data)")
# Random split deviance will be materially lower (better-looking).
# This is the metric inflation from temporal leakage.
```

**Part B:**

```python
df_2025 = df_pd_ex2[df_pd_ex2["accident_year"] == 2025].copy()
df_2025["claim_count"] = (df_2025["claim_count"] * 0.5).astype(int)   # IBNR truncation

df_contaminated = pd.concat([
    df_pd_ex2[df_pd_ex2["accident_year"] < 2025],
    df_2025
])
df_test_clean = df_pd_ex2[df_pd_ex2["accident_year"] == 2025]

m_cont = CatBoostRegressor(loss_function="Poisson", iterations=200,
                           depth=4, learning_rate=0.08, random_seed=42, verbose=0)
m_cont.fit(Pool(
    df_contaminated[["ncb_years","vehicle_group","driver_age"]],
    df_contaminated["claim_count"].values,
    baseline=np.log(np.clip(df_contaminated["exposure"].values, 1e-6, None)),
))
pred_cont = m_cont.predict(Pool(
    df_test_clean[["ncb_years","vehicle_group","driver_age"]],
    baseline=np.log(np.clip(df_test_clean["exposure"].values, 1e-6, None)),
))
dev_contaminated = pd_deviance(df_test_clean["claim_count"].values, pred_cont,
                                df_test_clean["exposure"].values)

print(f"Clean temporal split deviance:  {dev_temporal:.5f}")
print(f"IBNR-contaminated deviance:     {dev_contaminated:.5f}")
# Contaminated deviance will be higher (worse) because the model learned to
# predict lower claim counts in 2025 (half the true value) — it systematically
# underpredicts on the held-out 2025 test set with true counts.
print("\nThe IBNR buffer excludes immature accident periods from training.")
print("Without it, the model learns to predict partially developed claim counts")
print("and will systematically underprice in the prospective period.")
```

</details>

---

## Exercise 3: Optuna tuning — understanding what you are actually optimising

**What this covers:** The relationship between hyperparameter choices and model behaviour. Not just how to run Optuna, but what to do with the results.

**Setup — use the temporal split from Exercise 2.**

**Part A.** Run an Optuna study with 15 trials. After it completes, plot the relationship between `depth` and `learning_rate` across all trials, coloured by deviance value. What pattern do you see? Is depth or learning rate more important for this dataset?

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df_pd_ex3 = df.to_pandas()
df_tr3 = df_pd_ex3[df_pd_ex3["accident_year"] < 2025]
df_te3 = df_pd_ex3[df_pd_ex3["accident_year"] == 2025]

# TODO: define objective function and run study with 15 trials.
# Collect trial.params and trial.value for the plot.
```

**Part B.** The optimal depth from your Optuna study is likely 4 or 5. Train two models: one with depth=3 (underfit), one with depth=8 (potential overfit). Compare their test deviances. Which direction of misspecification costs more?

**Part C.** Add `subsample` and `colsample_bylevel` to the Optuna search space. Do either of these parameters improve the best deviance substantially? Justify your answer in terms of what these parameters do.

<details>
<summary>Solution</summary>

**Part A:**

```python
import polars as pl

FEAT = ["ncb_years", "vehicle_group", "driver_age"]
tr_pool = Pool(df_tr3[FEAT], df_tr3["claim_count"].values,
               baseline=np.log(np.clip(df_tr3["exposure"].values, 1e-6, None)))
te_pool = Pool(df_te3[FEAT], df_te3["claim_count"].values,
               baseline=np.log(np.clip(df_te3["exposure"].values, 1e-6, None)))

def objective_ex3(trial):
    p = {
        "iterations":    trial.suggest_int("iterations", 100, 400),
        "depth":         trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "loss_function": "Poisson", "random_seed": 42, "verbose": 0,
    }
    m = CatBoostRegressor(**p)
    m.fit(tr_pool, eval_set=te_pool)
    pred = m.predict(te_pool)
    return pd_deviance(df_te3["claim_count"].values, pred, df_te3["exposure"].values)

study3 = optuna.create_study(direction="minimize")
study3.optimize(objective_ex3, n_trials=15)

# Collect for analysis
trial_df = pl.DataFrame({
    "depth":         [t.params["depth"] for t in study3.trials],
    "learning_rate": [t.params["learning_rate"] for t in study3.trials],
    "deviance":      [t.value for t in study3.trials],
}).sort("deviance")
print(trial_df)
print(f"\nBest: depth={study3.best_params['depth']}, "
      f"lr={study3.best_params['learning_rate']:.4f}, "
      f"deviance={study3.best_value:.5f}")
# Typically depth=4-5 dominates. Learning rate matters less than depth for
# small datasets — high lr with few trees can match low lr with many trees.
```

**Part B:**

```python
for depth in [3, study3.best_params["depth"], 8]:
    m = CatBoostRegressor(
        loss_function="Poisson", iterations=300, depth=depth,
        learning_rate=study3.best_params["learning_rate"],
        l2_leaf_reg=study3.best_params["l2_leaf_reg"],
        random_seed=42, verbose=0,
    )
    m.fit(tr_pool, eval_set=te_pool)
    pred = m.predict(te_pool)
    dev = pd_deviance(df_te3["claim_count"].values, pred, df_te3["exposure"].values)
    print(f"Depth {depth}: deviance = {dev:.5f}")
# Depth=3 underfits and depth=8 likely overfits slightly.
# For insurance data with 20,000 rows, overfitting is more harmful than underfitting —
# the model memorises noise in thin cells rather than learning generalizable patterns.
```

**Part C:**

```python
def objective_extended(trial):
    p = {
        "iterations":      trial.suggest_int("iterations", 100, 400),
        "depth":           trial.suggest_int("depth", 3, 7),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "loss_function": "Poisson", "random_seed": 42, "verbose": 0,
    }
    m = CatBoostRegressor(**p)
    m.fit(tr_pool, eval_set=te_pool)
    pred = m.predict(te_pool)
    return pd_deviance(df_te3["claim_count"].values, pred, df_te3["exposure"].values)

study_ext = optuna.create_study(direction="minimize")
study_ext.optimize(objective_extended, n_trials=15)
print(f"\nWith subsample + colsample: {study_ext.best_value:.5f}")
print(f"Without:                    {study3.best_value:.5f}")
print(f"Improvement: {study3.best_value - study_ext.best_value:.5f}")
# Typically subsample and colsample_bylevel provide small improvements
# (<0.001 deviance units) on datasets of this size. They matter more for
# large datasets (1M+ rows) where regularisation via column/row subsampling
# becomes important. For personal lines with 20,000-200,000 rows, depth and
# l2_leaf_reg are the dominant regularisation parameters.
```

</details>

---

## Exercise 4: Calibration testing — diagnosing a miscalibrated model

**What this covers:** The Murphy decomposition in practice. You will deliberately miscalibrate a model and diagnose whether the fix is RECALIBRATE or REFIT.

**Setup:**

```python
from insurance_monitoring.calibration import CalibrationChecker, rectify_balance

# Use the temporal split from Exercise 2.
# Train the model on 2022-2024, score on 2025.
# We already have: pred_t (predictions), df_te_t (test set).
```

**Part A.** Apply a systematic underestimation to the predictions: multiply all predictions by 0.88 (simulating 12% underprediction from a year of claims inflation). Run `CalibrationChecker` on the deflated predictions. What verdict do you get? Is it RECALIBRATE or REFIT?

**Part B.** Apply a structural miscalibration instead: reverse the prediction ordering for the top and bottom quartiles (swap the highest-predicted risks with the lowest-predicted risks). Run `CalibrationChecker`. What verdict do you get now? Why is this harder to fix than Part A?

**Part C.** Apply `rectify_balance()` to the Part A predictions (the 12% underestimate). Verify that the balance ratio returns to 1.0, and re-run the auto-calibration test. Does it pass?

<details>
<summary>Solution</summary>

**Part A:**

```python
checker = CalibrationChecker(distribution="poisson")

# Deflated predictions (12% underprediction)
pred_deflated = pred_t * 0.88

report_a = checker.check(
    y=df_te_t["claim_count"].values.astype(float),
    y_hat=pred_deflated,
    exposure=df_te_t["exposure"].values,
    n_bins=10, seed=42,
)
print("Part A: 12% uniform underprediction")
print(report_a.verdict())
print(f"Balance ratio: {report_a.balance_ratio:.4f}")
print(f"Murphy verdict: {report_a.murphy_verdict}")
# Verdict: RECALIBRATE
# Reason: GMCB (global miscalibration) dominates. The error is a uniform
# scalar shift — the model's rank ordering is correct but the level is wrong.
# Multiplying by the balance ratio fixes it.
```

**Part B:**

```python
# Structural miscalibration: swap top and bottom quartile predictions
pred_structural = pred_t.copy()
q25 = np.percentile(pred_structural, 25)
q75 = np.percentile(pred_structural, 75)
top_q  = pred_structural > q75
bot_q  = pred_structural < q25

# Get sorted indices
top_vals = np.sort(pred_structural[top_q])
bot_vals = np.sort(pred_structural[bot_q])

# Swap: high-risk predictions replaced with low-risk predictions, and vice versa
pred_swapped = pred_structural.copy()
pred_swapped[top_q] = bot_vals[np.argsort(np.argsort(pred_structural[top_q]))]
pred_swapped[bot_q] = top_vals[np.argsort(np.argsort(pred_structural[bot_q]))]

report_b = checker.check(
    y=df_te_t["claim_count"].values.astype(float),
    y_hat=pred_swapped,
    exposure=df_te_t["exposure"].values,
    n_bins=10, seed=42,
)
print("\nPart B: Structural miscalibration (quartile swap)")
print(report_b.verdict())
print(f"Balance ratio: {report_b.balance_ratio:.4f}  (near 1.0 — global balance preserved)")
print(f"Murphy verdict: {report_b.murphy_verdict}")
print(f"DSC: {report_b.murphy_dsc_pct:.1f}%")
# Verdict: REFIT (LMCB > GMCB)
# The swap destroys the model's discrimination — the rank ordering is wrong.
# Global balance is preserved (we swapped equal-magnitude values), so GMCB is low.
# LMCB is high because the reliability diagram shows high predictions for
# low-risk policies and low predictions for high-risk policies.
# No scalar correction can fix reversed rank ordering — the model must be refit.
```

**Part C:**

```python
pred_corrected = rectify_balance(
    y_hat=pred_deflated,
    y=df_te_t["claim_count"].values.astype(float),
    exposure=df_te_t["exposure"].values,
    method="multiplicative",
)
correction = float(pred_corrected.mean() / pred_deflated.mean())
print(f"\nPart C: Balance correction factor: {correction:.4f}")

report_c = checker.check(
    y=df_te_t["claim_count"].values.astype(float),
    y_hat=pred_corrected,
    exposure=df_te_t["exposure"].values,
    n_bins=10, seed=42,
)
print(f"Post-correction balance ratio: {report_c.balance_ratio:.4f}  (target: 1.0000)")
print(f"Post-correction auto-cal p:    {report_c.auto_cal_p_value:.4f}")
print(f"Post-correction verdict:       {report_c.murphy_verdict}")
# Balance ratio returns to 1.000. Auto-calibration test also passes because
# the underlying model had good per-decile calibration — the only error was
# the 12% global underprediction, which the multiplicative correction fixes.
```

</details>

---

## Exercise 5: SHAP relativities — comparing to GLM output

**What this covers:** Extracting and interpreting SHAP relativities, and comparing them to a parallel GLM.

**Setup — use the trained frequency model from Exercise 3 (or the tutorial Stage 6 model).**

**Part A.** Compute SHAP relativities for the frequency model using `SHAPRelativities`. Print the NCB deficit factor table (ncb_deficit levels 0-5 with relativities and 95% CIs). Do the relativities match the DGP used to generate the data?

**Part B.** Fit a Poisson GLM using `statsmodels` on the same training data with NCB deficit and driver age as features. Extract the GLM `exp(beta)` factor table and compare it to the SHAP relativities. For which features do they agree most closely? For which do they diverge?

**Part C.** The SHAP relativities have confidence intervals. Identify the level with the widest relative CI (upper_ci / relativity). What explains the width? How would you present this to the underwriting committee?

<details>
<summary>Solution</summary>

**Part A:**

```python
from shap_relativities import SHAPRelativities
import polars as pl

df_pd_ex5 = df.to_pandas()
df_tr5 = df_pd_ex5[df_pd_ex5["accident_year"] < 2025].copy()

FEAT5 = ["ncb_years", "vehicle_group", "driver_age"]
tr5_pool = Pool(df_tr5[FEAT5], df_tr5["claim_count"].values,
                baseline=np.log(np.clip(df_tr5["exposure"].values, 1e-6, None)))
model5 = CatBoostRegressor(loss_function="Poisson", iterations=300, depth=4,
                            learning_rate=0.08, random_seed=42, verbose=0)
model5.fit(tr5_pool)

df_te5 = df_pd_ex5[df_pd_ex5["accident_year"] == 2025]
X_te5  = pl.from_pandas(df_te5[FEAT5].reset_index(drop=True))
exp_te5 = pl.Series("exposure", df_te5["exposure"].tolist())

sr5 = SHAPRelativities(model=model5, X=X_te5, exposure=exp_te5)
sr5.fit()
rels5 = sr5.extract_relativities(normalise_to="mean")

ncb_rels = rels5.filter(pl.col("feature") == "ncb_years").sort("level")
print("NCB relativities (SHAP):")
print(ncb_rels.select(["level", "relativity", "lower_ci", "upper_ci", "n_obs"]))

# DGP relativities: {0: 2.3, 1: 1.8, 2: 1.4, 3: 1.1, 4: 0.85, 5: 0.62}
# normalised to mean: divide by (2.3+1.8+1.4+1.1+0.85+0.62)/6 = 1.345
# NCD 0 mean-normalised: 2.3/1.345 = 1.71
# NCD 5 mean-normalised: 0.62/1.345 = 0.46
print("\nDGP relativities (mean-normalised):")
raw = {0:2.3, 1:1.8, 2:1.4, 3:1.1, 4:0.85, 5:0.62}
mean_raw = sum(raw.values()) / len(raw)
for k, v in raw.items():
    print(f"  NCB {k}: {v/mean_raw:.3f}")
```

**Part B:**

```python
import statsmodels.formula.api as smf

# Add log_exposure as offset
df_tr5_sm = df_tr5.copy()
df_tr5_sm["log_exposure"] = np.log(np.clip(df_tr5_sm["exposure"].values, 1e-6, None))
df_tr5_sm["ncb_cat"] = df_tr5_sm["ncb_years"].astype("category")

glm_model = smf.glm(
    formula="claim_count ~ C(ncb_cat) + vehicle_group + driver_age",
    data=df_tr5_sm,
    family=smf.families.Poisson(),
    offset=df_tr5_sm["log_exposure"],
).fit()

print("GLM NCB factor relativities:")
for param, val in glm_model.params.items():
    if "ncb_cat" in param:
        level = int(param.split("[T.")[1].rstrip("]"))
        print(f"  NCB {level}: exp(beta) = {np.exp(val):.3f}")

print("\nSHAP vs GLM comparison:")
print("Feature 'ncb_years': generally close (monotone, similar magnitude)")
print("Feature 'driver_age': SHAP captures non-linear effects; GLM assumes linear")
print("Divergence in driver_age is the interaction that GLM cannot express without")
print("manual quadratic terms. SHAP picks it up automatically.")
```

**Part C:**

```python
# Add relative CI width
rels_with_width = rels5.with_columns(
    ((pl.col("upper_ci") - pl.col("lower_ci")) / pl.col("relativity")).alias("rel_ci_width")
).sort("rel_ci_width", descending=True)

print("Widest relative CI (most uncertain levels):")
print(rels_with_width.head(5).select(["feature", "level", "relativity",
                                       "lower_ci", "upper_ci", "n_obs", "rel_ci_width"]))

# Typically the widest CI is for a thin category (e.g., Scotland, which is 10% of
# the portfolio). Presenting to the underwriting committee:
# "The Scotland factor has a CI of ±18% around its point estimate of 0.87x.
#  This reflects 2,100 policies in the test year — about 1/4 the exposure of the
#  Midlands. We recommend applying a credibility weight to the Scotland relativity
#  blending it towards the portfolio mean before deployment, as in Module 6."
```

</details>

---

## Exercise 6: Pipeline resilience — what happens when a stage fails?

**What this covers:** Designing a pipeline that fails explicitly rather than silently, and recovering from stage failures.

**Context.** A production pipeline that silently produces wrong outputs is worse than one that crashes with a clear error message. This exercise practises adding explicit failure guards to each stage.

**Part A.** The severity model in Stage 6 predicts negative values for 3 policies when run on unusual data (a known CatBoost edge case with Tweedie loss on very small datasets). Write the assertion code that catches this before the pure premium computation, and the recovery strategy: clip predictions to the 1st percentile of positive predictions.

```python
# Simulate the edge case: inject 3 negative severity predictions
sev_pred_with_negatives = np.where(
    np.random.default_rng(0).random(1000) < 0.003,  # 3 negatives per 1000
    -np.abs(np.random.default_rng(1).normal(100, 50, 1000)),
    np.random.default_rng(2).gamma(2, 1500, 1000),
)

# TODO: Write assertion + recovery code
```

**Part B.** The Optuna study crashes on trial 7 because the cluster ran out of memory during a high-depth, high-iteration trial. Write a try-except wrapper around the objective function that returns the worst possible value (maximum deviance seen so far) when a trial fails, so Optuna can continue without interruption.

**Part C.** The `CalibrationChecker` returns `REFIT` for the frequency model. Write the code that stops the pipeline at Stage 8.5 with a meaningful error message, and writes a partial audit record to Delta that records the Murphy verdict and the run date, so the investigation can be traced.

<details>
<summary>Solution</summary>

**Part A:**

```python
def validate_severity_predictions(
    sev_pred: np.ndarray,
    fallback: str = "clip_to_p01",
) -> np.ndarray:
    """
    Validate and optionally repair severity predictions.
    Raises if non-finite values are present. Clips negatives if fallback="clip_to_p01".
    """
    n_nonfinite = (~np.isfinite(sev_pred)).sum()
    if n_nonfinite > 0:
        raise ValueError(
            f"Severity model produced {n_nonfinite} non-finite predictions. "
            f"This indicates a numerical instability in the Tweedie model. "
            f"Check for extreme values in the input features."
        )

    n_negative = (sev_pred <= 0).sum()
    if n_negative > 0:
        if fallback == "clip_to_p01":
            p01 = np.percentile(sev_pred[sev_pred > 0], 1)
            sev_pred_clean = np.where(sev_pred <= 0, p01, sev_pred)
            print(f"WARNING: {n_negative} non-positive severity predictions detected.")
            print(f"  Clipped to P1 of positive predictions: £{p01:,.2f}")
            print(f"  This is an edge case — investigate the model if it occurs frequently.")
            return sev_pred_clean
        else:
            raise ValueError(
                f"Severity model produced {n_negative} non-positive predictions. "
                f"This is a known Tweedie edge case. Set fallback='clip_to_p01' "
                f"to auto-correct, or investigate the model configuration."
            )
    return sev_pred

# Apply
sev_clean = validate_severity_predictions(sev_pred_with_negatives)
assert (sev_clean > 0).all(), "Validation failed to fix non-positive predictions"
print(f"All {len(sev_clean):,} predictions are now positive.")
```

**Part B:**

```python
worst_deviance = [0.15]   # mutable container so inner function can update it

def robust_objective(trial: optuna.Trial) -> float:
    """
    Optuna objective with graceful trial failure handling.
    Returns worst_deviance if the trial errors, so the study continues.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 600),
        "depth":         trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "loss_function": "Poisson", "random_seed": 42, "verbose": 0,
    }
    try:
        m = CatBoostRegressor(**params)
        m.fit(tr5_pool, eval_set=Pool(
            df_te5[FEAT5], df_te5["claim_count"].values,
            baseline=np.log(np.clip(df_te5["exposure"].values, 1e-6, None)),
        ))
        pred = m.predict(Pool(
            df_te5[FEAT5],
            baseline=np.log(np.clip(df_te5["exposure"].values, 1e-6, None)),
        ))
        dev = pd_deviance(df_te5["claim_count"].values, pred, df_te5["exposure"].values)
        worst_deviance[0] = max(worst_deviance[0], dev)
        return dev
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}. Returning worst deviance.")
        return worst_deviance[0] * 1.05   # slightly worse than worst seen

robust_study = optuna.create_study(direction="minimize")
robust_study.optimize(robust_objective, n_trials=10)
print(f"Best deviance (robust study): {robust_study.best_value:.5f}")
```

**Part C:**

```python
def stop_pipeline_with_partial_audit(
    murphy_verdict: str,
    cal_report,
    audit_partial: dict,
    tables: dict,
) -> None:
    """
    Write a partial audit record and raise RuntimeError when the pipeline
    encounters a REFIT verdict.
    """
    partial_record = {
        **audit_partial,
        "pipeline_stopped":      True,
        "stop_reason":           f"CalibrationChecker verdict: {murphy_verdict}",
        "cal_balance_ratio":     round(float(cal_report.balance_ratio), 4),
        "cal_murphy_verdict":    murphy_verdict,
        "cal_dsc_pct":           round(float(cal_report.murphy_dsc_pct), 2),
        "cal_mcb_pct":           round(float(cal_report.murphy_mcb_pct), 2),
        "freq_run_id":           audit_partial.get("freq_run_id", "NOT_SET"),
        "pipeline_notes":        "PIPELINE STOPPED AT STAGE 8.5 — MODEL REQUIRES REFIT",
    }

    spark.createDataFrame([partial_record]) \
         .write.format("delta").mode("append") \
         .saveAsTable(tables["pipeline_audit"])

    raise RuntimeError(
        f"\n{'='*60}\n"
        f"PIPELINE STOPPED: Frequency model requires REFIT.\n"
        f"{'='*60}\n"
        f"Murphy verdict: {murphy_verdict}\n"
        f"Discrimination: {cal_report.murphy_dsc_pct:.1f}% of UNC\n"
        f"Miscalibration: {cal_report.murphy_mcb_pct:.1f}% of UNC\n"
        f"\nNext steps:\n"
        f"  1. Examine the per-decile reliability table for structural patterns\n"
        f"  2. Check for feature distribution shift between training and test year\n"
        f"  3. Check for IBNR contamination in the training data\n"
        f"  4. Consider isotonic recalibration on a large holdout set\n"
        f"  5. Partial audit record written to {tables['pipeline_audit']}\n"
    )

# Usage in Stage 8.5:
if cal_murphy_verdict == "REFIT":
    stop_pipeline_with_partial_audit(
        murphy_verdict=cal_murphy_verdict,
        cal_report=cal_report,
        audit_partial={
            "run_date": RUN_DATE,
            "raw_table_version": int(raw_version),
            "freq_run_id": freq_run_id,
        },
        tables=TABLES,
    )
```

</details>
