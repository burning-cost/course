# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: GBMs for Insurance Pricing - CatBoost Frequency/Severity
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC This notebook trains CatBoost frequency and severity models on a synthetic UK motor
# MAGIC portfolio, evaluates them against the GLM from Module 2, and registers them in MLflow.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads the motor portfolio from Unity Catalog (or generates it)
# MAGIC 2. Fits a Poisson frequency model with walk-forward CV and IBNR buffer
# MAGIC 3. Fits a Gamma-equivalent severity model on claims-only data
# MAGIC 4. Tunes hyperparameters with Optuna (40 trials)
# MAGIC 5. Compares GBM vs GLM on Gini coefficient and calibration
# MAGIC 6. Produces a double lift chart for the pricing committee
# MAGIC 7. Registers the challenger model in the MLflow model registry
# MAGIC
# MAGIC **Runtime:** 30-45 minutes on a 4-core cluster (Standard_DS3_v2).
# MAGIC
# MAGIC **Prerequisites:** Module 1 and Module 2 notebooks completed.
# MAGIC Unity Catalog schemas `pricing.motor` and `pricing.governance` must exist.

# COMMAND ----------

# Install libraries
# Use uv in Databricks shell cells for reproducible installs
%pip install catboost insurance-cv optuna mlflow polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date

import numpy as np
import polars as pl
import catboost
from catboost import CatBoostRegressor, Pool
import optuna
import mlflow
import mlflow.catboost
from mlflow import MlflowClient
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CatBoost version: {catboost.__version__}")
print(f"MLflow version:   {mlflow.__version__}")
print(f"Today:            {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data from Unity Catalog
# MAGIC
# MAGIC We read the motor portfolio written by Module 1's notebook.
# MAGIC If you skipped Module 1, generate synthetic data with the
# MAGIC `generate_motor_portfolio()` function below.
# MAGIC
# MAGIC Key columns we need:
# MAGIC - `policy_id`, `accident_year`: identifiers
# MAGIC - `exposure_years`: earned policy-years (offset for the frequency model)
# MAGIC - `claim_count`: Poisson target for frequency
# MAGIC - `incurred`: total incurred cost (used to derive average severity)
# MAGIC - Rating factors: `area`, `ncd_years`, `vehicle_group`, `driver_age`, `conviction_points`

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"
TABLE   = "claims_exposure"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

try:
    df = pl.from_pandas(spark.table(FULL_TABLE).toPandas())
    # NOTE: For larger datasets (2M+ rows), filter to your modelling sample before
    # calling toPandas() — loading a full production portfolio to the driver will OOM.
    print(f"Loaded from Delta: {len(df):,} rows")
    print(f"Accident years: {sorted(df['accident_year'].unique().to_list())}")
except Exception as e:
    print(f"Could not load table: {e}")
    print("Generating synthetic portfolio (100,000 policies)...")

    # Fallback: generate synthetic data
    # (Copy generate_motor_portfolio from Module 1 notebook if needed)
    from datetime import timedelta

    def generate_motor_portfolio(n_policies=100_000, accident_years=(2019,2020,2021,2022,2023,2024), seed=42):
        rng = np.random.default_rng(seed)
        rows_per_year = n_policies // len(accident_years)
        records = []
        for ay in accident_years:
            n = rows_per_year
            area_bands = rng.choice(["A","B","C","D","E","F"], n, p=[0.15,0.25,0.25,0.20,0.10,0.05])
            ncd_years  = rng.choice([0,1,2,3,4,5], n, p=[0.08,0.07,0.10,0.15,0.20,0.40])
            vehicle_group = rng.integers(1, 51, n)
            driver_age = rng.integers(17, 85, n)
            conviction_points = rng.choice([0,0,0,0,0,3,6,9], n)
            exposure = rng.uniform(0.25, 1.0, n)
            has_conv = (conviction_points > 0).astype(int)
            area_eff = {"A":0.0,"B":0.10,"C":0.20,"D":0.35,"E":0.50,"F":0.65}
            log_mu = (
                -3.0
                + np.array([area_eff[a] for a in area_bands])
                + (-0.12) * ncd_years
                + 0.45 * has_conv
                + 0.010 * (vehicle_group - 25)
                + np.where(driver_age < 25, 0.55 * (25 - driver_age) / 8, 0.0)
                + np.where(driver_age > 70, 0.30 * (driver_age - 70) / 14, 0.0)
            )
            claim_count = rng.poisson(np.exp(log_mu) * exposure).astype(int)
            log_sev = 7.8 + 0.008 * (vehicle_group - 25) + 0.30 * has_conv
            sev = rng.gamma(2.0, np.exp(log_sev) / 2.0)
            incurred = np.where(claim_count > 0, claim_count * sev, 0.0)
            for i in range(n):
                records.append({
                    "policy_id": f"POL{ay}{i:06d}",
                    "accident_year": ay,
                    "exposure_years": round(float(exposure[i]), 4),
                    "claim_count": int(claim_count[i]),
                    "incurred": round(float(incurred[i]), 2),
                    "area": area_bands[i],
                    "ncd_years": int(ncd_years[i]),
                    "vehicle_group": int(vehicle_group[i]),
                    "driver_age": int(driver_age[i]),
                    "conviction_points": int(conviction_points[i]),
                })
        return pl.DataFrame(records)

    df = generate_motor_portfolio()
    print(f"Generated: {len(df):,} rows")

# Add policy_year alias for insurance-cv
df = df.with_columns(pl.col("accident_year").alias("policy_year"))

print(f"\nPortfolio summary:")
print(f"  Rows: {len(df):,}")
print(f"  Exposure: {df['exposure_years'].sum():.0f} earned years")
print(f"  Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum()/df['exposure_years'].sum():.4f} per year)")
print(f"  Total incurred: {df['incurred'].sum()/1e6:.1f}m")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature setup
# MAGIC
# MAGIC Key design decision: pass area as a categorical feature to CatBoost.
# MAGIC CatBoost's native categorical handling uses ordered target statistics - essentially
# MAGIC a Bayesian mean encoding computed on a permuted training set that prevents leakage.
# MAGIC This is better than one-hot encoding for factors with 5-10 levels because it preserves
# MAGIC the ordinality information the tree splits can exploit.
# MAGIC
# MAGIC conviction_points is passed as categorical despite being numeric, because the
# MAGIC values (0, 3, 6, 9) represent ordered categories, not a linear quantity.
# MAGIC CatBoost handles this correctly if we declare it categorical.

# COMMAND ----------

CONTINUOUS_FEATURES = ["driver_age", "vehicle_group", "ncd_years"]
CAT_FEATURES        = ["area", "conviction_points"]
FEATURES            = CONTINUOUS_FEATURES + CAT_FEATURES

FREQ_TARGET  = "claim_count"
SEV_TARGET   = "avg_severity"
EXPOSURE_COL = "exposure_years"

# Derive average severity for the severity model.
# Zero-claim rows receive null (not zero) via otherwise(None), which avoids a
# division-by-zero. The null rows are excluded before the severity model trains
# (see the filter(pl.col("claim_count") > 0) calls in section 7 below).
df = df.with_columns(
    pl.when(pl.col("claim_count") > 0)
    .then(pl.col("incurred") / pl.col("claim_count"))
    .otherwise(None)
    .alias("avg_severity")
)

print(f"Features: {FEATURES}")
print(f"Categorical: {CAT_FEATURES}")
print(f"\nSeverity statistics (claims-only):")
claims_only = df.filter(pl.col("claim_count") > 0)
print(f"  n policies with claims: {len(claims_only):,} ({100*len(claims_only)/len(df):.1f}%)")
print(f"  Mean severity: {claims_only['avg_severity'].mean():,.0f}")
print(f"  95th pctile:   {claims_only['avg_severity'].quantile(0.95):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Walk-forward cross-validation
# MAGIC
# MAGIC This is the section that most pricing teams get wrong. Random train/test splits are
# MAGIC wrong for insurance data because they mix policy years in both sets. A model that
# MAGIC trains on 80% of 2019-2024 data and validates on the remaining 20% of 2019-2024 data
# MAGIC is not being asked to predict the future - it is filling in gaps in a history it has
# MAGIC already seen. The deviance looks good because the test data shares the same temporal
# MAGIC autocorrelation patterns as the training data.
# MAGIC
# MAGIC Walk-forward CV replicates the actual deployment scenario: train on earlier years,
# MAGIC predict on later years. The IBNR buffer (1 year) excludes the most recent training
# MAGIC year from each fold because its claims are not yet fully reported.

# COMMAND ----------

try:
    from insurance_cv import WalkForwardCV

    cv = WalkForwardCV(
        year_col="policy_year",
        min_train_years=2,
        ibnr_buffer_years=1,
        n_splits=3,
    )

    # cv.split() works on a pandas DataFrame (or numpy array with same length)
    df_pd = df.to_pandas()
    folds = list(cv.split(df_pd))

    print(f"Walk-forward folds: {len(folds)}")
    for i, (train_idx, val_idx) in enumerate(folds):
        train_years = sorted(df_pd.iloc[train_idx]["policy_year"].unique().tolist())
        val_years   = sorted(df_pd.iloc[val_idx]["policy_year"].unique().tolist())
        print(f"  Fold {i+1}: train={train_years}, validate={val_years}")

except ImportError:
    print("insurance-cv not installed. Using manual walk-forward folds.")
    df_pd = df.to_pandas()

    # Manual walk-forward: 3 folds
    all_years = sorted(df_pd["policy_year"].unique().tolist())
    folds = []
    for cutoff in all_years[2:5]:  # use first 3 valid cutoffs
        # IBNR buffer: exclude the year before cutoff from training
        train_idx = df_pd[df_pd["policy_year"] <= cutoff - 2].index.to_numpy()
        val_idx   = df_pd[df_pd["policy_year"] == cutoff].index.to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
            print(f"  Fold: train<={cutoff-2}, validate={cutoff}")

    print(f"\n{len(folds)} folds created manually")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Poisson deviance helper and CV loop
# MAGIC
# MAGIC The Poisson deviance is the correct loss function for count data. Do not use RMSE
# MAGIC for frequency model evaluation - it treats a miss of 2 claims on a rare-claims
# MAGIC policy the same as a miss of 2 claims on a high-frequency policy, which is wrong.
# MAGIC
# MAGIC Important: CatBoost's `baseline` takes log(exposure), not exposure. Passing the
# MAGIC raw exposure value (not log-transformed) is the single most common CatBoost bug
# MAGIC for insurance models. It silently produces wrong predictions.

# COMMAND ----------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """
    Scaled Poisson deviance per unit exposure.

    y_true and y_pred are on the count scale (not frequency).
    Dividing by exposure converts to a per-unit-time deviance that
    is comparable across portfolios with different exposure distributions.
    """
    freq_pred = np.clip(y_pred / exposure, 1e-10, None)
    freq_true = y_true / exposure
    deviance = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true / freq_pred), 0.0)
        - (freq_true - freq_pred)
    )
    return float(deviance.sum() / exposure.sum())


cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_train = df_pd.iloc[train_idx]
    df_val   = df_pd.iloc[val_idx]

    X_train = df_train[FEATURES]
    y_train = df_train[FREQ_TARGET].values
    w_train = df_train[EXPOSURE_COL].values

    X_val   = df_val[FEATURES]
    y_val   = df_val[FREQ_TARGET].values
    w_val   = df_val[EXPOSURE_COL].values

    # log(exposure) enters as baseline: the model predicts lambda * exposure
    # where lambda is the per-unit-time frequency
    train_pool = Pool(X_train, y_train, baseline=np.log(w_train), cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val,   baseline=np.log(w_val),   cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        verbose=0,
    )
    model.fit(train_pool, eval_set=val_pool)

    y_pred = model.predict(val_pool)
    fold_dev = poisson_deviance(y_val, y_pred, w_val)
    cv_deviances.append(fold_dev)
    print(f"Fold {fold_idx+1}: Poisson deviance = {fold_dev:.4f}  "
          f"(val years: {sorted(df_val['policy_year'].unique().tolist())})")

print(f"\nMean CV deviance: {np.mean(cv_deviances):.4f} (+/- {np.std(cv_deviances):.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Hyperparameter tuning with Optuna
# MAGIC
# MAGIC We tune on the last fold only. Tuning on all folds is more rigorous but 3x slower.
# MAGIC For a 100,000-policy book, 40 trials on a single fold takes 15-20 minutes.
# MAGIC
# MAGIC The three parameters that matter most for insurance frequency models:
# MAGIC - depth: controls complexity of captured interactions (4-6 is usually right)
# MAGIC - learning_rate: lower = slower convergence but better generalisation
# MAGIC - l2_leaf_reg: regularisation strength; increase if train deviance >> CV deviance
# MAGIC
# MAGIC iterations is also tunable, but it interacts with learning_rate. With early stopping
# MAGIC you can simply set iterations high and let the model find the right number.

# COMMAND ----------

train_idx_tune, val_idx_tune = folds[-1]

df_train_tune = df_pd.iloc[train_idx_tune]
df_val_tune   = df_pd.iloc[val_idx_tune]

X_train_t = df_train_tune[FEATURES]
y_train_t = df_train_tune[FREQ_TARGET].values
w_train_t = df_train_tune[EXPOSURE_COL].values

X_val_t   = df_val_tune[FEATURES]
y_val_t   = df_val_tune[FREQ_TARGET].values
w_val_t   = df_val_tune[EXPOSURE_COL].values

# Pool objects are constructed ONCE here, outside the objective function.
# Constructing them inside the objective would repeat feature hashing and
# categorical encoding on every trial — 40x wasteful on a 100,000-row dataset.
train_pool_t = Pool(X_train_t, y_train_t, baseline=np.log(w_train_t), cat_features=CAT_FEATURES)
val_pool_t   = Pool(X_val_t,   y_val_t,   baseline=np.log(w_val_t),   cat_features=CAT_FEATURES)


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


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest hyperparameters (40 trials):")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest CV deviance: {study.best_value:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train final frequency model and log to MLflow
# MAGIC
# MAGIC The final model trains on all years except the held-out test year.
# MAGIC The test year is the most recent year - the closest approximation to
# MAGIC "predicting next year from this year's data."
# MAGIC
# MAGIC We log the model, all parameters, and key metrics to MLflow.
# MAGIC This is the FCA audit trail for the model development step:
# MAGIC everything needed to reproduce this run is recorded in the MLflow run.

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/motor-gbm-module03")

# Train/test split: test = most recent year
max_year = df["accident_year"].max()
df_train_final = df.filter(pl.col("accident_year") < max_year)
df_test        = df.filter(pl.col("accident_year") == max_year)

X_train_f = df_train_final[FEATURES].to_pandas()
y_train_f = df_train_final[FREQ_TARGET].to_numpy()
w_train_f = df_train_final[EXPOSURE_COL].to_numpy()

X_test_f  = df_test[FEATURES].to_pandas()
y_test_f  = df_test[FREQ_TARGET].to_numpy()
w_test_f  = df_test[EXPOSURE_COL].to_numpy()

final_train_pool = Pool(X_train_f, y_train_f, baseline=np.log(w_train_f), cat_features=CAT_FEATURES)
final_test_pool  = Pool(X_test_f,  y_test_f,  baseline=np.log(w_test_f),  cat_features=CAT_FEATURES)

freq_params = {
    **best_params,
    "loss_function": "Poisson",
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="freq_catboost_tuned") as run_freq:
    mlflow.log_params(freq_params)
    mlflow.log_param("model_type",    "catboost_frequency")
    mlflow.log_param("cv_strategy",   "walk_forward_ibnr1")
    mlflow.log_param("n_cv_folds",    len(folds))
    mlflow.log_param("features",      json.dumps(FEATURES))
    mlflow.log_param("cat_features",  json.dumps(CAT_FEATURES))
    mlflow.log_param("train_years",   str(sorted(df_train_final["accident_year"].unique().to_list())))
    mlflow.log_param("test_year",     str(max_year))
    mlflow.log_param("run_date",      str(date.today()))

    freq_model = CatBoostRegressor(**freq_params)
    freq_model.fit(final_train_pool, eval_set=final_test_pool)

    y_pred_freq = freq_model.predict(final_test_pool)
    test_dev    = poisson_deviance(y_test_f, y_pred_freq, w_test_f)

    mlflow.log_metric("test_poisson_deviance",  test_dev)
    mlflow.log_metric("mean_cv_deviance",       np.mean(cv_deviances))
    mlflow.log_metric("cv_deviance_std",        np.std(cv_deviances))

    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = run_freq.info.run_id

print(f"Frequency model: run_id={freq_run_id}")
print(f"Test Poisson deviance: {test_dev:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Severity model - Gamma-equivalent Tweedie
# MAGIC
# MAGIC The severity model predicts average cost per claim conditional on having a claim.
# MAGIC Only policies with claim_count > 0 enter the severity model.
# MAGIC
# MAGIC No exposure offset for severity: the cost of a claim does not depend on how long
# MAGIC the policy was in force. The Poisson frequency model accounts for exposure.
# MAGIC
# MAGIC CatBoost does not have a dedicated Gamma loss function by that name.
# MAGIC "Tweedie:variance_power=2" is mathematically equivalent to the Gamma log-link model.
# MAGIC Power=1 is Poisson. Power between 1 and 2 is compound Poisson-Gamma. Power=2 is Gamma.

# COMMAND ----------

df_train_sev = df_train_final.filter(pl.col("claim_count") > 0)
df_test_sev  = df_test.filter(pl.col("claim_count") > 0)

print(f"Severity training set: {len(df_train_sev):,} policies with claims")
print(f"Severity test set:     {len(df_test_sev):,} policies with claims")

X_train_s = df_train_sev[FEATURES].to_pandas()
y_train_s = df_train_sev[SEV_TARGET].to_numpy()

X_test_s  = df_test_sev[FEATURES].to_pandas()
y_test_s  = df_test_sev[SEV_TARGET].to_numpy()

sev_train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

# NOTE: In production, tune severity hyperparameters separately. The optimal depth for
# a Poisson frequency model on 100,000 policies is not necessarily optimal for a Gamma
# severity model on the 7-10% of policies with claims. Shared hyperparameters are a
# tutorial simplification.
sev_params = {
    **best_params,
    "loss_function": "Tweedie:variance_power=2",   # Gamma equivalent
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="sev_catboost_tuned") as run_sev:
    mlflow.log_params(sev_params)
    mlflow.log_param("model_type",       "catboost_severity")
    mlflow.log_param("n_claims_train",   len(df_train_sev))
    mlflow.log_param("n_claims_test",    len(df_test_sev))
    mlflow.log_param("severity_target",  "avg_cost_per_claim")

    sev_model = CatBoostRegressor(**sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    y_pred_sev = sev_model.predict(sev_test_pool)
    rmse_sev   = float(np.sqrt(np.mean((y_test_s - y_pred_sev)**2)))
    mae_sev    = float(np.mean(np.abs(y_test_s - y_pred_sev)))
    mean_err   = float(np.mean(y_pred_sev) / np.mean(y_test_s) - 1)

    mlflow.log_metric("test_rmse",          rmse_sev)
    mlflow.log_metric("test_mae",           mae_sev)
    mlflow.log_metric("mean_severity_bias", mean_err)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = run_sev.info.run_id

print(f"\nSeverity model: run_id={sev_run_id}")
print(f"RMSE: {rmse_sev:,.0f}  MAE: {mae_sev:,.0f}")
print(f"Mean severity bias: {mean_err*100:+.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare GBM to GLM baseline
# MAGIC
# MAGIC The comparison that determines what we do with the GBM. Two diagnostics:
# MAGIC
# MAGIC 1. Gini coefficient: how well does each model discriminate between high-risk
# MAGIC    and low-risk policies? GBM should be higher.
# MAGIC
# MAGIC 2. Double lift chart: of the policies the GBM calls high-risk that the GLM
# MAGIC    does not, are they actually high-risk? A flat chart means the GBM is
# MAGIC    not finding anything the GLM cannot.

# COMMAND ----------

import statsmodels.formula.api as smf
import statsmodels.api as sm

# Fit a Poisson GLM on the same training data for comparison
df_glm_train = df_train_final.to_pandas()
df_glm_train["log_exposure"] = np.log(df_glm_train[EXPOSURE_COL].clip(lower=1e-6))

glm_formula = (
    "claim_count ~ C(area) + C(ncd_years, Treatment(0)) + "
    "vehicle_group + driver_age + conviction_points"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_model = smf.glm(
        formula=glm_formula,
        data=df_glm_train,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_glm_train["log_exposure"],
    ).fit()

df_glm_test = df_test.to_pandas()
df_glm_test["log_exposure"] = np.log(df_glm_test[EXPOSURE_COL].clip(lower=1e-6))
glm_pred_counts = glm_model.predict(df_glm_test, exposure=df_glm_test[EXPOSURE_COL])

print(f"GLM converged: {glm_model.converged}")
print(f"GLM parameters: {len(glm_model.params)}")

# COMMAND ----------

def gini_coefficient(y_counts: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Gini coefficient using binary AUC formulation.
    y_counts: actual claim counts
    y_scores: predicted frequencies (score, not count)
    """
    y_binary = (y_counts > 0).astype(int)
    auc = roc_auc_score(y_binary, y_scores)
    return float(2 * auc - 1)


gbm_freq_pred = y_pred_freq / w_test_f
glm_freq_pred = glm_pred_counts.values / w_test_f

gini_gbm = gini_coefficient(y_test_f, gbm_freq_pred)
gini_glm = gini_coefficient(y_test_f, glm_freq_pred)

print(f"Gini coefficient - GBM: {gini_gbm:.3f}")
print(f"Gini coefficient - GLM: {gini_glm:.3f}")
print(f"Lift:                   {gini_gbm - gini_glm:+.3f} ({(gini_gbm/gini_glm - 1)*100:+.1f}%)")

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_metric("gini_gbm",  gini_gbm)
    mlflow.log_metric("gini_glm",  gini_glm)
    mlflow.log_metric("gini_lift", gini_gbm - gini_glm)

# COMMAND ----------

# Double lift chart
ratio = gbm_freq_pred / (glm_freq_pred + 1e-10)
actual_freq = y_test_f / w_test_f

n_bins = 10
bin_edges = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
bin_idx   = np.digitize(ratio, bin_edges[1:-1])

ratio_means, actual_means, n_obs = [], [], []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    ratio_means.append(ratio[mask].mean())
    actual_means.append(actual_freq[mask].mean())
    n_obs.append(mask.sum())

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ratio_means, actual_means, "o-", color="steelblue", linewidth=2, label="Actual frequency")
ax.axhline(actual_freq.mean(), linestyle="--", color="grey",
           label=f"Portfolio mean ({actual_freq.mean():.4f})")

for x, y, n in zip(ratio_means, actual_means, n_obs):
    ax.annotate(f"n={n:,}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")

ax.set_xlabel("GBM / GLM predicted frequency ratio (by decile)")
ax.set_ylabel("Actual observed frequency")
ax.set_title("Double Lift Chart: CatBoost vs GLM - Motor Frequency")
ax.legend()
plt.tight_layout()

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_figure(fig, "double_lift_gbm_vs_glm.png")

plt.show()
print("A positively-sloping curve confirms the GBM identifies genuine additional risk signals.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature importance
# MAGIC
# MAGIC CatBoost's default importance metric is PredictionValuesChange (the average change
# MAGIC in prediction value when a feature is varied across the training data), normalised
# MAGIC to sum to 100. This is NOT the mean absolute SHAP value. To get SHAP-based
# MAGIC importances, pass type='ShapValues' explicitly (covered in Module 4).
# MAGIC
# MAGIC This tells you which features the model relies on, but not how they influence
# MAGIC predictions for individual policies. Module 4 covers SHAP values in full for
# MAGIC relativity extraction.

# COMMAND ----------

importances = freq_model.get_feature_importance(type="FeatureImportance")

imp_df = (
    pl.DataFrame({"feature": FEATURES, "importance": importances.tolist()})
    .sort("importance", descending=True)
)

print("Frequency model feature importances:")
print(imp_df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(imp_df["feature"].to_list()[::-1], imp_df["importance"].to_list()[::-1], color="steelblue")
ax.set_xlabel("Feature importance (PredictionValuesChange)")
ax.set_title("CatBoost frequency model: feature importances")
plt.tight_layout()

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_figure(fig, "feature_importance.png")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Register the challenger model
# MAGIC
# MAGIC We register the frequency model as "challenger" - not "production". The model
# MAGIC has not yet passed governance review. In Module 4, the SHAP relativities are
# MAGIC extracted and reviewed by the pricing committee. Only after sign-off does the
# MAGIC model get promoted from "challenger" to "production".
# MAGIC
# MAGIC We use aliases rather than stages: MLflow stages (Staging/Production) were
# MAGIC deprecated in MLflow 2.9+. Aliases are the recommended replacement.

# COMMAND ----------

client = MlflowClient()

MODEL_NAME = "motor_freq_catboost_m03"

try:
    freq_uri = f"runs:/{freq_run_id}/freq_model"
    registered = mlflow.register_model(model_uri=freq_uri, name=MODEL_NAME)

    # Set alias: "challenger" means competing with the production GLM
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="challenger",
        version=registered.version,
    )

    client.set_model_version_tag(name=MODEL_NAME, version=registered.version,
                                  key="module",      value="module_03")
    client.set_model_version_tag(name=MODEL_NAME, version=registered.version,
                                  key="cv_strategy", value="walk_forward_ibnr1")
    client.set_model_version_tag(name=MODEL_NAME, version=registered.version,
                                  key="gini_lift",   value=str(round(gini_gbm - gini_glm, 4)))

    print(f"Registered: {MODEL_NAME} version {registered.version} as 'challenger'")
    print(f"Load later: mlflow.catboost.load_model('models:/{MODEL_NAME}@challenger')")

except Exception as e:
    print(f"Registry error (may not have permissions on Free Edition): {e}")
    print(f"Model logged at run_id: {freq_run_id}")
    print(f"Load with: mlflow.catboost.load_model('runs:/{freq_run_id}/freq_model')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What this notebook built:
# MAGIC
# MAGIC | Step | What was produced |
# MAGIC |------|-------------------|
# MAGIC | Walk-forward CV | 3 folds, IBNR buffer of 1 year, mean deviance logged |
# MAGIC | Optuna tuning | 40 trials on last fold; best depth/lr/l2/iterations logged |
# MAGIC | Frequency model | CatBoost Poisson, trained on all years except the test year |
# MAGIC | Severity model | CatBoost Tweedie (power=2), claims-only subset |
# MAGIC | GLM comparison | Gini lift, double lift chart, logged to MLflow run |
# MAGIC | Model registry | Challenger alias set; ready for Module 4 SHAP extraction |
# MAGIC
# MAGIC The double lift chart is the key output for the pricing committee.
# MAGIC A positively-sloping chart means the GBM is finding real additional signal.
# MAGIC A flat chart means the feature set is already well-captured by the GLM.
# MAGIC
# MAGIC Next: Module 4 extracts SHAP relativities from the frequency model,
# MAGIC converting the GBM's internal representation into a reviewable factor table.

# COMMAND ----------

print("=" * 60)
print("MODULE 3 COMPLETE")
print("=" * 60)
print()
print(f"Frequency model run:  {freq_run_id}")
print(f"Severity model run:   {sev_run_id}")
print(f"GBM Gini:             {gini_gbm:.3f}")
print(f"GLM Gini:             {gini_glm:.3f}")
print(f"Gini lift:            {gini_gbm - gini_glm:+.3f}")
print(f"Test Poisson deviance:{test_dev:.4f}")
print()
print("Next: Module 4 - SHAP Relativities")
print("The double lift chart and Gini lift are inputs to the pricing committee decision.")
print("Module 4 extracts the underlying relativities for governance sign-off.")
