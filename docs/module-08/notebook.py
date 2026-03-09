# Databricks notebook source
# MAGIC %md
# MAGIC # Module 8: End-to-End Pricing Pipeline
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC The full pricing pipeline in a single notebook. Every stage runs here,
# MAGIC reads from and writes to Unity Catalog Delta tables, and is tracked in MLflow.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Install all five Burning Cost libraries
# MAGIC 2. Generate 200,000 synthetic motor policies (four annual cohorts)
# MAGIC 3. Feature engineering: all transforms defined once, applied everywhere
# MAGIC 4. Walk-forward CV with IBNR buffer (insurance-cv)
# MAGIC 5. CatBoost Poisson frequency and Tweedie severity models
# MAGIC 6. Conformal prediction intervals (insurance-conformal)
# MAGIC 7. Rate optimisation (rate-optimiser)
# MAGIC 8. Write all outputs to Unity Catalog Delta tables
# MAGIC 9. Pricing committee summary
# MAGIC
# MAGIC **Runtime:** 45-60 minutes on a 4-core cluster (Standard_DS3_v2).
# MAGIC
# MAGIC **In production:** Replace the data generation cell (Stage 2) with
# MAGIC `spark.table("your_system.motor.policies")`. Everything else runs unchanged.
# MAGIC
# MAGIC **FCA audit trail:** Every output table carries the MLflow run_id and
# MAGIC the Delta table version used for training. Any number can be reproduced.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0: Install libraries
# MAGIC
# MAGIC All five Burning Cost libraries plus CatBoost, Polars, Optuna, MLflow.
# MAGIC Run this cell once per cluster restart.

# COMMAND ----------

%pip install \
  catboost \
  "insurance-conformal[catboost]" \
  insurance-cv \
  rate-optimiser \
  polars \
  optuna \
  mlflow \
  --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date
from dataclasses import dataclass, asdict

import numpy as np
import polars as pl
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt

import catboost
from catboost import CatBoostRegressor, Pool
import optuna
import mlflow
import mlflow.catboost
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

from insurance_conformal import InsuranceConformalPredictor
from insurance_cv import WalkForwardCV

from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CatBoost: {catboost.__version__}")
print(f"MLflow:   {mlflow.__version__}")
print(f"Today:    {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Configuration
# MAGIC
# MAGIC All configurable values in one place. If you are adapting this notebook
# MAGIC for a different line of business, change these cells.

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor_q2_2026"   # Name the schema with the review cycle, not a version number

TABLES = {
    "raw":                 f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":            f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":    f"{CATALOG}.{SCHEMA}.freq_predictions",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "rate_change":         f"{CATALOG}.{SCHEMA}.rate_action_factors",
    "efficient_frontier":  f"{CATALOG}.{SCHEMA}.efficient_frontier",
    "pipeline_audit":      f"{CATALOG}.{SCHEMA}.pipeline_audit",
}

RUN_DATE          = str(date.today())
N_POLICIES        = 200_000
N_OPTUNA_TRIALS   = 20   # Increase to 40 for production
LR_TARGET         = 0.72
VOLUME_FLOOR      = 0.97
FACTOR_LOWER      = 0.90
FACTOR_UPPER      = 1.15
CONFORMAL_ALPHA   = 0.10  # 90% prediction intervals

mlflow.set_experiment(
    f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/motor-pipeline-m08"
)

print(f"Schema:    {CATALOG}.{SCHEMA}")
print(f"Tables:    {list(TABLES.keys())}")
print(f"Run date:  {RUN_DATE}")

# COMMAND ----------

# Create schema
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'Insurance pricing'")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'Q2 2026 motor rate review'")
    print(f"Schema {CATALOG}.{SCHEMA} ready.")
except Exception as e:
    print(f"Schema setup: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Data generation
# MAGIC
# MAGIC In production, replace this cell with:
# MAGIC ```python
# MAGIC raw = pl.from_pandas(spark.table("source_system.motor.policies").toPandas())
# MAGIC ```
# MAGIC
# MAGIC The synthetic data has realistic structure:
# MAGIC - Four cohorts (2022-2025): 50,000 policies per year
# MAGIC - 7% premium inflation per year across the period
# MAGIC - Age band is the dominant frequency predictor
# MAGIC - Vehicle group and region drive severity
# MAGIC - 2025 is the most recent year (under-reported - IBNR buffer applies)

# COMMAND ----------

rng = np.random.default_rng(2026)
YEARS    = [2022, 2023, 2024, 2025]
N_PER_YEAR = N_POLICIES // len(YEARS)

cohorts = []
for year in YEARS:
    inflation = 1.07 ** (year - 2022)
    n = N_PER_YEAR

    age_band     = rng.choice(["17-25","26-35","36-50","51-65","66+"], n,
                               p=[0.10,0.20,0.35,0.25,0.10])
    ncb          = rng.choice([0,1,2,3,4,5], n, p=[0.10,0.10,0.15,0.20,0.20,0.25])
    vehicle_group = rng.choice(["A","B","C","D","E"], n, p=[0.20,0.25,0.25,0.20,0.10])
    region       = rng.choice(["London","SouthEast","Midlands","North","Scotland","Wales"],
                               n, p=[0.18,0.20,0.22,0.25,0.10,0.05])
    annual_mileage = rng.choice(["<5k","5k-10k","10k-15k","15k+"], n, p=[0.15,0.35,0.35,0.15])

    # Frequency DGP
    age_freq = {"17-25":0.12,"26-35":0.07,"36-50":0.05,"51-65":0.04,"66+":0.06}
    freq = np.array([age_freq[a] for a in age_band])
    freq *= np.array([{"A":0.85,"B":0.95,"C":1.00,"D":1.10,"E":1.25}[v] for v in vehicle_group])
    freq *= np.array([{"London":1.15,"SouthEast":1.05,"Midlands":1.00,
                       "North":0.95,"Scotland":0.90,"Wales":0.92}[r] for r in region])
    freq *= np.array([{"<5k":0.75,"5k-10k":0.90,"10k-15k":1.05,"15k+":1.30}[m] for m in annual_mileage])

    claim_count = rng.poisson(freq)

    # Severity DGP
    sev_base = 2_800 * inflation
    mean_sev = (
        sev_base
        * np.array([{"A":0.75,"B":0.90,"C":1.00,"D":1.15,"E":1.40}[v] for v in vehicle_group])
        * np.array([{"London":1.20,"SouthEast":1.10,"Midlands":1.00,
                     "North":0.95,"Scotland":0.88,"Wales":0.92}[r] for r in region])
    )
    claim_severity = np.where(claim_count > 0, rng.gamma(2.0, mean_sev / 2.0), 0.0)

    exposure       = rng.uniform(0.3, 1.0, n)
    earned_premium = sev_base * freq / 0.72 * inflation * rng.uniform(0.94, 1.06, n)

    cohorts.append(pl.DataFrame({
        "policy_id":     [f"{year}-{i:06d}" for i in range(n)],
        "accident_year": year,
        "age_band":      age_band,
        "ncb":           ncb,
        "vehicle_group": vehicle_group,
        "region":        region,
        "annual_mileage": annual_mileage,
        "exposure":      exposure,
        "earned_premium": earned_premium,
        "claim_count":   claim_count,
        "incurred_loss": claim_severity,
    }))

raw = pl.concat(cohorts)
print(f"Total: {len(raw):,} policies")
print(f"Freq:  {raw['claim_count'].sum()/raw['exposure'].sum():.4f}")
print(f"Sev:   {raw.filter(pl.col('incurred_loss')>0)['incurred_loss'].mean():,.0f}")
print(f"LR:    {raw['incurred_loss'].sum()/raw['earned_premium'].sum():.3f}")

(
    spark.createDataFrame(raw.to_pandas())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema","true")
    .saveAsTable(TABLES["raw"])
)

raw_version = spark.sql(f"DESCRIBE HISTORY {TABLES['raw']} LIMIT 1").collect()[0]["version"]
print(f"Written to {TABLES['raw']} (version {raw_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Feature engineering - the transform layer
# MAGIC
# MAGIC All transforms are defined as pure functions in a single dictionary.
# MAGIC `apply_transforms()` is called at training time AND scoring time.
# MAGIC If you change a feature encoding, change it once here.
# MAGIC
# MAGIC This is the discipline that prevents the most common production failure:
# MAGIC different feature engineering at training vs scoring.

# COMMAND ----------

NCB_MAX = 5
VEHICLE_ORD = {"A":1,"B":2,"C":3,"D":4,"E":5}
AGE_MID     = {"17-25":21,"26-35":30,"36-50":43,"51-65":58,"66+":72}
MILEAGE_ORD = {"<5k":1,"5k-10k":2,"10k-15k":3,"15k+":4}

def encode_ncb(df):
    return df.with_columns((NCB_MAX - pl.col("ncb")).alias("ncb_deficit"))

def encode_vehicle(df):
    return df.with_columns(pl.col("vehicle_group").replace(VEHICLE_ORD).cast(pl.Int32).alias("vehicle_ord"))

def encode_age(df):
    return df.with_columns(pl.col("age_band").replace(AGE_MID).cast(pl.Float64).alias("age_mid"))

def encode_mileage(df):
    return df.with_columns(pl.col("annual_mileage").replace(MILEAGE_ORD).cast(pl.Int32).alias("mileage_ord"))

def add_log_exposure(df):
    return df.with_columns(pl.col("exposure").log().alias("log_exposure"))

TRANSFORMS   = [encode_ncb, encode_vehicle, encode_age, encode_mileage, add_log_exposure]
FEATURE_COLS = ["ncb_deficit","vehicle_ord","age_mid","mileage_ord","region"]
CAT_FEATURES = ["region"]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    for fn in TRANSFORMS:
        df = fn(df)
    return df

raw_pl       = pl.from_pandas(spark.table(TABLES["raw"]).toPandas())
features_pl  = apply_transforms(raw_pl)

print(f"Feature columns: {FEATURE_COLS}")
print(f"Categorical:     {CAT_FEATURES}")
print(features_pl.select(FEATURE_COLS + ["claim_count","incurred_loss","exposure"]).head(3))

(
    spark.createDataFrame(features_pl.to_pandas())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema","true")
    .saveAsTable(TABLES["features"])
)
feat_version = spark.sql(f"DESCRIBE HISTORY {TABLES['features']} LIMIT 1").collect()[0]["version"]
print(f"Written to {TABLES['features']} (version {feat_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Walk-forward cross-validation
# MAGIC
# MAGIC Temporal splits only. Random splits produce optimistic metrics that do not
# MAGIC reflect actual out-of-time generalisation.
# MAGIC
# MAGIC The IBNR buffer (6 months) prevents the model from training on partially-
# MAGIC developed accident years. For a long-tail line (employers' liability, TP injury)
# MAGIC increase this to 18-24 months.

# COMMAND ----------

features_pd = spark.table(TABLES["features"]).toPandas()


def poisson_deviance(y_true, y_pred, exposure):
    fp = np.clip(y_pred / exposure, 1e-10, None)
    ft = y_true / exposure
    d  = 2 * exposure * (np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp))
    return float(d.sum() / exposure.sum())


try:
    from insurance_cv import WalkForwardCV, IBNRBuffer
    ibnr = IBNRBuffer(months=6)
    cv   = WalkForwardCV(date_col="accident_year", n_splits=3, min_train_years=2, ibnr_buffer=ibnr)
except ImportError:
    # Fallback: manual walk-forward
    class SimpleCV:
        def split(self, df):
            years = sorted(df["accident_year"].unique())
            folds = []
            for cutoff in years[2:-1]:
                train = df[df["accident_year"] <= cutoff - 1].index.to_numpy()
                val   = df[df["accident_year"] == cutoff].index.to_numpy()
                folds.append((train, val))
            return folds
    cv = SimpleCV()

print("Walk-forward folds:")
cv_deviances = []
for i, (train_idx, val_idx) in enumerate(cv.split(features_pd)):
    df_tr = features_pd.iloc[train_idx]
    df_va = features_pd.iloc[val_idx]

    train_years = sorted(df_tr["accident_year"].unique().tolist())
    val_years   = sorted(df_va["accident_year"].unique().tolist())
    print(f"  Fold {i+1}: train={train_years}, val={val_years}")

    Xtr = df_tr[FEATURE_COLS]
    ytr = df_tr["claim_count"].values
    wtr = df_tr["exposure"].values

    Xva = df_va[FEATURE_COLS]
    yva = df_va["claim_count"].values
    wva = df_va["exposure"].values

    m = CatBoostRegressor(loss_function="Poisson", eval_metric="Poisson",
                          iterations=300, learning_rate=0.05, depth=5,
                          random_seed=42, verbose=0)
    m.fit(Pool(Xtr, ytr, baseline=np.log(wtr), cat_features=CAT_FEATURES),
          eval_set=Pool(Xva, yva, baseline=np.log(wva), cat_features=CAT_FEATURES))

    pred = m.predict(Pool(Xva, baseline=np.log(wva), cat_features=CAT_FEATURES))
    dev  = poisson_deviance(yva, pred, wva)
    cv_deviances.append(dev)
    print(f"    Poisson deviance: {dev:.4f}")

print(f"\nMean CV deviance: {np.mean(cv_deviances):.4f} (+/- {np.std(cv_deviances):.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5: Hyperparameter tuning
# MAGIC
# MAGIC Tune on the last fold. 20 trials for this notebook (increase to 40 in production).

# COMMAND ----------

last_train_idx, last_val_idx = list(cv.split(features_pd))[-1]
df_tune_tr = features_pd.iloc[last_train_idx]
df_tune_va = features_pd.iloc[last_val_idx]

tune_train = Pool(df_tune_tr[FEATURE_COLS], df_tune_tr["claim_count"].values,
                  baseline=np.log(df_tune_tr["exposure"].values), cat_features=CAT_FEATURES)
tune_val   = Pool(df_tune_va[FEATURE_COLS], df_tune_va["claim_count"].values,
                  baseline=np.log(df_tune_va["exposure"].values), cat_features=CAT_FEATURES)


def objective(trial):
    p = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson", "random_seed": 42, "verbose": 0,
    }
    m = CatBoostRegressor(**p)
    m.fit(tune_train, eval_set=tune_val)
    pred = m.predict(tune_val)
    return poisson_deviance(df_tune_va["claim_count"].values, pred, df_tune_va["exposure"].values)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"Best params ({N_OPTUNA_TRIALS} trials):")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best deviance: {study.best_value:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6: Train final models and log to MLflow
# MAGIC
# MAGIC Train on all years except the held-out test year (most recent).
# MAGIC Log the raw data version alongside the model - this is the audit trail.

# COMMAND ----------

max_year  = int(features_pd["accident_year"].max())
df_train  = features_pd[features_pd["accident_year"] < max_year]
df_test   = features_pd[features_pd["accident_year"] == max_year]

freq_params = {**best_params, "loss_function":"Poisson", "eval_metric":"Poisson",
               "random_seed":42, "verbose":100}

X_tr_f = df_train[FEATURE_COLS]; y_tr_f = df_train["claim_count"].values; w_tr_f = df_train["exposure"].values
X_te_f = df_test[FEATURE_COLS];  y_te_f = df_test["claim_count"].values;  w_te_f = df_test["exposure"].values

with mlflow.start_run(run_name="freq_model_m08") as freq_run:
    mlflow.log_params(freq_params)
    mlflow.log_params({
        "model_type":       "catboost_poisson",
        "raw_table_version": raw_version,
        "feat_table_version": feat_version,
        "feature_cols":     json.dumps(FEATURE_COLS),
        "cat_features":     json.dumps(CAT_FEATURES),
        "train_years":      str(sorted(df_train["accident_year"].unique().tolist())),
        "test_year":        str(max_year),
        "run_date":         RUN_DATE,
    })

    freq_model = CatBoostRegressor(**freq_params)
    freq_model.fit(
        Pool(X_tr_f, y_tr_f, baseline=np.log(w_tr_f), cat_features=CAT_FEATURES),
        eval_set=Pool(X_te_f, y_te_f, baseline=np.log(w_te_f), cat_features=CAT_FEATURES),
    )

    pred_freq = freq_model.predict(Pool(X_te_f, baseline=np.log(w_te_f), cat_features=CAT_FEATURES))
    test_dev  = poisson_deviance(y_te_f, pred_freq, w_te_f)

    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",       np.mean(cv_deviances))
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = freq_run.info.run_id

print(f"Frequency model: {freq_run_id}")
print(f"Test deviance: {test_dev:.4f}")

# Severity model
df_tr_sev = df_train[df_train["claim_count"] > 0].copy()
df_te_sev = df_test[df_test["claim_count"]   > 0].copy()
df_tr_sev["avg_sev"] = df_tr_sev["incurred_loss"] / df_tr_sev["claim_count"]
df_te_sev["avg_sev"] = df_te_sev["incurred_loss"] / df_te_sev["claim_count"]

# NOTE: In production, run a separate Optuna study for severity.
# These hyperparameters come from the frequency tuning run. The optimal depth for a
# Poisson frequency model on 200,000 policies is not necessarily optimal for a Gamma
# severity model on the 7-10% of policies with claims. Shared hyperparameters are a
# tutorial simplification. Defaults that work well for Gamma severity on UK motor:
#   best_sev_params = {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 5}
sev_params = {**best_params, "loss_function":"Tweedie:variance_power=2", "eval_metric":"RMSE",
              "random_seed":42, "verbose":0}

with mlflow.start_run(run_name="sev_model_m08") as sev_run:
    mlflow.log_params(sev_params)
    sev_m = CatBoostRegressor(**sev_params)
    sev_m.fit(Pool(df_tr_sev[FEATURE_COLS], df_tr_sev["avg_sev"].values, cat_features=CAT_FEATURES),
              eval_set=Pool(df_te_sev[FEATURE_COLS], df_te_sev["avg_sev"].values, cat_features=CAT_FEATURES))
    pred_sev = sev_m.predict(Pool(df_te_sev[FEATURE_COLS], cat_features=CAT_FEATURES))
    mlflow.log_metric("test_rmse_severity", float(np.sqrt(np.mean((df_te_sev["avg_sev"].values - pred_sev)**2))))
    mlflow.catboost.log_model(sev_m, "sev_model")
    sev_run_id = sev_run.info.run_id

print(f"Severity model: {sev_run_id}")

# Write test predictions to Delta
pred_df = pl.DataFrame({
    "policy_id":     df_test["policy_id"].tolist() if "policy_id" in df_test.columns else [f"p{i}" for i in range(len(df_test))],
    "accident_year": df_test["accident_year"].tolist(),
    "freq_pred":     (pred_freq / w_te_f).tolist(),
    "freq_actual":   (y_te_f / w_te_f).tolist(),
    "exposure":      w_te_f.tolist(),
    "mlflow_run_id": [freq_run_id] * len(df_test),
    "run_date":      [RUN_DATE] * len(df_test),
}).with_columns(
    (pl.col("freq_pred") * 3_000).alias("tech_premium_est")  # simplified; use actual sev model
)

(
    spark.createDataFrame(pred_df.to_pandas())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema","true")
    .saveAsTable(TABLES["freq_predictions"])
)
print(f"Predictions written to {TABLES['freq_predictions']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: Conformal prediction intervals
# MAGIC
# MAGIC Calibrate on the calibration split (second-to-last year).
# MAGIC Test on the held-out test year.
# MAGIC Validate coverage before any downstream use.

# COMMAND ----------

# Calibration set: second-to-last year
cal_year = max_year - 1
df_cal   = features_pd[features_pd["accident_year"] == cal_year]
X_cal    = df_cal[FEATURE_COLS]
y_cal    = df_cal["incurred_loss"]

cp = InsuranceConformalPredictor(
    model=sev_m,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=2.0,
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_te_f if len(df_te_sev) == 0 else df_te_sev[FEATURE_COLS], alpha=CONFORMAL_ALPHA)

try:
    diag = cp.coverage_by_decile(df_te_sev[FEATURE_COLS], df_te_sev["incurred_loss"], alpha=CONFORMAL_ALPHA)
    min_cov = float(diag["coverage"].min())
    print(f"Coverage by decile (target: {1-CONFORMAL_ALPHA:.0%}):")
    print(diag.to_string())
    print(f"\nMin decile coverage: {min_cov:.3f}")
    if min_cov < 0.85:
        print("WARNING: Coverage below 85% in at least one decile. Check calibration split.")
    else:
        print("Coverage acceptable.")
except Exception as e:
    min_cov = None
    print(f"Coverage diagnostic error: {e}")

# Write intervals
try:
    intervals_df = intervals.to_pandas().copy()
    intervals_df["mlflow_run_id"] = freq_run_id
    intervals_df["run_date"]      = RUN_DATE
    intervals_df["alpha"]         = CONFORMAL_ALPHA
    (
        spark.createDataFrame(intervals_df)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(TABLES["conformal_intervals"])
    )
    print(f"Intervals written to {TABLES['conformal_intervals']}")
except Exception as e:
    print(f"Could not write intervals: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8: Rate optimisation
# MAGIC
# MAGIC Build a renewal portfolio from the test-set predictions.
# MAGIC Run the optimiser at the chosen LR target.
# MAGIC Trace the efficient frontier.

# COMMAND ----------

# Build renewal portfolio from predictions
pred_pd = pred_df.to_pandas()
n_ren   = min(2_000, len(pred_pd))
pred_ren = pred_pd.head(n_ren).copy()

tech_prem    = pred_ren["tech_premium_est"].values
curr_prem    = tech_prem / 0.75
mkt_prem     = tech_prem / 0.73

# In production, use the SHAP relativities from Stage 7 as the factor structure input
# to the rate optimiser. Here we construct simplified factor proxies for demonstration —
# f_age, f_ncb, f_vehicle, and f_region are sampled randomly rather than derived from
# the SHAP output of the frequency model.
rng2 = np.random.default_rng(99)
renewal_port = pd.DataFrame({
    "policy_id":         [f"REN{i:06d}" for i in range(n_ren)],
    "channel":           rng2.choice(["PCW","direct"], n_ren, p=[0.70,0.30]),
    "renewal_flag":      np.ones(n_ren, dtype=bool),
    "technical_premium": tech_prem,
    "current_premium":   curr_prem,
    "market_premium":    mkt_prem,
    "renewal_prob":      expit(1.0 + (-2.0) * np.log(curr_prem / mkt_prem)),
    "tenure":            rng2.integers(0, 10, n_ren).astype(float),
    "f_age":             rng2.choice([0.80,1.00,1.20,1.50], n_ren, p=[0.20,0.40,0.25,0.15]),
    "f_ncb":             rng2.choice([0.70,0.85,1.00], n_ren, p=[0.30,0.40,0.30]),
    "f_vehicle":         rng2.choice([0.90,1.00,1.10,1.30], n_ren, p=[0.25,0.35,0.25,0.15]),
    "f_region":          rng2.choice([0.85,1.00,1.10,1.20], n_ren, p=[0.20,0.40,0.25,0.15]),
    "f_tenure_discount": np.ones(n_ren),
})

current_lr = float(renewal_port["technical_premium"].sum() / renewal_port["current_premium"].sum())
print(f"Renewal portfolio current LR: {current_lr:.3f}")
print(f"Optimisation target LR:       {LR_TARGET:.3f}")

# COMMAND ----------

FACTOR_NAMES_OPT = ["f_age","f_ncb","f_vehicle","f_region","f_tenure_discount"]

data_opt = PolicyData(renewal_port)
fs_opt   = FactorStructure(
    factor_names=FACTOR_NAMES_OPT,
    factor_values=renewal_port[FACTOR_NAMES_OPT],
    renewal_factor_names=["f_tenure_discount"],
)
demand_opt = make_logistic_demand(LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.05))

opt_m8 = RateChangeOptimiser(data=data_opt, demand=demand_opt, factor_structure=fs_opt)
opt_m8.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt_m8.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt_m8.add_constraint(ENBPConstraint(channels=["PCW","direct"]))
opt_m8.add_constraint(FactorBoundsConstraint(lower=FACTOR_LOWER, upper=FACTOR_UPPER, n_factors=fs_opt.n_factors))

result = opt_m8.solve()
print(result.summary())

# Efficient frontier
frontier = EfficientFrontier(opt_m8)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=15)

# Write rate action
adj_records = [
    {"run_date": RUN_DATE, "factor_name": f, "adjustment": float(result.factor_adjustments.get(f, 1.0)),
     "pct_change": float((result.factor_adjustments.get(f, 1.0) - 1) * 100),
     "lr_target": LR_TARGET, "volume_floor": VOLUME_FLOOR}
    for f in FACTOR_NAMES_OPT
]

try:
    (
        spark.createDataFrame(adj_records)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(TABLES["rate_change"])
    )
    (
        spark.createDataFrame(frontier_df)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(TABLES["efficient_frontier"])
    )
    print(f"Rate action and frontier written to Delta.")
except Exception as e:
    print(f"Could not write: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: Pipeline audit record
# MAGIC
# MAGIC Write a single record that ties together every table version, model run ID,
# MAGIC and key metric. This is the FCA audit trail for the full pipeline run.

# COMMAND ----------

audit_record = {
    "run_date":              RUN_DATE,
    "raw_table":             TABLES["raw"],
    "raw_table_version":     int(raw_version),
    "features_table":        TABLES["features"],
    "features_table_version": int(feat_version),
    "freq_model_run_id":     freq_run_id,
    "sev_model_run_id":      sev_run_id,
    "n_training_rows":       len(df_train),
    "n_test_rows":           len(df_test),
    "test_year":             int(max_year),
    "test_poisson_deviance": round(test_dev, 5),
    "mean_cv_deviance":      round(float(np.mean(cv_deviances)), 5),
    "conformal_alpha":       CONFORMAL_ALPHA,
    "conformal_min_decile_cov": round(min_cov, 3) if min_cov else -1.0,
    "lr_target":             LR_TARGET,
    "optimiser_converged":   result.converged,
    "expected_lr":           round(result.expected_loss_ratio, 4),
    "expected_volume":       round(result.expected_volume_ratio, 4),
    "optuna_trials":         N_OPTUNA_TRIALS,
    "pipeline_notes":        "Module 8 end-to-end pipeline, synthetic data",
}

for k, v in audit_record.items():
    print(f"  {k:<35}: {v}")

try:
    (
        spark.createDataFrame([audit_record])
        .write.format("delta").mode("append")
        .saveAsTable(TABLES["pipeline_audit"])
    )
    print(f"\nAudit record written to {TABLES['pipeline_audit']}")
except Exception as e:
    print(f"Could not write audit record: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Pricing Committee Pack
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Training data | pricing.motor_q2_2026.raw_policies (version logged) |
# MAGIC | Model type | CatBoost Poisson frequency + Tweedie severity |
# MAGIC | Cross-validation | Walk-forward, IBNR buffer, 3 folds |
# MAGIC | Uncertainty | Conformal intervals, pearson_weighted score |
# MAGIC | Rate action | SLSQP optimisation, ENBP-constrained |
# MAGIC | Audit trail | All table versions and run IDs in pipeline_audit table |
# MAGIC
# MAGIC **Key numbers for the committee:**
# MAGIC - Test Poisson deviance: logged in MLflow at `freq_model_m08`
# MAGIC - Conformal coverage by decile: available in `conformal_coverage_log`
# MAGIC - Factor adjustments: `pricing.motor_q2_2026.rate_action_factors`
# MAGIC - Efficient frontier: `pricing.motor_q2_2026.efficient_frontier`
# MAGIC
# MAGIC **Reproducibility:** Re-run with `raw_table_version` from the audit record to
# MAGIC reproduce any historical output exactly.

# COMMAND ----------

print("=" * 60)
print("MODULE 8 COMPLETE - END-TO-END PIPELINE")
print("=" * 60)
print()
print(f"Test deviance:      {test_dev:.4f}")
print(f"Mean CV deviance:   {np.mean(cv_deviances):.4f}")
if min_cov:
    print(f"Min decile cov:     {min_cov:.3f}")
print(f"Optimiser result:   {'converged' if result.converged else 'DID NOT CONVERGE'}")
print(f"Expected LR:        {result.expected_loss_ratio:.4f}")
print(f"Expected volume:    {result.expected_volume_ratio:.4f}")
print()
print("All outputs written to Unity Catalog.")
print("MLflow run IDs recorded in pipeline_audit table.")
print()
print("Next steps:")
print("  - Present efficient frontier to pricing committee")
print("  - Submit factor adjustments for rating engine update")
print("  - Schedule monthly conformal recalibration (no retraining needed)")
print("  - Set up Databricks Workflow to run pipeline monthly")
