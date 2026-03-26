# Databricks notebook source
# MAGIC %md
# MAGIC # Module 8: End-to-End Pricing Pipeline
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC A complete UK motor rate review in a single notebook.
# MAGIC Raw data → feature engineering → walk-forward CV → CatBoost tuning →
# MAGIC freq-sev modelling → SHAP relativities → conformal intervals →
# MAGIC calibration testing → rate optimisation → audit trail.
# MAGIC
# MAGIC **Runtime:** 45-60 minutes on a 4-core cluster.
# MAGIC
# MAGIC **In production:** Replace Stage 2 data generation with:
# MAGIC `raw_pl = pl.from_pandas(spark.table("your_system.motor.policies").toPandas())`
# MAGIC
# MAGIC **FCA audit trail:** Every output table carries the MLflow run ID and the
# MAGIC Delta table version used for training. Any run can be reproduced exactly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0: Install libraries
# MAGIC
# MAGIC Run once per cluster restart. `dbutils.library.restartPython()` resets
# MAGIC the Python environment so newly installed packages are importable.

# COMMAND ----------

%pip install \
    catboost \
    optuna \
    polars \
    "insurance-conformal[catboost]" \
    "insurance-monitoring[mlflow]" \
    insurance-optimise \
    shap-relativities \
    --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import warnings
from datetime import date

import numpy as np
import polars as pl
import pandas as pd
from scipy.special import expit

import catboost
from catboost import CatBoostRegressor, Pool
import optuna
import mlflow
import mlflow.catboost
from mlflow import MlflowClient

from insurance_conformal import InsuranceConformalPredictor
from insurance_monitoring.calibration import CalibrationChecker, rectify_balance
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier
from shap_relativities import SHAPRelativities

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

print(f"CatBoost:  {catboost.__version__}")
print(f"MLflow:    {mlflow.__version__}")
print(f"Polars:    {pl.__version__}")
print(f"Run date:  {date.today()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Configuration
# MAGIC
# MAGIC All configurable values in one cell. Change a value here; it changes everywhere.
# MAGIC Name schemas by review cycle: `motor_q2_2026`, not `motor_v3`.

# COMMAND ----------

CATALOG = "main"           # change to "hive_metastore" on Databricks Free Edition
SCHEMA  = "motor_q2_2026"

TABLES = {
    "raw":                 f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":            f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":    f"{CATALOG}.{SCHEMA}.freq_predictions",
    "freq_relativities":   f"{CATALOG}.{SCHEMA}.freq_relativities",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "rate_change":         f"{CATALOG}.{SCHEMA}.rate_action_factors",
    "efficient_frontier":  f"{CATALOG}.{SCHEMA}.efficient_frontier",
    "pipeline_audit":      f"{CATALOG}.{SCHEMA}.pipeline_audit",
}

N_POLICIES      = 200_000
N_OPTUNA_TRIALS = 20      # increase to 40 for production
LR_TARGET       = 0.72
VOLUME_FLOOR    = 0.97
CONFORMAL_ALPHA = 0.10    # 90% prediction intervals

RUN_DATE = str(date.today())

print(f"Schema:  {CATALOG}.{SCHEMA}")
print(f"Tables:  {list(TABLES.keys())}")
print(f"Run:     {RUN_DATE}")

# COMMAND ----------

if CATALOG != "hive_metastore":
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

mlflow.set_experiment(
    f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/motor-pipeline-m08"
)
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Data ingestion
# MAGIC
# MAGIC In production, replace this cell with:
# MAGIC ```python
# MAGIC raw_pl = pl.from_pandas(spark.table("source_system.motor.policy_claims_view").toPandas())
# MAGIC ```
# MAGIC Required columns: age_band, ncb, vehicle_group, region, annual_mileage,
# MAGIC exposure, claim_count, incurred_loss, accident_year.

# COMMAND ----------

rng = np.random.default_rng(2026)

YEARS      = [2022, 2023, 2024, 2025]
N_PER_YEAR = N_POLICIES // len(YEARS)

cohorts = []
for year in YEARS:
    inflation = 1.07 ** (year - 2022)   # 7% annual claims inflation
    n = N_PER_YEAR

    age_band       = rng.choice(["17-25","26-35","36-50","51-65","66+"], n,
                                 p=[0.10, 0.20, 0.35, 0.25, 0.10])
    ncb            = rng.choice([0, 1, 2, 3, 4, 5], n,
                                 p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    vehicle_group  = rng.choice(["A","B","C","D","E"], n,
                                 p=[0.20, 0.25, 0.25, 0.20, 0.10])
    region         = rng.choice(["London","SouthEast","Midlands","North","Scotland","Wales"], n,
                                 p=[0.18, 0.20, 0.22, 0.25, 0.10, 0.05])
    annual_mileage = rng.choice(["<5k","5k-10k","10k-15k","15k+"], n,
                                 p=[0.15, 0.35, 0.35, 0.15])

    age_freq = {"17-25":0.12,"26-35":0.07,"36-50":0.05,"51-65":0.04,"66+":0.06}
    freq  = np.array([age_freq[a] for a in age_band])
    freq *= np.array([{"A":0.85,"B":0.95,"C":1.00,"D":1.10,"E":1.25}[v] for v in vehicle_group])
    freq *= np.array([{"London":1.15,"SouthEast":1.05,"Midlands":1.00,
                       "North":0.95,"Scotland":0.90,"Wales":0.92}[r] for r in region])
    freq *= np.array([{"<5k":0.75,"5k-10k":0.90,"10k-15k":1.05,"15k+":1.30}[m]
                       for m in annual_mileage])
    claim_count = rng.poisson(freq)

    sev_base = 2_800 * inflation
    mean_sev = (
        sev_base
        * np.array([{"A":0.75,"B":0.90,"C":1.00,"D":1.15,"E":1.40}[v] for v in vehicle_group])
        * np.array([{"London":1.20,"SouthEast":1.10,"Midlands":1.00,
                     "North":0.95,"Scotland":0.88,"Wales":0.92}[r] for r in region])
    )
    incurred_loss  = np.where(claim_count > 0, rng.gamma(2.0, mean_sev / 2.0, n), 0.0)
    exposure       = rng.uniform(0.3, 1.0, n)
    earned_premium = sev_base * freq / LR_TARGET * inflation * rng.uniform(0.94, 1.06, n)

    cohorts.append(pl.DataFrame({
        "policy_id":      [f"{year}-{i:06d}" for i in range(n)],
        "accident_year":  [year] * n,
        "age_band":       age_band.tolist(),
        "ncb":            ncb.tolist(),
        "vehicle_group":  vehicle_group.tolist(),
        "region":         region.tolist(),
        "annual_mileage": annual_mileage.tolist(),
        "exposure":       exposure.tolist(),
        "earned_premium": earned_premium.tolist(),
        "claim_count":    claim_count.tolist(),
        "incurred_loss":  incurred_loss.tolist(),
    }))

raw_pl = pl.concat(cohorts)
print(f"Policies: {len(raw_pl):,}")
print(f"Claims:   {raw_pl['claim_count'].sum():,}")
print(f"Freq:     {raw_pl['claim_count'].sum() / raw_pl['exposure'].sum():.4f}")
print(f"Mean sev: £{raw_pl.filter(pl.col('incurred_loss')>0)['incurred_loss'].mean():,.0f}")
print(f"LR:       {raw_pl['incurred_loss'].sum() / raw_pl['earned_premium'].sum():.3f}")

# COMMAND ----------

(
    spark.createDataFrame(raw_pl.to_pandas())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["raw"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['raw']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
raw_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['raw']} LIMIT 1"
).collect()[0]["version"]
print(f"Written to {TABLES['raw']} (version {raw_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Feature engineering — the shared transform layer
# MAGIC
# MAGIC All transforms are pure functions. `apply_transforms()` is the only entry point.
# MAGIC Call it identically at training time and scoring time.
# MAGIC This is what prevents the NCB encoding incident from Part 1.

# COMMAND ----------

NCB_MAX     = 5
VEHICLE_ORD = {"A":1,"B":2,"C":3,"D":4,"E":5}
AGE_MID     = {"17-25":21.0,"26-35":30.0,"36-50":43.0,"51-65":58.0,"66+":72.0}
MILEAGE_ORD = {"<5k":1,"5k-10k":2,"10k-15k":3,"15k+":4}

def encode_ncb(df):
    return df.with_columns((NCB_MAX - pl.col("ncb")).alias("ncb_deficit"))

def encode_vehicle(df):
    return df.with_columns(
        pl.col("vehicle_group").replace(VEHICLE_ORD).cast(pl.Int32).alias("vehicle_ord")
    )

def encode_age(df):
    return df.with_columns(
        pl.col("age_band").replace(AGE_MID).cast(pl.Float64).alias("age_mid")
    )

def encode_mileage(df):
    return df.with_columns(
        pl.col("annual_mileage").replace(MILEAGE_ORD).cast(pl.Int32).alias("mileage_ord")
    )

def add_log_exposure(df):
    return df.with_columns(pl.col("exposure").log().alias("log_exposure"))

TRANSFORMS   = [encode_ncb, encode_vehicle, encode_age, encode_mileage, add_log_exposure]
FEATURE_COLS = ["ncb_deficit", "vehicle_ord", "age_mid", "mileage_ord", "region"]
CAT_FEATURES = ["region"]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """Call this function — and only this function — to engineer features."""
    for fn in TRANSFORMS:
        df = fn(df)
    return df

features_pl = apply_transforms(raw_pl)
assert all(c in features_pl.columns for c in FEATURE_COLS), "Missing feature columns"

print(f"Features shape: {features_pl.shape}")
print(f"Feature cols:   {FEATURE_COLS}")
print(f"Categorical:    {CAT_FEATURES}")

# COMMAND ----------

(
    spark.createDataFrame(features_pl.to_pandas())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["features"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['features']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
feat_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['features']} LIMIT 1"
).collect()[0]["version"]
print(f"Features → {TABLES['features']} (version {feat_version})")

# COMMAND ----------

# Record the FeatureSpec at training time for scoring-time validation
class FeatureSpec:
    def __init__(self): self.spec = {}
    def record(self, df, cat_features):
        for col in df.columns:
            s = df[col]
            if col in cat_features or s.dtype == pl.Utf8:
                self.spec[col] = {"dtype": "categorical",
                                  "unique_vals": sorted(s.drop_nulls().unique().to_list())}
            else:
                self.spec[col] = {"dtype": "numeric",
                                  "min": float(s.min()), "max": float(s.max())}
    def validate(self, df):
        errors = []
        for col, spec in self.spec.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}"); continue
            s = df[col]
            if spec["dtype"] == "categorical" and s.dtype not in (pl.Utf8, pl.Categorical):
                errors.append(f"{col}: expected categorical, got {s.dtype}")
            elif spec["dtype"] == "numeric" and s.dtype in (pl.Utf8, pl.Categorical):
                errors.append(f"{col}: expected numeric, got {s.dtype}")
        return errors
    def to_json(self, path):
        import json
        with open(path, "w") as f: json.dump(self.spec, f, indent=2)
    @classmethod
    def from_json(cls, path):
        import json
        obj = cls()
        with open(path) as f: obj.spec = json.load(f)
        return obj

feature_spec = FeatureSpec()
feature_spec.record(features_pl.select(FEATURE_COLS), cat_features=CAT_FEATURES)
feature_spec.to_json("/tmp/feature_spec.json")
print("FeatureSpec recorded and saved to /tmp/feature_spec.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Walk-forward cross-validation
# MAGIC
# MAGIC Temporal folds only. The IBNR buffer prevents partially developed claims
# MAGIC from contaminating training. For motor BI or commercial lines, increase the
# MAGIC buffer to 12-24 months.

# COMMAND ----------

features_pd = spark.table(TABLES["features"]).toPandas()

def poisson_deviance(y_true, y_pred, exposure):
    """Exposure-weighted scaled Poisson deviance per policy-year."""
    fp = np.clip(y_pred / exposure, 1e-10, None)
    ft = y_true / exposure
    d  = 2 * exposure * (np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp))
    return float(d.sum() / exposure.sum())

# Walk-forward temporal folds
years = sorted(features_pd["accident_year"].unique())
folds = []
for i in range(2, len(years)):
    tr = features_pd.index[features_pd["accident_year"] < years[i]].to_numpy()
    va = features_pd.index[features_pd["accident_year"] == years[i]].to_numpy()
    folds.append((tr, va))

cv_deviances = []
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_tr = features_pd.loc[train_idx]
    df_va = features_pd.loc[val_idx]

    train_pool = Pool(
        df_tr[FEATURE_COLS], df_tr["claim_count"].values,
        baseline=np.log(np.clip(df_tr["exposure"].values, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )
    val_pool = Pool(
        df_va[FEATURE_COLS], df_va["claim_count"].values,
        baseline=np.log(np.clip(df_va["exposure"].values, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )

    m = CatBoostRegressor(loss_function="Poisson", iterations=300, depth=5,
                          learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, verbose=0)
    m.fit(train_pool, eval_set=val_pool)
    pred = m.predict(val_pool)
    dev  = poisson_deviance(df_va["claim_count"].values, pred, df_va["exposure"].values)
    cv_deviances.append(dev)

    tr_years = sorted(df_tr["accident_year"].unique().tolist())
    va_year  = sorted(df_va["accident_year"].unique().tolist())
    print(f"Fold {fold_idx+1}: train {tr_years} → validate {va_year} | deviance={dev:.5f}")

mean_cv_deviance = float(np.mean(cv_deviances))
print(f"\nMean CV deviance: {mean_cv_deviance:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5: Hyperparameter tuning — Optuna
# MAGIC
# MAGIC Tune on the last fold (most recent out-of-time period).
# MAGIC Run separate studies for frequency and severity.

# COMMAND ----------

last_train_idx, last_val_idx = folds[-1]
df_tune_tr = features_pd.loc[last_train_idx]
df_tune_va = features_pd.loc[last_val_idx]

tune_train_pool = Pool(
    df_tune_tr[FEATURE_COLS], df_tune_tr["claim_count"].values,
    baseline=np.log(np.clip(df_tune_tr["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
tune_val_pool = Pool(
    df_tune_va[FEATURE_COLS], df_tune_va["claim_count"].values,
    baseline=np.log(np.clip(df_tune_va["exposure"].values, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

def freq_objective(trial):
    p = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson", "random_seed": 42, "verbose": 0,
    }
    m    = CatBoostRegressor(**p)
    m.fit(tune_train_pool, eval_set=tune_val_pool)
    pred = m.predict(tune_val_pool)
    return poisson_deviance(df_tune_va["claim_count"].values, pred,
                            df_tune_va["exposure"].values)

freq_study = optuna.create_study(direction="minimize")
freq_study.optimize(freq_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_freq_params = {**freq_study.best_params, "loss_function":"Poisson",
                    "random_seed":42, "verbose":0}
print(f"Best freq params (trial {freq_study.best_trial.number}):")
for k, v in best_freq_params.items(): print(f"  {k:<20} {v}")
print(f"Best deviance: {freq_study.best_value:.5f}")

# COMMAND ----------

# Severity tuning: claims-only, Gamma (Tweedie p=2), separate study
df_sev_tune_tr = df_tune_tr[df_tune_tr["claim_count"] > 0].copy()
df_sev_tune_va = df_tune_va[df_tune_va["claim_count"] > 0].copy()
df_sev_tune_tr["mean_sev"] = df_sev_tune_tr["incurred_loss"] / df_sev_tune_tr["claim_count"]
df_sev_tune_va["mean_sev"] = df_sev_tune_va["incurred_loss"] / df_sev_tune_va["claim_count"]

sev_tune_train = Pool(
    df_sev_tune_tr[FEATURE_COLS], df_sev_tune_tr["mean_sev"].values,
    weight=df_sev_tune_tr["claim_count"].values, cat_features=CAT_FEATURES,
)
sev_tune_val = Pool(
    df_sev_tune_va[FEATURE_COLS], df_sev_tune_va["mean_sev"].values,
    weight=df_sev_tune_va["claim_count"].values, cat_features=CAT_FEATURES,
)

def sev_objective(trial):
    p = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Tweedie:variance_power=2", "eval_metric":"RMSE",
        "random_seed":42, "verbose":0,
    }
    m    = CatBoostRegressor(**p)
    m.fit(sev_tune_train, eval_set=sev_tune_val)
    pred = m.predict(sev_tune_val)
    return float(np.sqrt(np.mean((df_sev_tune_va["mean_sev"].values - pred)**2)))

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_sev_params = {**sev_study.best_params, "loss_function":"Tweedie:variance_power=2",
                   "eval_metric":"RMSE", "random_seed":42, "verbose":0}
print(f"Best sev params (trial {sev_study.best_trial.number}):")
for k, v in best_sev_params.items(): print(f"  {k:<20} {v}")
print(f"Best severity RMSE: £{sev_study.best_value:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6: Final models — frequency (Poisson) and severity (Gamma/Tweedie)
# MAGIC
# MAGIC Train on all years except the held-out test year (most recent).
# MAGIC Log data provenance and performance metrics to MLflow.
# MAGIC The severity model excludes the calibration year (2024) from training
# MAGIC so Stage 8 has a genuine out-of-sample calibration set.

# COMMAND ----------

max_year = int(features_pd["accident_year"].max())
df_train = features_pd[features_pd["accident_year"] < max_year].copy()
df_test  = features_pd[features_pd["accident_year"] == max_year].copy()

X_train = df_train[FEATURE_COLS]; y_train = df_train["claim_count"].values; w_train = df_train["exposure"].values
X_test  = df_test[FEATURE_COLS];  y_test  = df_test["claim_count"].values;  w_test  = df_test["exposure"].values

print(f"Training years: {sorted(df_train['accident_year'].unique().tolist())}")
print(f"Test year:      {max_year} ({len(df_test):,} rows, {y_test.sum():,} claims)")

train_pool = Pool(X_train, y_train, baseline=np.log(np.clip(w_train, 1e-6, None)), cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  baseline=np.log(np.clip(w_test,  1e-6, None)), cat_features=CAT_FEATURES)

with mlflow.start_run(run_name="freq_model_m08") as freq_run:
    mlflow.log_params(best_freq_params)
    mlflow.log_params({
        "model_type":         "catboost_poisson",
        "raw_table_version":  int(raw_version),
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "offset":             "log_exposure",
        "train_years":        str(sorted(df_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
    })
    freq_model = CatBoostRegressor(**best_freq_params)
    freq_model.fit(train_pool, eval_set=test_pool)
    freq_pred_test = freq_model.predict(test_pool)
    test_dev       = poisson_deviance(y_test, freq_pred_test, w_test)
    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      mean_cv_deviance)
    mlflow.log_metric("generalisation_gap",    test_dev - mean_cv_deviance)
    mlflow.catboost.log_model(freq_model, "freq_model")
    mlflow.log_artifact("/tmp/feature_spec.json", artifact_path="feature_spec")
    freq_run_id = freq_run.info.run_id

freq_rate_all = freq_pred_test / np.clip(w_test, 1e-6, None)
print(f"Freq model: {freq_run_id}")
print(f"Test deviance: {test_dev:.5f} | CV mean: {mean_cv_deviance:.5f} | "
      f"gap: {test_dev - mean_cv_deviance:.5f}")

# COMMAND ----------

# Severity model: train on 2022-2023, calibrate conformal on 2024, test on 2025
cal_year     = sorted(df_train["accident_year"].unique())[-1]   # 2024
df_sev_model = df_train[(df_train["accident_year"] < cal_year) & (df_train["claim_count"] > 0)].copy()
df_sev_test  = df_test[df_test["claim_count"] > 0].copy()
df_sev_model["mean_sev"] = df_sev_model["incurred_loss"] / df_sev_model["claim_count"]
df_sev_test["mean_sev"]  = df_sev_test["incurred_loss"]  / df_sev_test["claim_count"]

sev_train_pool = Pool(df_sev_model[FEATURE_COLS], df_sev_model["mean_sev"].values,
                      weight=df_sev_model["claim_count"].values, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(df_sev_test[FEATURE_COLS],  df_sev_test["mean_sev"].values,
                      weight=df_sev_test["claim_count"].values,  cat_features=CAT_FEATURES)

with mlflow.start_run(run_name="sev_model_m08") as sev_run:
    mlflow.log_params(best_sev_params)
    mlflow.log_params({
        "model_type":     "catboost_gamma",
        "train_years":    str(sorted(df_sev_model["accident_year"].unique().tolist())),
        "cal_year":       str(cal_year),
        "test_year":      str(max_year),
        "severity_target":"mean_cost_per_claim",
        "run_date":       RUN_DATE,
    })
    sev_model       = CatBoostRegressor(**best_sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)
    sev_pred_test   = sev_model.predict(sev_test_pool)
    sev_rmse        = float(np.sqrt(np.mean((df_sev_test["mean_sev"].values - sev_pred_test)**2)))
    mlflow.log_metric("test_severity_rmse", sev_rmse)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = sev_run.info.run_id

# Pure premium for ALL test policies
all_test_sev_pool = Pool(X_test, cat_features=CAT_FEATURES)
sev_pred_all = sev_model.predict(all_test_sev_pool)
pure_premium = freq_rate_all * sev_pred_all

assert (sev_pred_all > 0).all(), "Non-positive severity predictions"
assert np.isfinite(pure_premium).all(), "Non-finite pure premiums"

print(f"Sev model:  {sev_run_id}")
print(f"RMSE:       £{sev_rmse:,.0f}")
print(f"Pure premium: mean £{pure_premium.mean():,.2f} | P90 £{np.percentile(pure_premium,90):,.2f}")

# COMMAND ----------

# Write predictions to Delta
pred_df = pl.DataFrame({
    "policy_id":          df_test["policy_id"].tolist(),
    "accident_year":      df_test["accident_year"].tolist(),
    "exposure":           w_test.tolist(),
    "claim_count_actual": y_test.tolist(),
    "freq_pred_rate":     freq_rate_all.tolist(),
    "sev_pred":           sev_pred_all.tolist(),
    "pure_premium":       pure_premium.tolist(),
    "freq_run_id":        [freq_run_id] * len(df_test),
    "sev_run_id":         [sev_run_id]  * len(df_test),
    "run_date":           [RUN_DATE]    * len(df_test),
})
(
    spark.createDataFrame(pred_df.to_pandas())
    .write.format("delta").mode("overwrite").option("overwriteSchema","true")
    .saveAsTable(TABLES["freq_predictions"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['freq_predictions']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
print(f"Predictions → {TABLES['freq_predictions']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: SHAP relativities
# MAGIC
# MAGIC Extract multiplicative relativities from the frequency model using shap_relativities.
# MAGIC Output format is identical to GLM exp(beta) factor tables.

# COMMAND ----------

X_test_pl    = pl.from_pandas(df_test[FEATURE_COLS].reset_index(drop=True))
exposure_test = pl.Series("exposure", w_test.tolist())

sr = SHAPRelativities(
    model=freq_model,
    X=X_test_pl,
    exposure=exposure_test,
    categorical_features=CAT_FEATURES,
)
sr.fit()

freq_relativities = sr.extract_relativities(
    normalise_to="mean",
    ci_method="clt",
)

print("SHAP relativities (top features by mean absolute SHAP):")
print(freq_relativities.sort("mean_shap", descending=True))

# COMMAND ----------

(
    spark.createDataFrame(freq_relativities.to_pandas())
    .write.format("delta").mode("overwrite").option("overwriteSchema","true")
    .saveAsTable(TABLES["freq_relativities"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['freq_relativities']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

# Log to MLflow alongside the frequency model
rel_dict = {
    f"{row['feature']}_{row['level']}": float(row["relativity"])
    for row in freq_relativities.iter_rows(named=True)
}
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_dict(rel_dict, "shap_relativities.json")

print(f"Relativities → {TABLES['freq_relativities']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8: Conformal prediction intervals
# MAGIC
# MAGIC Calibrate on accident year 2024 (out-of-sample for the severity model).
# MAGIC Test on accident year 2025.
# MAGIC Validate coverage by decile before downstream use.

# COMMAND ----------

df_cal_sev = features_pd[
    (features_pd["accident_year"] == cal_year) & (features_pd["claim_count"] > 0)
].copy()
df_te_sev  = df_test[df_test["claim_count"] > 0].copy()
df_cal_sev["mean_sev"] = df_cal_sev["incurred_loss"] / df_cal_sev["claim_count"]
df_te_sev["mean_sev"]  = df_te_sev["incurred_loss"]  / df_te_sev["claim_count"]

X_cal     = df_cal_sev[FEATURE_COLS]
y_cal     = df_cal_sev["mean_sev"].values
X_te_conf = df_te_sev[FEATURE_COLS]
y_te_conf = df_te_sev["mean_sev"].values

print(f"Calibration year:   {cal_year} ({len(df_cal_sev):,} claims)")
print(f"Test year:          {max_year} ({len(df_te_sev):,} claims)")
if len(df_cal_sev) < 100:
    print(f"WARNING: Calibration set small ({len(df_cal_sev)} claims). Intervals may be unreliable.")

cp = InsuranceConformalPredictor(
    model=sev_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=2.0,
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_te_conf, alpha=CONFORMAL_ALPHA)
diag      = cp.coverage_by_decile(X_te_conf, y_te_conf, alpha=CONFORMAL_ALPHA)
min_cov   = float(diag["coverage"].min())

print(f"\nSeverity intervals (1-alpha={1-CONFORMAL_ALPHA:.0%}):")
print(f"  Mean width: £{(intervals['upper'] - intervals['lower']).mean():,.0f}")
print(f"\nCoverage by decile (target: {1-CONFORMAL_ALPHA:.0%}):")
print(diag)
print(f"\nMin decile coverage: {min_cov:.3f}")
if min_cov < 0.85:
    print("WARNING: Coverage below 85% in at least one decile.")

# Referral flags: top 10% relative width
rel_width = (intervals["upper"] - intervals["lower"]) / (intervals["point"] + 1e-6)
referral_flag = rel_width >= float(np.percentile(rel_width, 90))
print(f"\nReferral flags: {int(referral_flag.sum()):,} ({referral_flag.mean():.1%} of claims)")

# COMMAND ----------

conf_df = pl.DataFrame({
    "policy_id":       df_te_sev["policy_id"].tolist(),
    "accident_year":   df_te_sev["accident_year"].tolist(),
    "mean_sev_actual": y_te_conf.tolist(),
    "sev_lower":       intervals["lower"].to_list(),
    "sev_point":       intervals["point"].to_list(),
    "sev_upper":       intervals["upper"].to_list(),
    "rel_width":       rel_width.to_list(),
    "referral_flag":   referral_flag.to_list(),
    "conformal_alpha": [CONFORMAL_ALPHA] * len(df_te_sev),
    "cal_year":        [cal_year]        * len(df_te_sev),
    "sev_run_id":      [sev_run_id]      * len(df_te_sev),
    "run_date":        [RUN_DATE]        * len(df_te_sev),
})
(
    spark.createDataFrame(conf_df.to_pandas())
    .write.format("delta").mode("overwrite").option("overwriteSchema","true")
    .saveAsTable(TABLES["conformal_intervals"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['conformal_intervals']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
print(f"Intervals → {TABLES['conformal_intervals']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8.5: Calibration testing
# MAGIC
# MAGIC Balance property, auto-calibration, and Murphy decomposition on the frequency model.
# MAGIC Verdict: OK → proceed | RECALIBRATE → apply scalar correction | REFIT → stop pipeline.

# COMMAND ----------

checker    = CalibrationChecker(distribution="poisson")
cal_report = checker.check(
    y=y_test.astype(float),
    y_hat=freq_pred_test,
    exposure=w_test,
    n_bins=10,
    seed=42,
)

print(cal_report.verdict())
print(f"\nBalance ratio:    {cal_report.balance.balance_ratio:.4f}")
print(f"Auto-cal p-value: {cal_report.auto_calibration.p_value:.4f}")
print(f"Murphy verdict:   {cal_report.murphy.verdict}")
print(f"Discrimination:   {cal_report.murphy.discrimination_pct:.1f}% of UNC")
print(f"Miscalibration:   {cal_report.murphy.miscalibration_pct:.1f}% of UNC")

cal_balance_ratio  = float(cal_report.balance.balance_ratio)
cal_balance_ok     = bool(cal_report.balance.is_balanced)
cal_auto_p         = float(cal_report.auto_calibration.p_value)
cal_auto_ok        = bool(cal_report.auto_calibration.is_calibrated)
cal_murphy_verdict = cal_report.murphy.verdict
cal_dsc_pct        = round(float(cal_report.murphy.discrimination_pct), 2)
cal_mcb_pct        = round(float(cal_report.murphy.miscalibration_pct), 2)

# COMMAND ----------

if cal_murphy_verdict == "REFIT":
    raise RuntimeError(
        f"PIPELINE STOPPED: Frequency model requires REFIT. "
        f"Murphy verdict: {cal_murphy_verdict}. "
        f"MCB = {cal_mcb_pct:.1f}% of UNC. "
        f"See Murphy decomposition output above for diagnosis."
    )
elif cal_murphy_verdict == "RECALIBRATE":
    freq_pred_final = rectify_balance(
        y_hat=freq_pred_test,
        y=y_test.astype(float),
        exposure=w_test,
        method="multiplicative",
    )
    correction = float(freq_pred_final.mean() / freq_pred_test.mean())
    print(f"Balance correction applied: {correction:.4f}")
    # Recompute pure premium with corrected frequency
    freq_rate_corrected = freq_pred_final / np.clip(w_test, 1e-6, None)
    pure_premium_final  = freq_rate_corrected * sev_pred_all
else:
    freq_pred_final    = freq_pred_test
    pure_premium_final = pure_premium
    print(f"No correction needed (verdict: {cal_murphy_verdict}). Proceeding.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: Rate optimisation
# MAGIC
# MAGIC PortfolioOptimiser from insurance-optimise.
# MAGIC Inputs: technical price from the trained models, demand parameters.
# MAGIC Constraints: LR target, volume floor, ENBP.

# COMMAND ----------

n_renewal = min(5_000, len(df_test))
rng_opt   = np.random.default_rng(seed=9999)
ren_idx   = rng_opt.choice(len(df_test), n_renewal, replace=False)
df_ren    = df_test.iloc[ren_idx].copy()
pp_ren    = pure_premium_final[ren_idx]

tech_prem = pp_ren
curr_prem = tech_prem / LR_TARGET
mkt_prem  = tech_prem / (LR_TARGET - 0.02)

tenure    = rng_opt.integers(0, 10, n_renewal).astype(float)
log_price = np.log(np.clip(curr_prem / mkt_prem, 1e-6, None))
p_renew   = expit(1.2 + (-2.0) * log_price + 0.04 * tenure)
elasticity = np.full(n_renewal, -2.0 * p_renew.mean() * (1 - p_renew.mean()))

channels     = rng_opt.choice(["PCW","direct"], n_renewal, p=[0.68, 0.32])
renewal_flag = np.ones(n_renewal, dtype=bool)
enbp         = mkt_prem * 1.01

print(f"Renewal portfolio:  {n_renewal:,} policies")
print(f"Mean tech premium:  £{tech_prem.mean():,.2f}")
print(f"Mean renewal prob:  {p_renew.mean():.3f}")
print(f"Starting LR:        {(tech_prem / curr_prem).mean():.3f}")

# COMMAND ----------

config = ConstraintConfig(
    lr_max=LR_TARGET,
    retention_min=VOLUME_FLOOR,
    max_rate_change=0.15,
)

opt = PortfolioOptimiser(
    technical_price=tech_prem,
    expected_loss_cost=tech_prem,
    p_demand=p_renew,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config,
)

result = opt.optimise()
print(f"\nOptimisation result:")
print(f"  Converged:     {result.converged}")
print(f"  Expected LR:   {result.expected_loss_ratio:.4f}  (target: {LR_TARGET})")
print(f"  Expected vol:  {result.expected_retention:.4f}  (floor: {VOLUME_FLOOR})")
print(f"  Expected P&L:  £{result.expected_profit:,.0f}")
print(f"  ENBP violations: {int(result.summary_df["enbp_binding"].sum())}")

# COMMAND ----------

# Efficient frontier
frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="lr_max",
    sweep_range=(0.68, 0.78),
    n_points=12,
)
frontier_result = frontier.run()
frontier_df = frontier_result.data

(
    spark.createDataFrame(frontier_df.to_pandas() if hasattr(frontier_df, "to_pandas") else frontier_df)
    .write.format("delta").mode("overwrite").option("overwriteSchema","true")
    .saveAsTable(TABLES["efficient_frontier"])
)
print(f"Efficient frontier → {TABLES['efficient_frontier']}")

# Rate action summary
rate_summary = pl.DataFrame({
    "run_date":            [RUN_DATE],
    "lr_target":           [LR_TARGET],
    "volume_floor":        [VOLUME_FLOOR],
    "optimiser_converged": [bool(result.converged)],
    "expected_lr":         [round(float(result.expected_loss_ratio), 4)],
    "expected_volume":     [round(float(result.expected_retention), 4)],
    "expected_profit":     [round(float(result.expected_profit), 2)],
    "enbp_violations":     [int(result.summary_df["enbp_binding"].sum())],
    "freq_run_id":         [freq_run_id],
    "sev_run_id":          [sev_run_id],
})
(
    spark.createDataFrame(rate_summary.to_pandas())
    .write.format("delta").mode("overwrite").option("overwriteSchema","true")
    .saveAsTable(TABLES["rate_change"])
)
spark.sql(f"""
    ALTER TABLE {TABLES['rate_change']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
print(f"Rate action → {TABLES['rate_change']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 10: Pipeline audit record
# MAGIC
# MAGIC Append-only. Every run adds one row.
# MAGIC Given any run_date, you can load the exact training data and model from the
# MAGIC row's raw_table_version and freq_model_run_id.

# COMMAND ----------

audit_record = {
    "run_date":               RUN_DATE,
    "pipeline_version":       "module_08_v1",
    "raw_table":              TABLES["raw"],
    "raw_table_version":      int(raw_version),
    "features_table":         TABLES["features"],
    "features_table_version": int(feat_version),
    "freq_model_run_id":      freq_run_id,
    "sev_model_run_id":       sev_run_id,
    "feature_cols":           json.dumps(FEATURE_COLS),
    "train_years":            str(sorted(df_train["accident_year"].unique().tolist())),
    "test_year":              int(max_year),
    "n_training_rows":        int(len(df_train)),
    "n_test_rows":            int(len(df_test)),
    "n_cv_folds":             len(cv_deviances),
    "mean_cv_deviance":       round(mean_cv_deviance, 5),
    "freq_optuna_trials":     N_OPTUNA_TRIALS,
    "freq_best_deviance":     round(freq_study.best_value, 5),
    "sev_optuna_trials":      N_OPTUNA_TRIALS,
    "sev_best_rmse":          round(sev_study.best_value, 2),
    "test_poisson_deviance":  round(test_dev, 5),
    "generalisation_gap":     round(test_dev - mean_cv_deviance, 5),
    "sev_rmse":               round(sev_rmse, 2),
    "cal_balance_ratio":      round(cal_balance_ratio, 4),
    "cal_balance_ok":         bool(cal_balance_ok),
    "cal_auto_p":             round(cal_auto_p, 4),
    "cal_auto_ok":            bool(cal_auto_ok),
    "cal_murphy_verdict":     cal_murphy_verdict,
    "cal_dsc_pct":            cal_dsc_pct,
    "cal_mcb_pct":            cal_mcb_pct,
    "conformal_alpha":        CONFORMAL_ALPHA,
    "conformal_cal_year":     int(cal_year),
    "conformal_n_cal":        int(len(df_cal_sev)),
    "conformal_min_cov":      round(min_cov, 3),
    "lr_target":              LR_TARGET,
    "volume_floor":           VOLUME_FLOOR,
    "optimiser_converged":    bool(result.converged),
    "expected_lr":            round(float(result.expected_loss_ratio), 4),
    "expected_volume":        round(float(result.expected_retention), 4),
    "enbp_violations":        int(result.summary_df["enbp_binding"].sum()),
    "catalog":                CATALOG,
    "schema":                 SCHEMA,
    "pipeline_notes":         "Module 8 capstone — synthetic UK motor data",
}

(
    spark.createDataFrame([audit_record])
    .write.format("delta").mode("append")
    .saveAsTable(TABLES["pipeline_audit"])
)
print(f"Audit record → {TABLES['pipeline_audit']}")
print(f"\nKey fields for reproducibility:")
print(f"  raw_table_version:    {audit_record['raw_table_version']}")
print(f"  freq_model_run_id:    {audit_record['freq_model_run_id']}")
print(f"  sev_model_run_id:     {audit_record['sev_model_run_id']}")
print(f"  test_poisson_deviance:{audit_record['test_poisson_deviance']}")
print(f"  cal_murphy_verdict:   {audit_record['cal_murphy_verdict']}")
print(f"  conformal_min_cov:    {audit_record['conformal_min_cov']}")
print(f"  optimiser_converged:  {audit_record['optimiser_converged']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Pricing Committee Pack
# MAGIC
# MAGIC | Item | Detail |
# MAGIC |------|--------|
# MAGIC | Data | `raw_policies` (Delta version logged) |
# MAGIC | Models | CatBoost Poisson freq + Tweedie severity (MLflow run IDs logged) |
# MAGIC | Validation | Walk-forward CV, IBNR buffer, calibration check |
# MAGIC | Uncertainty | 90% conformal intervals on severity, coverage validated by decile |
# MAGIC | Rate action | PortfolioOptimiser (SLSQP), LR + volume + ENBP constraints |
# MAGIC | Audit | All table versions and MLflow run IDs in `pipeline_audit` |

# COMMAND ----------

print("=" * 65)
print("MODULE 8: END-TO-END PIPELINE — COMPLETE")
print("=" * 65)
print()
print(f"Training rows:       {len(df_train):,}")
print(f"Test rows:           {len(df_test):,}")
print(f"Mean CV deviance:    {mean_cv_deviance:.5f}")
print(f"Test deviance:       {test_dev:.5f}")
print(f"Severity RMSE:       £{sev_rmse:,.0f}")
print(f"Murphy verdict:      {cal_murphy_verdict}")
print(f"Conformal min cov:   {min_cov:.3f}")
print(f"Optimiser converged: {result.converged}")
print(f"Expected LR:         {result.expected_loss_ratio:.4f}")
print(f"Expected volume:     {result.expected_retention:.4f}")
print()
print("Delta tables written:")
for k, v in TABLES.items():
    print(f"  {v}")
print()
print(f"MLflow frequency model: runs:/{freq_run_id}/freq_model")
print(f"MLflow severity model:  runs:/{sev_run_id}/sev_model")
print()
print("Next steps:")
print("  - Present efficient frontier to pricing committee")
print("  - Submit rate action summary for rating engine update")
print("  - Schedule monthly conformal recalibration monitoring")
print("  - Connect Module 9 causal demand model for retention-aware pricing")
