# Databricks notebook source
# MAGIC %md
# MAGIC # Module 6: Credibility & Bayesian Pricing - The Thin-Cell Problem
# MAGIC
# MAGIC Full workflow on synthetic UK motor data. Runs end-to-end on a single-node
# MAGIC Databricks cluster (DBR 14.x ML runtime recommended, 4+ cores).
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates a synthetic UK motor portfolio with area-level variation and deliberately thin cells
# MAGIC 2. Computes Bühlmann-Straub credibility estimates using a self-contained NumPy implementation
# MAGIC 3. Fits a hierarchical Bayesian frequency model directly in PyMC (no wrapper library)
# MAGIC 4. Produces the shrinkage plot - observed rate vs credibility-weighted estimate
# MAGIC 5. Compares classical Z to Bayesian posterior shrinkage per segment
# MAGIC 6. Checks convergence (R-hat, ESS, divergences) including a separate ESS check for variance components
# MAGIC 7. Stores results in Delta tables with MLflow tracking
# MAGIC
# MAGIC **Runtime:** Classical credibility: < 1 minute. Bayesian MCMC: 5–15 minutes on a 4-core cluster.
# MAGIC
# MAGIC **Dependencies:** pymc, arviz, polars, numpy, matplotlib, mlflow — all available on PyPI.
# MAGIC No private GitHub repositories required.

# COMMAND ----------

# MAGIC %pip install pymc arviz polars --quiet
# MAGIC # numpy, matplotlib, mlflow are pre-installed on Databricks ML runtime

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import multiprocessing
from datetime import date

import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import mlflow

print("Libraries imported successfully.")
print(f"Available CPU cores: {multiprocessing.cpu_count()}")
print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic data
# MAGIC
# MAGIC We generate a UK motor portfolio directly in NumPy/Polars, with:
# MAGIC - Most postcode districts have moderate exposure (200–2,000 policy-years)
# MAGIC - 30% of districts are deliberately thin (< 50 policy-years)
# MAGIC - True area-level frequency relativities vary from 0.7x to 2.1x the portfolio mean
# MAGIC - Claim frequency follows a Poisson process at 6.8% per annum base rate
# MAGIC
# MAGIC This setup ensures we have a realistic mix of data-rich and data-poor segments,
# MAGIC so the shrinkage effect of credibility is visible.

# COMMAND ----------

RNG = np.random.default_rng(42)

N_DISTRICTS = 120
THIN_FRACTION = 0.30        # 30% of districts are thin (< 50 earned years)
BASE_FREQUENCY = 0.068      # 6.8% per annum
N_YEARS = 5                 # accident years
TOTAL_POLICIES = 80_000

# --- True district log-relativities ---
# Drawn from Normal(0, 0.4) on the log scale, so most districts are within
# ~1.5x of the base rate, with a few genuinely extreme districts at the tails.
true_log_rels = RNG.normal(0, 0.4, N_DISTRICTS)
district_names = [f"D{i:03d}" for i in range(N_DISTRICTS)]
TRUE_AREA_PARAMS = dict(zip(district_names, true_log_rels))

# --- Exposure distribution ---
# 30% thin districts (< 50 earned years total over N_YEARS)
# 70% normal districts (200–2,000 earned years total)
n_thin = int(N_DISTRICTS * THIN_FRACTION)
n_normal = N_DISTRICTS - n_thin

total_exposure_per_district = np.concatenate([
    RNG.uniform(5, 45, n_thin),          # thin: 5–45 earned years total
    RNG.uniform(200, 2000, n_normal),    # normal: 200–2,000 earned years total
])
RNG.shuffle(total_exposure_per_district)

# --- Build policy-level data as district × year sufficient statistics ---
rows = []
for d_idx, (d_name, log_rel) in enumerate(zip(district_names, true_log_rels)):
    true_freq = BASE_FREQUENCY * np.exp(log_rel)
    total_exp = total_exposure_per_district[d_idx]
    for yr in range(2018, 2018 + N_YEARS):
        year_exp = RNG.uniform(0.1, 0.4) * total_exp   # uneven year-to-year exposure
        year_exp = max(year_exp, 0.5)                   # at least half a year
        year_claims = RNG.poisson(true_freq * year_exp)
        rows.append({
            "postcode_district": d_name,
            "accident_year": yr,
            "earned_years": round(year_exp, 2),
            "claim_count": int(year_claims),
            "incurred": float(year_claims * RNG.gamma(shape=2.5, scale=3000)),
        })

df = pl.DataFrame(rows)

print(f"Portfolio: {len(df):,} district-year rows")
print(f"Exposure:  {df['earned_years'].sum():.0f} earned years")
print(f"Claims:    {df['claim_count'].sum():,}")
print(f"Raw frequency: {df['claim_count'].sum() / df['earned_years'].sum():.4f} per earned year")
print()

# District exposure distribution
dist_exp = df.group_by("postcode_district").agg(
    pl.col("earned_years").sum().alias("total_earned_years")
)
exp_vals = dist_exp["total_earned_years"]
print("District exposure distribution (quartiles):")
print(f"  count  {len(exp_vals)}")
print(f"  mean   {exp_vals.mean():.1f}")
print(f"  std    {exp_vals.std():.1f}")
print(f"  min    {exp_vals.min():.1f}")
print(f"  25%    {exp_vals.quantile(0.25):.1f}")
print(f"  50%    {exp_vals.quantile(0.50):.1f}")
print(f"  75%    {exp_vals.quantile(0.75):.1f}")
print(f"  max    {exp_vals.max():.1f}")
print()
print(f"Districts with < 50 earned years: {(exp_vals < 50).sum()} / {len(exp_vals)}")
print(f"Districts with < 10 earned years: {(exp_vals < 10).sum()} / {len(exp_vals)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### True area parameters
# MAGIC
# MAGIC Because we generated the data, we know the true frequency relativities.
# MAGIC After fitting, we can check whether our credibility estimates recover them better
# MAGIC than naive observed rates do.

# COMMAND ----------

print("True area-level frequency relativities (sample):")
print(f"{'District':<20} {'True log-rel':>14} {'True relativity':>16}")
print("-" * 52)
for district, log_rel in sorted(TRUE_AREA_PARAMS.items())[:15]:
    print(f"{district:<20} {log_rel:>14.3f} {np.exp(log_rel):>16.3f}")

print(f"\n... ({len(TRUE_AREA_PARAMS)} districts total)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate to district level
# MAGIC
# MAGIC Bühlmann-Straub requires the multi-period (district × year) structure.
# MAGIC The Bayesian model works on district-level totals (sufficient statistics for Poisson).
# MAGIC
# MAGIC We filter rows with earned_years < 0.5 to exclude near-zero exposure cells
# MAGIC that would produce near-infinite frequencies. Using `.clip(lower=1e-6)` on
# MAGIC exposure would mask bad data; explicit filtering removes it.

# COMMAND ----------

# District × year sufficient statistics - needed for B-S (multi-period structure)
dist_year = (
    df.group_by(["postcode_district", "accident_year"])
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
    .with_columns([
        (pl.col("claims") / pl.col("earned_years")).alias("claim_frequency"),
    ])
    .sort(["postcode_district", "accident_year"])
)

print(f"District × year rows: {len(dist_year):,}")
print(f"Years in data: {sorted(dist_year['accident_year'].unique().to_list())}")
print()
print("Sample:")
print(dist_year.head(12))

# COMMAND ----------

# District-level totals - sufficient statistics for the Bayesian Poisson model
dist_totals = (
    df.group_by("postcode_district")
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
    .with_columns([
        (pl.col("claims") / pl.col("earned_years")).alias("observed_rate"),
    ])
    .sort("postcode_district")
)

print(f"District-level summary: {len(dist_totals)} districts")
print(dist_totals.select(["postcode_district", "claims", "earned_years", "observed_rate"]).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bühlmann-Straub credibility
# MAGIC
# MAGIC Self-contained implementation in NumPy/Polars — no external library required.
# MAGIC
# MAGIC Key outputs:
# MAGIC - `grand_mean`: the portfolio collective mean (mu_hat)
# MAGIC - `v_hat`: Expected Process Variance (EPV) - within-district year-to-year variation
# MAGIC - `a_hat`: Variance of Hypothetical Means (VHM) - between-district heterogeneity
# MAGIC - `k`: Bühlmann's K = v/a - how much exposure you need for Z = 0.5
# MAGIC - `results`: per-district credibility factors and blended estimates
# MAGIC
# MAGIC We use log_transform=True because we are working in a multiplicative (Poisson log-link)
# MAGIC framework. Applying B-S in rate space and then converting to relativities introduces
# MAGIC a Jensen's inequality bias — the log of the expected value does not equal the expected
# MAGIC value of the log. log_transform=True applies the blending in log-rate space, correcting
# MAGIC for this bias. For most UK motor portfolios the correction is small in absolute terms;
# MAGIC for extreme relativities (thin cells far from the mean) it is material.

# COMMAND ----------

def buhlmann_straub(
    data: pl.DataFrame,
    group_col: str,
    value_col: str,
    weight_col: str,
    log_transform: bool = True,
) -> dict:
    """
    Bühlmann-Straub credibility estimator.

    Parameters
    ----------
    data : Polars DataFrame with one row per (group, period).
    group_col : column identifying the group (e.g. postcode_district).
    value_col : observed loss rate per unit of exposure (e.g. claim_frequency).
    weight_col : exposure weight (e.g. earned_years or claim_count for severity).
    log_transform : if True, apply B-S in log-rate space to avoid the
        Jensen's inequality bias in multiplicative (log-link) frameworks.
        Set False for additive models or severity data.

    Returns
    -------
    dict with keys:
        grand_mean, v_hat, a_hat, a_hat_raw, k, results (Polars DataFrame)
    """
    if log_transform:
        data = data.with_columns(
            pl.col(value_col).clip(lower_bound=1e-9).log().alias("_y")
        )
        y_col = "_y"
    else:
        y_col = value_col

    groups = data[group_col].unique().sort().to_list()
    r = len(groups)

    # Per-group sufficient statistics
    group_agg = (
        data.group_by(group_col)
        .agg([
            pl.col(weight_col).sum().alias("w_i"),
            (
                (pl.col(y_col) * pl.col(weight_col)).sum()
                / pl.col(weight_col).sum()
            ).alias("x_bar_i"),
            pl.col(weight_col).count().alias("T_i"),
        ])
        .sort(group_col)
    )

    # Filter out single-period groups for EPV calculation.
    # Groups with T_i = 1 contribute 0 to the EPV numerator (no within-group
    # deviation) but -1 to the denominator, incorrectly reducing the effective
    # sample size and biasing v_hat upward.
    group_agg_epv = group_agg.filter(pl.col("T_i") > 1)

    w_i = group_agg["w_i"].to_numpy()
    x_bar_i = group_agg["x_bar_i"].to_numpy()
    w = w_i.sum()

    # --- Collective mean ---
    mu_hat = (w_i * x_bar_i).sum() / w

    # --- EPV (v_hat): within-group variance ---
    def _epv_num_for_group(grp_name: str) -> float:
        grp = data.filter(pl.col(group_col) == grp_name)
        if grp.height <= 1:
            return 0.0
        x_bar = float(
            (grp[y_col] * grp[weight_col]).sum() / grp[weight_col].sum()
        )
        resid_sq = (grp[y_col].to_numpy() - x_bar) ** 2
        return float((resid_sq * grp[weight_col].to_numpy()).sum())

    epv_groups = group_agg_epv[group_col].to_list()
    epv_num = sum(_epv_num_for_group(g) for g in epv_groups)
    epv_den = float(group_agg_epv["T_i"].sum() - len(epv_groups))
    v_hat = epv_num / epv_den if epv_den > 0 else 0.0

    # --- VHM (a_hat): between-group variance ---
    c = w - (w_i ** 2).sum() / w
    s_sq = (w_i * (x_bar_i - mu_hat) ** 2).sum()
    a_hat_raw = (s_sq - (r - 1) * v_hat) / c
    # a_hat can be negative when within-group variance dominates.
    # Truncate at zero: all Z_i = 0, every group gets the portfolio mean.
    # A substantially negative a_hat_raw is a diagnostic — see tutorial for
    # the implications.
    a_hat = max(a_hat_raw, 0.0)

    # --- Credibility factors and estimates ---
    k = v_hat / a_hat if a_hat > 0 else np.inf
    z_i = w_i / (w_i + k) if np.isfinite(k) else np.zeros(r)
    cred_est_y = z_i * x_bar_i + (1 - z_i) * mu_hat

    if log_transform:
        grand_mean = np.exp(mu_hat)
        cred_est = np.exp(cred_est_y)
        obs_mean = np.exp(x_bar_i)
    else:
        grand_mean = mu_hat
        cred_est = cred_est_y
        obs_mean = x_bar_i

    results = pl.DataFrame({
        group_col: group_agg[group_col].to_list(),
        "exposure": w_i.tolist(),
        "obs_mean": obs_mean.tolist(),
        "Z": z_i.tolist(),
        "credibility_estimate": cred_est.tolist(),
    })

    return {
        "grand_mean": grand_mean,
        "v_hat": v_hat,
        "a_hat": a_hat,
        "a_hat_raw": a_hat_raw,
        "k": k,
        "results": results,
    }

# COMMAND ----------

bs = buhlmann_straub(
    data=dist_year,
    group_col="postcode_district",
    value_col="claim_frequency",
    weight_col="earned_years",
    log_transform=True,
)

print("Bühlmann-Straub structural parameters:")
print(f"  Grand mean (mu):  {bs['grand_mean']:.5f}  ({bs['grand_mean'] * 100:.3f}% per year)")
print(f"  EPV (v):          {bs['v_hat']:.7f}  (within-district variance)")
print(f"  VHM (a):          {bs['a_hat']:.7f}  (between-district variance)")
print(f"  a_hat (raw, before truncation): {bs['a_hat_raw']:.7f}")
if bs['a_hat_raw'] < 0:
    print("  WARNING: a_hat was negative before truncation. The data cannot")
    print("  distinguish district effects. All Z = 0; check grouping structure.")
print(f"  K = v/a:          {bs['k']:.1f}")
print(f"  Implied half-credibility exposure: {bs['k']:.0f} earned years")
print()
# Z = w/(w+K); solve for w at Z = 0.50, 0.67, 0.90
print("Interpretation of K (Z = w/(w+K)):")
print(f"  A district needs {bs['k']:.0f} earned years for Z = 0.50")
print(f"  A district needs {2 * bs['k']:.0f} earned years for Z = 0.67")
print(f"  A district needs {9 * bs['k']:.0f} earned years for Z = 0.90")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Credibility factor distribution

# COMMAND ----------

bs_results = bs["results"]  # already a Polars DataFrame

print(f"Credibility factor distribution across {len(bs_results)} districts:")
z_vals = bs_results["Z"]
print(f"  count  {len(z_vals)}")
print(f"  mean   {z_vals.mean():.3f}")
print(f"  std    {z_vals.std():.3f}")
print(f"  min    {z_vals.min():.3f}")
print(f"  25%    {z_vals.quantile(0.25):.3f}")
print(f"  50%    {z_vals.quantile(0.50):.3f}")
print(f"  75%    {z_vals.quantile(0.75):.3f}")
print(f"  max    {z_vals.max():.3f}")
print()
print("Districts with Z < 0.10 (nearly all prior):")
thin = bs_results.filter(pl.col("Z") < 0.10).sort("Z")
print(thin.select(["postcode_district", "exposure", "obs_mean", "Z", "credibility_estimate"]).head(10))
print()
print("Districts with Z > 0.80 (mostly own experience):")
thick = bs_results.filter(pl.col("Z") > 0.80).sort("Z", descending=True)
print(thick.select(["postcode_district", "exposure", "obs_mean", "Z", "credibility_estimate"]).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare B-S estimates to true relativities

# COMMAND ----------

bs_vs_true = bs_results.with_columns([
    pl.col("postcode_district").map_elements(
        lambda d: TRUE_AREA_PARAMS.get(d, 0.0), return_dtype=pl.Float64
    ).alias("true_log_rel"),
])
bs_vs_true = bs_vs_true.with_columns([
    (bs["grand_mean"] * (pl.col("true_log_rel").exp())).alias("true_rate"),
])

mse_observed = ((bs_vs_true["obs_mean"] - bs_vs_true["true_rate"]) ** 2).mean()
mse_bs = ((bs_vs_true["credibility_estimate"] - bs_vs_true["true_rate"]) ** 2).mean()

mape_observed = ((bs_vs_true["obs_mean"] - bs_vs_true["true_rate"]).abs() / bs_vs_true["true_rate"]).mean()
mape_bs = ((bs_vs_true["credibility_estimate"] - bs_vs_true["true_rate"]).abs() / bs_vs_true["true_rate"]).mean()

print("Accuracy comparison (lower is better):")
print(f"  MSE  - Observed rate vs true rate:             {mse_observed:.8f}")
print(f"  MSE  - B-S credibility estimate vs true rate:  {mse_bs:.8f}")
print(f"  MSE reduction from B-S: {(1 - mse_bs / mse_observed) * 100:.1f}%")
print()
print(f"  MAPE - Observed rate vs true rate:             {mape_observed * 100:.2f}%")
print(f"  MAPE - B-S credibility estimate vs true rate:  {mape_bs * 100:.2f}%")
print(f"  MAPE reduction from B-S: {(1 - mape_bs / mape_observed) * 100:.1f}%")
print()
print("Note: MSE gives disproportionate weight to thin cells with high rate volatility.")
print("MAPE is more informative about typical accuracy across the portfolio.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hierarchical Bayesian frequency model
# MAGIC
# MAGIC We fit a Poisson hierarchical model directly in PyMC — no wrapper library.
# MAGIC The model structure is:
# MAGIC
# MAGIC ```
# MAGIC claims_i | lambda_i  ~  Poisson(lambda_i * exposure_i)
# MAGIC log(lambda_i) = alpha + u_district[i]
# MAGIC u_district[k] = u_raw[k] * sigma_district   (non-centered)
# MAGIC u_raw[k]      ~ Normal(0, 1)
# MAGIC sigma_district ~ HalfNormal(0.3)
# MAGIC alpha          ~ Normal(log(mu_portfolio), 0.5)
# MAGIC ```
# MAGIC
# MAGIC **Non-centered parameterization** is mandatory. The centered form
# MAGIC `u_district ~ Normal(0, sigma_district)` creates funnel geometry when
# MAGIC sigma_district is near zero. The sampler under-samples the funnel neck,
# MAGIC biasing variance components toward zero — your credibility factors will be
# MAGIC too low without obvious warning. Non-centered decouples u_raw from sigma,
# MAGIC eliminating the funnel.
# MAGIC
# MAGIC **Prior justification:**
# MAGIC - alpha SD=0.5 (log scale): allows the intercept to range from ~0.6x to ~1.65x
# MAGIC   the observed portfolio mean at ±1 SD. For a UK motor book at 6-8% frequency,
# MAGIC   this permits 3.7% to 13% at ±1 SD - not unduly informative.
# MAGIC - sigma_district HalfNormal(0.3): places most mass below ~0.6 log points,
# MAGIC   implying most districts fall within roughly 0.5x-2.0x the portfolio mean.
# MAGIC   Widen to HalfNormal(0.5) or HalfNormal(0.7) for factor types with broader
# MAGIC   expected ranges (e.g. vehicle group effects which can reach 3-4x).

# COMMAND ----------

n_chains = min(4, multiprocessing.cpu_count())
print(f"Fitting hierarchical model with {n_chains} chains...")

# Encode districts as integer indices for PyMC
districts_sorted = dist_totals["postcode_district"].sort().to_list()
district_to_idx = {d: i for i, d in enumerate(districts_sorted)}
n_districts_model = len(districts_sorted)

district_idx_arr = np.array([
    district_to_idx[d] for d in dist_totals["postcode_district"].to_list()
])
claims_arr   = dist_totals["claims"].to_numpy()
exposure_arr = dist_totals["earned_years"].to_numpy()

# Portfolio log-rate for prior centre
log_mu_portfolio = np.log(claims_arr.sum() / exposure_arr.sum())

coords = {"district": districts_sorted}

mlflow.set_experiment("/pricing/credibility-bayesian/module06")

with mlflow.start_run(run_name="hierarchical_frequency_v1"):

    with pm.Model(coords=coords) as hierarchical_model:

        # Global intercept
        alpha = pm.Normal("alpha", mu=log_mu_portfolio, sigma=0.5)

        # Between-district variance (log scale)
        sigma_district = pm.HalfNormal("sigma_district", sigma=0.3)

        # Non-centered parameterization: u_district_raw is N(0,1),
        # independent of sigma_district. No funnel geometry.
        u_district_raw = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
        u_district = pm.Deterministic(
            "u_district", u_district_raw * sigma_district, dims="district"
        )

        # Log-rate per district
        log_lambda = alpha + u_district[district_idx_arr]

        # Poisson likelihood with exposure offset
        claims_obs = pm.Poisson(
            "claims_obs",
            mu=pm.math.exp(log_lambda) * exposure_arr,
            observed=claims_arr,
        )

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=n_chains,
            cores=n_chains,
            target_accept=0.90,
            return_inferencedata=True,
            random_seed=42,
        )

    print("Fit complete.")

    # ---- Convergence diagnostics ----
    rhat = az.rhat(trace)
    ess_bulk = az.ess(trace, method="bulk")

    max_rhat = float(rhat.max().to_array().max())
    min_ess = float(ess_bulk.min().to_array().min())
    n_div = int(trace.sample_stats["diverging"].sum())

    # Variance components need higher ESS than the global minimum.
    # sigma_district drives the credibility factors for all districts.
    # Underpowered sigma estimation is the most common way Bayesian credibility
    # factors are wrong without the model appearing to fail convergence.
    sigma_ess = float(ess_bulk["sigma_district"].min())

    print()
    print("Convergence diagnostics:")
    print(f"  Max R-hat:              {max_rhat:.4f}  ({'OK' if max_rhat < 1.01 else 'INVESTIGATE'})")
    print(f"  Min ESS (bulk):         {min_ess:.0f}  ({'OK' if min_ess > 400 else 'INVESTIGATE'})")
    print(f"  sigma_district ESS:     {sigma_ess:.0f}  ({'OK' if sigma_ess > 1000 else 'INVESTIGATE - increase draws'})")
    print(f"  Divergences:            {n_div}  ({'OK' if n_div == 0 else 'INVESTIGATE'})")

    # ---- Variance components ----
    sigma_district_mean = float(trace.posterior["sigma_district"].mean())
    alpha_mean = float(trace.posterior["alpha"].mean())
    grand_mean_bayes = np.exp(alpha_mean)

    print()
    print("Variance components:")
    print(f"  sigma_district: {sigma_district_mean:.4f}  (log-scale between-district SD)")
    print(f"  Implied between-district range (±1 SD): "
          f"[{np.exp(-sigma_district_mean):.3f}, {np.exp(sigma_district_mean):.3f}]")
    print(f"  Grand mean (posterior): {grand_mean_bayes:.5f}")

    # ---- MLflow logging ----
    mlflow.log_metric("max_rhat", max_rhat)
    mlflow.log_metric("min_ess_bulk", min_ess)
    mlflow.log_metric("sigma_ess", sigma_ess)
    mlflow.log_metric("n_divergences", n_div)
    mlflow.log_metric("n_segments", n_districts_model)
    mlflow.log_metric("grand_mean", grand_mean_bayes)
    mlflow.log_metric("sigma_district", sigma_district_mean)

    trace.to_netcdf("/tmp/posteriors_module06.nc")
    mlflow.log_artifact("/tmp/posteriors_module06.nc", "posteriors")

    print()
    print("Run logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract posterior results

# COMMAND ----------

# Posterior mean log-rate per district
u_post_mean = trace.posterior["u_district"].mean(dim=("chain", "draw")).values  # shape: (n_districts,)

log_rate_post = alpha_mean + u_post_mean
posterior_mean_rate = np.exp(log_rate_post)

# 90% credible intervals from posterior samples
log_rate_samples = (
    trace.posterior["alpha"].values[:, :, None]     # (chains, draws, 1)
    + trace.posterior["u_district"].values           # (chains, draws, n_districts)
)  # shape: (chains, draws, n_districts)
log_rate_flat = log_rate_samples.reshape(-1, n_districts_model)   # (chains*draws, n_districts)

lower_90 = np.exp(np.percentile(log_rate_flat, 5, axis=0))
upper_90 = np.exp(np.percentile(log_rate_flat, 95, axis=0))
posterior_sd = np.exp(log_rate_flat).std(axis=0)

observed_rate_arr = claims_arr / exposure_arr

# Approximate credibility factor Z from the shrinkage plot:
# Z_i = 1 - |posterior_mean - grand_mean| / |observed_rate - grand_mean|
# This is zero for fully-pooled estimates and one for unpooled estimates.
# It is an approximation for non-conjugate models but gives the right intuition.
z_approx = np.where(
    np.abs(observed_rate_arr - grand_mean_bayes) > 1e-9,
    1.0 - np.abs(posterior_mean_rate - grand_mean_bayes)
         / np.abs(observed_rate_arr - grand_mean_bayes),
    1.0,
).clip(0, 1)

results = pl.DataFrame({
    "postcode_district": districts_sorted,
    "claims": claims_arr.tolist(),
    "earned_years": exposure_arr.tolist(),
    "observed_rate": observed_rate_arr.tolist(),
    "posterior_mean": posterior_mean_rate.tolist(),
    "posterior_sd": posterior_sd.tolist(),
    "lower_90": lower_90.tolist(),
    "upper_90": upper_90.tolist(),
    "credibility_factor": z_approx.tolist(),
})

print("Bayesian results (sample):")
print(results.select([
    "postcode_district", "claims", "earned_years",
    "observed_rate", "posterior_mean", "posterior_sd",
    "credibility_factor", "lower_90", "upper_90"
]).head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The shrinkage plot
# MAGIC
# MAGIC This is the key diagnostic. We plot observed rates against posterior means.
# MAGIC - Points near the 45° line: dense segments trusted by the model
# MAGIC - Points near the horizontal (grand mean) line: thin segments shrunk toward the portfolio average
# MAGIC - Colour = approximate credibility factor Z (green = high Z, red = low Z)
# MAGIC - Point size = log exposure

# COMMAND ----------

log_exposure = np.log1p(results["earned_years"].to_numpy())
sizes = 20 + 100 * (log_exposure - log_exposure.min()) / (log_exposure.max() - log_exposure.min())

fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    results["observed_rate"].to_numpy(),
    results["posterior_mean"].to_numpy(),
    s=sizes,
    alpha=0.65,
    c=results["credibility_factor"].to_numpy(),
    cmap="RdYlGn",
    vmin=0, vmax=1,
    edgecolors="none",
    zorder=3,
)

# 45-degree line (no shrinkage)
obs_arr = results["observed_rate"].to_numpy()
rate_range = [obs_arr.min() * 0.9, obs_arr.max() * 1.1]
ax.plot(rate_range, rate_range, "k--", alpha=0.25, lw=1.5, label="Observed = estimate (no shrinkage)")

# Grand mean line
ax.axhline(grand_mean_bayes, color="steelblue", linestyle=":", alpha=0.6, lw=1.5,
           label=f"Grand mean = {grand_mean_bayes:.4f}")

plt.colorbar(sc, label="Approximate credibility factor Z")
ax.set_xlabel("Observed claim frequency")
ax.set_ylabel("Posterior mean claim frequency")
ax.set_title("Shrinkage plot: credibility effect on thin cells\n"
             "(point size = log exposure, colour = credibility factor Z)")
ax.legend(loc="upper left")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Bühlmann-Straub Z to Bayesian credibility_factor

# COMMAND ----------

# Join Polars DataFrames
bs_for_join = bs_results.rename({
    "Z": "Z_bs",
    "credibility_estimate": "bs_estimate",
}).select(["postcode_district", "Z_bs", "bs_estimate", "obs_mean"])

comparison = results.join(bs_for_join, on="postcode_district", how="inner")

print("Credibility factor comparison: Bühlmann-Straub Z vs Bayesian (approximate) Z")
print()
print(comparison.select([
    "postcode_district", "earned_years",
    "Z_bs", "credibility_factor",
    "bs_estimate", "posterior_mean",
]).head(20))

# COMMAND ----------

# Correlation between B-S Z and Bayesian Z (numpy for scalar stats)
z_bs_arr = comparison["Z_bs"].to_numpy()
z_bay_arr = comparison["credibility_factor"].to_numpy()
corr = np.corrcoef(z_bs_arr, z_bay_arr)[0, 1]
mad = np.abs(z_bs_arr - z_bay_arr).mean()

print(f"\nCorrelation between B-S Z and Bayesian credibility_factor: {corr:.4f}")
print(f"Mean absolute difference in Z values: {mad:.4f}")
print()
print("Note: the Bayesian credibility_factor here is a numerical approximation from")
print("posterior shrinkage. For this non-conjugate Poisson-lognormal model it is not")
print("exact. For very thin groups (Z < 0.10) the two may diverge for reasons beyond")
print("the Normal approximation alone.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot Z comparison

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Z vs Z scatter
ax = axes[0]
ax.scatter(comparison["Z_bs"].to_numpy(), comparison["credibility_factor"].to_numpy(),
           alpha=0.6, s=30, edgecolors="none", c="steelblue")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1.5)
ax.set_xlabel("Bühlmann-Straub Z")
ax.set_ylabel("Bayesian credibility factor (approx)")
ax.set_title("Credibility factor comparison\nB-S vs Bayesian")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Right: estimate vs estimate scatter
ax = axes[1]
bs_est_arr = comparison["bs_estimate"].to_numpy()
post_mean_arr = comparison["posterior_mean"].to_numpy()
ax.scatter(bs_est_arr, post_mean_arr,
           alpha=0.6, s=30, edgecolors="none", c="darkorange")
est_range = [min(bs_est_arr.min(), post_mean_arr.min()) * 0.9,
             max(bs_est_arr.max(), post_mean_arr.max()) * 1.1]
ax.plot(est_range, est_range, "k--", alpha=0.3, lw=1.5)
ax.set_xlabel("Bühlmann-Straub estimate")
ax.set_ylabel("Bayesian posterior mean")
ax.set_title("Credibility estimate comparison\nB-S vs Bayesian")

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Posterior predictive check
# MAGIC
# MAGIC Simulate datasets from the fitted Bayesian model and compare to observed data.
# MAGIC If the model is well-calibrated, the observed claim counts should fall within
# MAGIC the posterior predictive distribution.
# MAGIC
# MAGIC A model that converges (good R-hat, ESS, zero divergences) can still be
# MAGIC misspecified. PPCs are how you detect misspecification.

# COMMAND ----------

with hierarchical_model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# Simulated total claims per replicated dataset
sim_claims_ppc = ppc.posterior_predictive["claims_obs"].values.reshape(
    -1, n_districts_model
)  # shape: (n_samples, n_districts)

sim_totals = sim_claims_ppc.sum(axis=1)
obs_total = int(claims_arr.sum())

print("Posterior predictive check - total claims:")
print(f"  Observed: {obs_total:,}")
print(f"  PPC mean: {sim_totals.mean():.0f}")
print(f"  PPC 5th percentile:  {np.percentile(sim_totals, 5):.0f}")
print(f"  PPC 95th percentile: {np.percentile(sim_totals, 95):.0f}")
in_interval = np.percentile(sim_totals, 5) <= obs_total <= np.percentile(sim_totals, 95)
print(f"  Observed in 90% predictive interval: {in_interval}")
pvalue_total = (sim_totals >= obs_total).mean()
print(f"  Posterior predictive p-value: {pvalue_total:.3f}  (should be 0.05-0.95)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### PPC: district-level calibration
# MAGIC
# MAGIC What fraction of districts have their observed claim count inside the 90% posterior
# MAGIC predictive interval? We expect roughly 90% - that is what "calibrated" means.
# MAGIC
# MAGIC If district-level 90% PPC coverage is materially below 90% (e.g. 75%), the model
# MAGIC is misspecified. Options: switch to Negative Binomial, revisit the random effect
# MAGIC prior, or revert to Bühlmann-Straub and document why the Bayesian model failed.

# COMMAND ----------

lower_ppc = np.percentile(sim_claims_ppc, 5, axis=0)
upper_ppc = np.percentile(sim_claims_ppc, 95, axis=0)
obs_claims_arr = claims_arr  # already numpy

within = ((obs_claims_arr >= lower_ppc) & (obs_claims_arr <= upper_ppc))
coverage_rate = within.mean()

print(f"District-level 90% PPC coverage: {coverage_rate * 100:.1f}%  (target: 90%)")

if abs(coverage_rate - 0.90) < 0.05:
    print("  Coverage is within 5pp of target - model is well calibrated.")
elif coverage_rate < 0.85:
    print("  Coverage is below 85% - model may be overconfident.")
    print("  Actions: (1) try Negative Binomial likelihood, (2) widen sigma prior,")
    print("  (3) if persistent, revert to Bühlmann-Straub and document why.")
else:
    print("  Coverage is above 95% - model may be underconfident (too much uncertainty).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Recovery check: how well do credibility estimates recover true rates?

# COMMAND ----------

true_rates_arr = np.array([
    BASE_FREQUENCY * np.exp(TRUE_AREA_PARAMS.get(d, 0.0))
    for d in districts_sorted
])

comparison = comparison.with_columns(
    pl.Series("true_rate", [
        BASE_FREQUENCY * np.exp(TRUE_AREA_PARAMS.get(d, 0.0))
        for d in comparison["postcode_district"].to_list()
    ])
)

obs_arr_c   = comparison["observed_rate"].to_numpy()
bs_arr_c    = comparison["bs_estimate"].to_numpy()
bayes_arr_c = comparison["posterior_mean"].to_numpy()
true_arr_c  = comparison["true_rate"].to_numpy()

mse_observed_c = ((obs_arr_c   - true_arr_c) ** 2).mean()
mse_bs_c       = ((bs_arr_c    - true_arr_c) ** 2).mean()
mse_bayes_c    = ((bayes_arr_c - true_arr_c) ** 2).mean()

mape_observed_c = (np.abs(obs_arr_c   - true_arr_c) / true_arr_c).mean()
mape_bs_c       = (np.abs(bs_arr_c    - true_arr_c) / true_arr_c).mean()
mape_bayes_c    = (np.abs(bayes_arr_c - true_arr_c) / true_arr_c).mean()

print("MSE vs true rate (lower is better):")
print(f"  Observed rate:              {mse_observed_c:.8f}  (no pooling)")
print(f"  Bühlmann-Straub estimate:   {mse_bs_c:.8f}  ({(1 - mse_bs_c/mse_observed_c)*100:.1f}% reduction)")
print(f"  Bayesian posterior mean:    {mse_bayes_c:.8f}  ({(1 - mse_bayes_c/mse_observed_c)*100:.1f}% reduction)")
print()
print("MAPE vs true rate (lower is better):")
print(f"  Observed rate:              {mape_observed_c * 100:.2f}%")
print(f"  Bühlmann-Straub estimate:   {mape_bs_c * 100:.2f}%  ({(1 - mape_bs_c/mape_observed_c)*100:.1f}% reduction)")
print(f"  Bayesian posterior mean:    {mape_bayes_c * 100:.2f}%  ({(1 - mape_bayes_c/mape_observed_c)*100:.1f}% reduction)")
print()
print("Both credibility methods substantially outperform naive observed rates.")
print("Bayesian and B-S are close - the Normal approximation in B-S works well here.")

# COMMAND ----------

# Split by exposure tier using Polars when/then/otherwise (no pd.cut)
comparison = comparison.with_columns(
    pl.when(pl.col("earned_years") < 50)
    .then(pl.lit("< 50 yr (thin)"))
    .when(pl.col("earned_years") < 200)
    .then(pl.lit("50-200 yr"))
    .when(pl.col("earned_years") < 1000)
    .then(pl.lit("200-1000 yr"))
    .otherwise(pl.lit("> 1000 yr (dense)"))
    .alias("exposure_tier")
)

print("\nMSE and MAPE vs true rate by exposure tier:")
print()
for tier in ["< 50 yr (thin)", "50-200 yr", "200-1000 yr", "> 1000 yr (dense)"]:
    grp = comparison.filter(pl.col("exposure_tier") == tier)
    n = len(grp)
    if n == 0:
        continue
    o = grp["observed_rate"].to_numpy()
    b = grp["bs_estimate"].to_numpy()
    y = grp["posterior_mean"].to_numpy()
    t = grp["true_rate"].to_numpy()
    mse_o = ((o - t) ** 2).mean()
    mse_b = ((b - t) ** 2).mean()
    mse_y = ((y - t) ** 2).mean()
    mape_o = (np.abs(o - t) / t).mean() * 100
    mape_b = (np.abs(b - t) / t).mean() * 100
    mape_y = (np.abs(y - t) / t).mean() * 100
    print(f"  {tier} ({n} districts):")
    print(f"    MSE  - Observed: {mse_o:.8f}  |  B-S: {mse_b:.8f}  |  Bayes: {mse_y:.8f}")
    print(f"    MAPE - Observed: {mape_o:.1f}%  |  B-S: {mape_b:.1f}%  |  Bayes: {mape_y:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Format credibility estimates as a factor table
# MAGIC
# MAGIC Convert the posterior means into multiplicative relativities vs the grand mean,
# MAGIC with 90% credible intervals. This is the format your pricing committee expects.

# COMMAND ----------

factor_table = comparison.select([
    "postcode_district", "earned_years", "claims",
    "observed_rate", "posterior_mean", "lower_90", "upper_90",
    "credibility_factor",
])

# Convert to relativities vs grand mean
factor_table = factor_table.with_columns([
    (pl.col("posterior_mean") / grand_mean_bayes).alias("relativity"),
    (pl.col("lower_90") / grand_mean_bayes).alias("ci_lower"),
    (pl.col("upper_90") / grand_mean_bayes).alias("ci_upper"),
    (pl.col("observed_rate") / grand_mean_bayes).alias("observed_relativity"),
])

# Round for presentation
factor_table = factor_table.with_columns([
    pl.col("relativity").round(3),
    pl.col("ci_lower").round(3),
    pl.col("ci_upper").round(3),
    pl.col("credibility_factor").round(3),
    pl.col("observed_relativity").round(3),
])

factor_table_display = factor_table.sort("relativity", descending=True)

print("Credibility-weighted factor table (sorted by relativity, top 15):")
print()
print(factor_table_display.select([
    "postcode_district", "earned_years", "observed_relativity",
    "relativity", "ci_lower", "ci_upper", "credibility_factor"
]).head(15))

print()
print(f"\nGrand mean (base rate): {grand_mean_bayes:.5f} ({grand_mean_bayes * 100:.3f}% per year)")
print("Relativities are multiplicative vs the grand mean.")
print("90% credible intervals widen substantially for thin cells - this is correct, not a bug.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Write results to Unity Catalog
# MAGIC
# MAGIC Hard gate on convergence: if the model has not converged or has divergences,
# MAGIC we raise an error rather than silently write potentially biased posteriors.
# MAGIC A model that writes bad estimates to Delta is worse than one that fails loudly.

# COMMAND ----------

if max_rhat > 1.01 or n_div > 0:
    raise ValueError(
        f"Model failed convergence: max_rhat={max_rhat:.4f}, divergences={n_div}. "
        "Results not written to Unity Catalog."
    )

RUN_DATE = str(date.today())
MODEL_NAME_BAYES = "hierarchical_freq_v1_module06"
MODEL_NAME_BS = "buhlmann_straub_v1_module06"

# COMMAND ----------

# Bayesian results
bayes_out = factor_table.with_columns([
    pl.lit(MODEL_NAME_BAYES).alias("model_name"),
    pl.lit("hierarchical_poisson").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(max_rhat).alias("max_rhat"),
    pl.lit(n_div).alias("n_divergences"),
])

(
    spark.createDataFrame(bayes_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_credibility_estimates")
)

print(f"Written {len(bayes_out)} rows to main.pricing.module06_credibility_estimates")

# COMMAND ----------

# Bühlmann-Straub results
bs_out = bs_results.rename({
    "obs_mean": "observed_rate",
    "credibility_estimate": "bs_estimate",
}).with_columns([
    (pl.col("bs_estimate") / bs["grand_mean"]).alias("bs_relativity"),
    pl.lit(MODEL_NAME_BS).alias("model_name"),
    pl.lit("buhlmann_straub").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(bs["grand_mean"]).alias("grand_mean"),
    pl.lit(bs["v_hat"]).alias("v_hat"),
    pl.lit(bs["a_hat"]).alias("a_hat"),
    pl.lit(bs["k"]).alias("K"),
])

(
    spark.createDataFrame(bs_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_bs_estimates")
)

print(f"Written {len(bs_out)} rows to main.pricing.module06_bs_estimates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

print("=" * 65)
print("MODULE 6 - CREDIBILITY & BAYESIAN PRICING SUMMARY")
print("=" * 65)
print()
print("Portfolio:")
print(f"  {len(df):,} district-year rows")
print(f"  {len(dist_totals)} postcode districts")
thin_count = dist_totals.filter(pl.col("earned_years") < 50).height
print(f"  {thin_count} thin districts (< 50 earned years)")
print()
print("Bühlmann-Straub (inline NumPy implementation):")
print(f"  Grand mean:  {bs['grand_mean']:.5f}")
print(f"  K = v/a:     {bs['k']:.1f}  (half-credibility at {bs['k']:.0f} earned years)")
z_min = bs_results["Z"].min()
z_max = bs_results["Z"].max()
print(f"  Z range:     [{z_min:.3f}, {z_max:.3f}]")
print(f"  MSE vs true: {mse_bs_c:.8f}  ({(1 - mse_bs_c/mse_observed_c)*100:.1f}% vs observed)")
print(f"  MAPE vs true:{mape_bs_c * 100:.2f}%  ({(1 - mape_bs_c/mape_observed_c)*100:.1f}% vs observed)")
print()
print("Bayesian (hierarchical PyMC model):")
print(f"  Grand mean:             {grand_mean_bayes:.5f}")
print(f"  sigma_district:         {sigma_district_mean:.4f}  (log-scale between-district SD)")
print(f"  Max R-hat:              {max_rhat:.4f}")
print(f"  sigma ESS:              {sigma_ess:.0f}")
print(f"  Divergences:            {n_div}")
z_bay_min = results["credibility_factor"].min()
z_bay_max = results["credibility_factor"].max()
print(f"  Z range:                [{z_bay_min:.3f}, {z_bay_max:.3f}]")
print(f"  MSE vs true:            {mse_bayes_c:.8f}  ({(1 - mse_bayes_c/mse_observed_c)*100:.1f}% vs observed)")
print(f"  MAPE vs true:           {mape_bayes_c * 100:.2f}%  ({(1 - mape_bayes_c/mape_observed_c)*100:.1f}% vs observed)")
print(f"  PPC district coverage:  {coverage_rate * 100:.1f}%  (target 90%)")
print()
print("Delta tables written:")
print("  main.pricing.module06_credibility_estimates")
print("  main.pricing.module06_bs_estimates")
