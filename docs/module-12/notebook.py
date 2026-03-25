# Databricks notebook source
# MAGIC %md
# MAGIC # Module 12: Spatial Territory Rating
# MAGIC
# MAGIC **Modern Insurance Pricing with Python and Databricks**
# MAGIC
# MAGIC Territory is typically treated as a categorical variable in Emblem: bucket postcodes into
# MAGIC groups, fit a GLM, read off the group relativities. The problem is that this ignores
# MAGIC spatial structure. Adjacent areas share risk. A rating model that treats neighbouring
# MAGIC postcode sectors as independent is throwing away information.
# MAGIC
# MAGIC This notebook replaces that approach with BYM2 — a Bayesian hierarchical model that
# MAGIC explicitly encodes neighbourhood similarity and produces territory relativities with
# MAGIC proper credibility intervals.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates synthetic postcode-sector level claim data with deliberate spatial structure
# MAGIC 2. Builds a queen-contiguity adjacency matrix for a 10x10 area grid
# MAGIC 3. Runs Moran's I to confirm spatial autocorrelation before fitting anything
# MAGIC 4. Fits a BYM2 spatial model via PyMC 5 (ICAR + IID components)
# MAGIC 5. Checks MCMC convergence diagnostics (R-hat, ESS, divergences)
# MAGIC 6. Runs Moran's I post-fit to confirm spatial structure was absorbed
# MAGIC 7. Extracts territory relativities with 95% credibility intervals
# MAGIC 8. Compares BYM2 estimates to naive O/E relativities
# MAGIC 9. Shows how to use territory factors as a GLM log-offset
# MAGIC 10. Demonstrates the two-stage pipeline for production use
# MAGIC 11. Writes territory factors to Unity Catalog Delta tables
# MAGIC
# MAGIC **Runtime:** 30-50 minutes on a 4-core cluster (MCMC is the slow step).
# MAGIC
# MAGIC **Prerequisites:** None. This notebook is self-contained.
# MAGIC
# MAGIC **Key concept:** rho is the most important output. A posterior mean of rho near 1.0 means
# MAGIC the data support strong spatial structure and your territory factors are doing real work.
# MAGIC If rho is near 0.0, the data say geographic variation is essentially random noise and
# MAGIC BYM2 is not adding information over Bühlmann-Straub credibility (Module 6).

# COMMAND ----------

%pip install insurance-spatial pymc arviz polars matplotlib "numpy<2.0" --quiet
# Use %pip in Databricks notebooks. Outside Databricks: uv add "insurance-spatial[geo]" pymc arviz polars matplotlib
# The numpy<2.0 pin avoids a known compatibility issue in the PyMC 5 / ArviZ 0.18 ecosystem.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import date

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import arviz as az

from insurance_spatial import build_grid_adjacency, BYM2Model
from insurance_spatial.diagnostics import moran_i
from insurance_spatial.plots import plot_relativities, plot_trace

warnings.filterwarnings("ignore")

print(f"Today: {date.today()}")
print(f"NumPy:             {np.__version__}")
print(f"Polars:            {pl.__version__}")
print(f"ArviZ:             {az.__version__}")
print("insurance-spatial: imported OK")
print("All imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0: Configuration

# COMMAND ----------

CATALOG = "main"
SCHEMA  = "territory"

TABLES = {
    "relativities":   f"{CATALOG}.{SCHEMA}.bym2_relativities",
    "diagnostics":    f"{CATALOG}.{SCHEMA}.bym2_diagnostics",
    "factor_table":   f"{CATALOG}.{SCHEMA}.territory_factors",
}

RUN_DATE     = str(date.today())
MODEL_VERSION = "v1"

# Grid dimensions for the synthetic territory
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS   # 100 areas

print(f"Run date:   {RUN_DATE}")
print(f"Areas:      {N} ({NROWS}x{NCOLS} grid)")
print(f"Catalog:    {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Generate synthetic territory data
# MAGIC
# MAGIC We simulate a 10x10 grid of 100 areas. Each area is a synthetic postcode sector.
# MAGIC The spatial structure is deliberate:
# MAGIC - North (rows 0-2): elevated risk (urban concentration)
# MAGIC - South (rows 7-9): reduced risk (suburban/rural)
# MAGIC - Centre: near zero effect
# MAGIC
# MAGIC Exposure varies across areas (Gamma distribution) to mimic real portfolio sparsity.
# MAGIC Some areas will have zero or near-zero claims — this is the thin data problem we are
# MAGIC here to solve.

# COMMAND ----------

rng = np.random.default_rng(seed=42)

# True log-scale spatial effects
row_idx = np.array([r for r in range(NROWS) for c in range(NCOLS)])
true_log_effect = np.where(row_idx <= 2, 0.35, np.where(row_idx >= 7, -0.25, 0.0))

# Add smooth geographic noise (sinusoidal east-west gradient) plus area scatter
smooth_noise  = 0.05 * np.sin(np.linspace(0, 2 * np.pi, N))
area_scatter  = rng.normal(0, 0.08, N)
true_log_effect = true_log_effect + smooth_noise + area_scatter

# Exposure: policies per area (varies; some areas are sparse)
exposure = rng.gamma(shape=2.5, scale=20.0, size=N).astype(int) + 5

# Base claim frequency: 8% before territory effect
base_freq       = 0.08
expected_claims = exposure * base_freq * np.exp(true_log_effect)

# Observed claims: Poisson draw
claims = rng.poisson(expected_claims)

# Area labels: r0c0 ... r9c9
areas = [f"r{r}c{c}" for r in range(NROWS) for c in range(NCOLS)]

print(f"Areas:              {N}")
print(f"Total exposure:     {exposure.sum():,} policy-years")
print(f"Total claims:       {claims.sum():,}")
print(f"Overall frequency:  {claims.sum() / exposure.sum():.4f}")
print(f"Zero-claim areas:   {(claims == 0).sum()}")
print(f"Sparse areas (<5):  {(claims < 5).sum()}")

# COMMAND ----------

# Build a Polars DataFrame for the area-level data
df = pl.DataFrame({
    "area":              areas,
    "row":               row_idx.tolist(),
    "col":               [c for r in range(NROWS) for c in range(NCOLS)],
    "exposure":          exposure.tolist(),
    "claims":            claims.tolist(),
    "true_log_effect":   true_log_effect.tolist(),
})

df = df.with_columns(
    (pl.col("claims") / pl.col("exposure")).alias("obs_freq")
)

print(df.head(12))
print()
print(f"Schema: {df.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Build the adjacency matrix
# MAGIC
# MAGIC The adjacency matrix W encodes which areas are neighbours. We use queen contiguity
# MAGIC (shared edges and corners). This is the convention for UK postcode geography because
# MAGIC sectors often share only a single corner point, and ignoring those connections creates
# MAGIC artificially disconnected areas.
# MAGIC
# MAGIC The graph must be connected for the ICAR model to be identified. The `n_components()`
# MAGIC check confirms this. For real UK data, islands (Orkney, Shetland) are connected to
# MAGIC their nearest mainland sector by centroid distance when you use `from_geojson()`.

# COMMAND ----------

adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")

print(f"Areas (N):            {adj.n}")
print(f"Total edges:          {adj.W.nnz // 2}")   # symmetric: divide by 2
print(f"Mean neighbours:      {adj.neighbour_counts().mean():.2f}")
print(f"Min neighbours:       {adj.neighbour_counts().min()}")
print(f"Max neighbours:       {adj.neighbour_counts().max()}")
print(f"Connected components: {adj.n_components()}  (must be 1)")
print(f"Scaling factor (s):   {adj.scaling_factor:.4f}")
print()
print("Adjacency structure OK." if adj.n_components() == 1
      else "WARNING: graph is disconnected. ICAR model will fail.")

# COMMAND ----------

# Inspect the adjacency structure for one area
first_row     = adj.W.getrow(0).toarray().ravel()
neighbour_idx = np.where(first_row == 1)[0]
neighbour_lbl = [adj.areas[i] for i in neighbour_idx]

print(f"W type:  {type(adj.W)}")
print(f"W shape: {adj.W.shape}")
print(f"Density: {adj.W.nnz / (adj.n ** 2):.4f}")
print(f"Neighbours of r0c0 (top-left corner): {neighbour_lbl}")
print("  (Corner cell has 3 queen neighbours)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Pre-fit Moran's I test
# MAGIC
# MAGIC Before fitting any model, we test whether spatial autocorrelation is present.
# MAGIC If Moran's I is not significant (p > 0.05), BYM2 is not warranted — use
# MAGIC Bühlmann-Straub credibility (Module 6) instead.
# MAGIC
# MAGIC We use the log O/E ratio as the input: log((observed + 0.5) / (expected_null + 0.5)).
# MAGIC The +0.5 is Haldane's correction for zero-claim areas.

# COMMAND ----------

# Compute log O/E relative to the null (uniform frequency) model
portfolio_freq = claims.sum() / exposure.sum()
expected_null  = exposure * portfolio_freq
log_oe         = np.log((claims + 0.5) / (expected_null + 0.5))

# Add to DataFrame
df = df.with_columns(pl.Series("log_oe", log_oe.tolist()))

print(f"Log O/E summary:")
print(f"  Mean:  {log_oe.mean():.4f}  (should be near zero)")
print(f"  SD:    {log_oe.std():.4f}")
print(f"  Min:   {log_oe.min():.4f}")
print(f"  Max:   {log_oe.max():.4f}")

# COMMAND ----------

# Run Moran's I permutation test
test_pre = moran_i(log_oe, adj, n_permutations=999)

print(f"Moran's I:    {test_pre.statistic:.4f}")
print(f"Expected I:   {test_pre.expected:.4f}")
print(f"Z-score:      {test_pre.z_score:.2f}")
print(f"p-value:      {test_pre.p_value:.4f}")
print(f"Significant:  {test_pre.significant}")
print()
print(test_pre.interpretation)

if not test_pre.significant:
    print()
    print("Moran's I is not significant. Spatial smoothing is not warranted.")
    print("Consider Bühlmann-Straub credibility (Module 6) instead.")
else:
    print()
    print("Spatial autocorrelation confirmed. Proceeding to BYM2 model.")

# COMMAND ----------

# Visualise the raw spatial pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

freq_grid  = (claims / exposure).reshape(NROWS, NCOLS)
logoe_grid = log_oe.reshape(NROWS, NCOLS)

im0 = axes[0].imshow(freq_grid, cmap="RdYlGn_r", origin="upper")
axes[0].set_title("Observed claim frequency (raw)")
plt.colorbar(im0, ax=axes[0])
for r in range(NROWS):
    for c in range(NCOLS):
        axes[0].text(c, r, f"{freq_grid[r, c]:.2f}", ha="center", va="center",
                     fontsize=6, color="black")

im1 = axes[1].imshow(logoe_grid, cmap="RdYlGn_r", origin="upper")
axes[1].set_title(
    f"Log O/E  (Moran's I = {test_pre.statistic:.3f}, p = {test_pre.p_value:.3f})"
)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("/tmp/spatial_raw.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/spatial_raw.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Fit the BYM2 model
# MAGIC
# MAGIC The BYM2 model combines two components:
# MAGIC - **ICAR (phi):** spatially structured effect. Area i's expected value given neighbours
# MAGIC   is the average of its neighbours' values.
# MAGIC - **IID (theta):** unstructured per-area random effect. Captures genuinely anomalous
# MAGIC   areas not explained by their neighbours.
# MAGIC
# MAGIC The mixing parameter rho controls the balance:
# MAGIC - rho near 1.0: all geographic variation is spatially smooth
# MAGIC - rho near 0.0: all geographic variation is area-specific noise
# MAGIC
# MAGIC The scaling factor s (from the adjacency structure) makes sigma and rho interpretable
# MAGIC regardless of the graph topology. This is the key methodological innovation in Riebler
# MAGIC et al. (2016) over the original BYM parameterisation.
# MAGIC
# MAGIC **Runtime:** 3-8 minutes for this 100-area model on a 4-core cluster.

# COMMAND ----------

model = BYM2Model(
    adjacency=adj,
    draws=1000,          # posterior samples per chain (post-warmup)
    chains=4,            # number of independent MCMC chains
    target_accept=0.9,   # NUTS acceptance rate; increase to 0.95 if divergences occur
    tune=1000,           # warmup steps per chain
)

print(f"BYM2Model configured:")
print(f"  Areas:           {adj.n}")
print(f"  Scaling factor:  {adj.scaling_factor:.4f}")
print(f"  Draws per chain: {model.draws}")
print(f"  Chains:          {model.chains}")
print(f"  Total samples:   {model.draws * model.chains:,}")

# COMMAND ----------

result = model.fit(
    claims=claims,
    exposure=exposure.astype(float),
    random_seed=42,
)

print("Fitting complete.")
print(f"Areas in result:       {result.n_areas}")
print(f"Areas list (first 5):  {result.areas[:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5: MCMC convergence diagnostics
# MAGIC
# MAGIC Never use results from an MCMC model without checking convergence. MCMC always
# MAGIC produces samples — the question is whether they are reliable.
# MAGIC
# MAGIC **R-hat** (Gelman-Rubin): compares variance within and between chains. Want < 1.01.
# MAGIC **ESS** (effective sample size): equivalent independent samples. Want > 400.
# MAGIC **Divergences**: indicates regions where the sampler stepped inaccurately. Want 0.
# MAGIC
# MAGIC If R-hat > 1.01 for any parameter, the model has not converged. Increase `tune` and refit.
# MAGIC If you see > 10 divergences, increase `target_accept` to 0.95 and refit.

# COMMAND ----------

diag = result.diagnostics()

print("=== Convergence ===")
print(f"Max R-hat:       {diag.convergence.max_rhat:.4f}  (want < 1.01)")
print(f"Min ESS bulk:    {diag.convergence.min_ess_bulk:.0f}  (want > 400)")
print(f"Min ESS tail:    {diag.convergence.min_ess_tail:.0f}  (want > 400)")
print(f"Divergences:     {diag.convergence.n_divergences}  (want 0)")
print(f"Converged:       {diag.convergence.converged}")
print()
print("=== R-hat by parameter ===")
print(diag.convergence.rhat_by_param)
print()
print("=== rho posterior ===")
print(diag.rho_summary)
print()
print("=== sigma posterior ===")
print(diag.sigma_summary)

# COMMAND ----------

# Trace plots for key global parameters
# Healthy chains look like "hairy caterpillars": fast mixing, all chains overlapping
fig = plot_trace(result, params=["alpha", "sigma", "rho"])
plt.tight_layout()
plt.savefig("/tmp/bym2_trace.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/bym2_trace.png")

# COMMAND ----------

# ArviZ summary for global parameters (more detail)
idata = result.trace
az_summary = az.summary(
    idata,
    var_names=["alpha", "sigma", "rho"],
    round_to=4,
)
print(az_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6: Post-fit Moran's I
# MAGIC
# MAGIC We run Moran's I again on the posterior predictive residuals. If BYM2 has absorbed
# MAGIC the spatial structure, the post-fit residuals should show no significant autocorrelation.
# MAGIC
# MAGIC If post-fit I is still significant, the model has not fully captured the pattern.
# MAGIC Likely causes: insufficient MCMC draws, multi-scale spatial structure, or a missing
# MAGIC covariate with a strong spatial signal (e.g., flood risk, deprivation index).

# COMMAND ----------

# Posterior predictive mean of claims per area
mu_samples = result.trace.posterior["mu"].values  # shape: (chains, draws, N)
mu_mean    = mu_samples.mean(axis=(0, 1))          # shape: (N,)

postfit_log_oe = np.log((claims + 0.5) / (mu_mean + 0.5))

test_post = moran_i(postfit_log_oe, adj, n_permutations=999)

print("=== Pre-fit Moran's I ===")
print(f"  I = {test_pre.statistic:.4f}, p = {test_pre.p_value:.4f}")
print()
print("=== Post-fit Moran's I ===")
print(f"  I = {test_post.statistic:.4f}, p = {test_post.p_value:.4f}")
print()
print(test_post.interpretation)
print()
if not test_post.significant and test_pre.significant:
    print("Model has absorbed the spatial structure. Residuals are spatially random.")
elif test_post.significant:
    print("WARNING: Post-fit residuals still show spatial autocorrelation.")
    print("The model may need more tuning steps, or there is a missing spatial covariate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: Extract territory relativities
# MAGIC
# MAGIC Territory relativities are the multiplicative factors that go into the rating engine.
# MAGIC They are normalised to the geometric mean (product of all relativities = 1), so no
# MAGIC area is the explicit reference.
# MAGIC
# MAGIC Each relativity comes with a 95% credibility interval. Areas with sparse exposure
# MAGIC get wider intervals — this is the credibility principle made explicit. A naive O/E
# MAGIC approach gives you no such uncertainty quantification.

# COMMAND ----------

rels = result.territory_relativities(credibility_interval=0.95)

print(rels.head(10))
print()
print(f"Columns: {rels.columns}")
print()
print(f"Relativity range:         [{rels['relativity'].min():.4f}, {rels['relativity'].max():.4f}]")
print(f"Areas with rel > 1.10:    {(rels['relativity'] > 1.10).sum()}")
print(f"Areas with rel < 0.90:    {(rels['relativity'] < 0.90).sum()}")

# COMMAND ----------

# Add CI width as an explicit uncertainty measure
rels = rels.with_columns(
    (pl.col("upper") - pl.col("lower")).alias("ci_width")
)

print("Top 10 highest risk areas:")
print(rels.sort("relativity", descending=True).head(10))
print()
print("Top 10 lowest risk areas:")
print(rels.sort("relativity").head(10))
print()
print("Widest credibility intervals (most uncertain = sparsest areas):")
print(rels.sort("ci_width", descending=True).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8: Visualise territory relativities
# MAGIC
# MAGIC Three panels: true relativities (known because synthetic), BYM2 posterior mean,
# MAGIC and naive unsmoothed O/E. BYM2 should be visibly smoother than naive O/E and
# MAGIC visibly closer to the true pattern — especially in sparse areas where naive O/E
# MAGIC is dominated by sampling noise.

# COMMAND ----------

# Bar chart of top/bottom areas with credibility intervals
fig = plot_relativities(
    rels,
    title="BYM2 Territory Relativities (synthetic 10x10 grid)",
    n_areas=30,
)
plt.savefig("/tmp/bym2_relativities_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/bym2_relativities_bar.png")

# COMMAND ----------

# Grid choropleth: true vs BYM2 vs naive O/E
# Areas are sorted alphabetically (r0c0, r0c1, ..., r9c9)
rel_values = np.array(rels.sort("area")["relativity"].to_list())
rel_grid   = rel_values.reshape(NROWS, NCOLS)

true_rel_grid = np.exp(true_log_effect - true_log_effect.mean()).reshape(NROWS, NCOLS)

naive_freq = claims / exposure
naive_rel  = naive_freq / (claims.sum() / exposure.sum())
naive_grid = naive_rel.reshape(NROWS, NCOLS)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin, vmax = 0.7, 1.5

im0 = axes[0].imshow(true_rel_grid, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, origin="upper")
axes[0].set_title("True relativities (known; synthetic data)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(rel_grid, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, origin="upper")
axes[1].set_title("BYM2 posterior mean relativities")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(naive_grid, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, origin="upper")
axes[2].set_title("Naive O/E relativities (unsmoothed)")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("Territory relativity estimation: BYM2 vs. naive O/E", y=1.02)
plt.tight_layout()
plt.savefig("/tmp/bym2_choropleth.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/bym2_choropleth.png")

# COMMAND ----------

# Quantify the improvement
bym2_error  = np.abs(rel_values - true_rel_grid.ravel()).mean()
naive_error = np.abs(naive_rel  - true_rel_grid.ravel()).mean()

print(f"BYM2 MAE vs. truth:   {bym2_error:.4f}")
print(f"Naive O/E MAE:        {naive_error:.4f}")
print(f"Improvement:          {(1 - bym2_error / naive_error) * 100:.1f}%")
print()
print("BYM2 typically reduces MAE by 20-40% on this synthetic dataset.")
print("The improvement is largest in sparse areas where naive O/E is most noisy.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: Two-stage pipeline for production use
# MAGIC
# MAGIC In production you do not pass raw claims and exposure directly to BYM2 (integrated
# MAGIC model). You use the two-stage approach:
# MAGIC
# MAGIC **Stage 1:** Fit a non-spatial model (CatBoost, GLM) on individual policy records.
# MAGIC Compute sector-level observed and expected claims.
# MAGIC
# MAGIC **Stage 2:** Pass sector-level observed claims and base model expectations to BYM2.
# MAGIC The `exposure` argument is now the base model's prediction (not policy-years).
# MAGIC
# MAGIC The output relativities are applied as a multiplicative adjustment on top of the
# MAGIC base model. This decouples the spatial and non-spatial estimation problems, making
# MAGIC both easier to audit and update independently.

# COMMAND ----------

# Simulate Stage 1: an intercept-only "base model" that gets the portfolio
# frequency right but has no geographic variation
base_expected   = exposure * portfolio_freq   # intercept-only base model
sector_observed = claims.copy()
sector_expected = base_expected.copy()

# --- Stage 2: BYM2 on the base model's O/E ---
# exposure argument = base model's expected claims per sector (not policy-years)
model_2s = BYM2Model(
    adjacency=adj,
    draws=1000,
    chains=4,
    target_accept=0.9,
    tune=1000,
)

result_2s = model_2s.fit(
    claims=sector_observed,
    exposure=sector_expected.astype(float),
    random_seed=43,
)

print("Two-stage model fitted.")
diag_2s = result_2s.diagnostics()
print(f"Max R-hat:          {diag_2s.convergence.max_rhat:.4f}")
print(f"Divergences:        {diag_2s.convergence.n_divergences}")
print(f"rho posterior mean: {diag_2s.rho_summary['mean'][0]:.3f}")
print(f"sigma posterior mean: {diag_2s.sigma_summary['mean'][0]:.3f}")

# COMMAND ----------

# Compare integrated vs. two-stage relativities
rels_2s = result_2s.territory_relativities(credibility_interval=0.95)

rels_compare = rels.join(
    rels_2s.select(["area", pl.col("relativity").alias("rel_2s")]),
    on="area",
)

corr = (
    rels_compare["relativity"].to_numpy()
    - rels_compare["rel_2s"].to_numpy()
)
print(f"Integrated vs. two-stage relativities:")
print(f"  Correlation:        {np.corrcoef(rels_compare['relativity'], rels_compare['rel_2s'])[0,1]:.4f}")
print(f"  Mean abs difference: {np.abs(corr).mean():.4f}")
print()
print("In practice, both approaches give similar results when the base model is well-specified.")
print("The two-stage approach is preferred for production because of its auditability.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 10: Using territory factors in a rating engine
# MAGIC
# MAGIC Territory relativities feed into the rating formula as multiplicative factors.
# MAGIC There are two integration patterns:
# MAGIC
# MAGIC **Option 1 (recommended):** Fixed offset in a downstream GLM. The `ln_offset` column
# MAGIC from `territory_relativities()` is ready-made. Pass it as a fixed offset (coefficient
# MAGIC constrained to 1.0, not estimated). This forces the GLM to accept the territory factor
# MAGIC exactly as BYM2 estimated it.
# MAGIC
# MAGIC **Option 2:** Lookup factor in a multiplicative tariff. Multiply the base rate by the
# MAGIC territory relativity at quote time. Simpler for legacy rating engines.

# COMMAND ----------

# Option 1: prepare ln_offset for GLM integration
# -----------------------------------------------------------------------
# In a Polars pipeline, join the territory offset onto the policy DataFrame
# before passing to the GLM.

rels_for_join = rels.select(["area", "ln_offset"]).rename(
    {"ln_offset": "territory_log_offset"}
)

# Simulate a policy-level dataset
n_policies    = 1000
policy_areas  = [f"r{rng.integers(0, 10)}c{rng.integers(0, 10)}" for _ in range(n_policies)]

df_policies = pl.DataFrame({
    "policy_id":        list(range(n_policies)),
    "postcode_sector":  policy_areas,
    "driver_age":       rng.integers(18, 80, n_policies).tolist(),
    "ncd_years":        rng.integers(0, 9, n_policies).tolist(),
})

df_policies_with_territory = df_policies.join(
    rels_for_join.rename({"area": "postcode_sector"}),
    on="postcode_sector",
    how="left",
)

# Fill new or unmatched sectors with 0.0 (grand mean = no territory adjustment)
df_policies_with_territory = df_policies_with_territory.with_columns(
    pl.col("territory_log_offset").fill_null(0.0)
)

print("Policy DataFrame with territory log-offset:")
print(df_policies_with_territory.head(10))
print()
print(f"Offset range: [{df_policies_with_territory['territory_log_offset'].min():.4f}, "
      f"{df_policies_with_territory['territory_log_offset'].max():.4f}]")

# COMMAND ----------

# Option 2: lookup factor table for a multiplicative tariff
# -----------------------------------------------------------------------
factor_table = rels.select([
    pl.col("area").alias("postcode_sector"),
    pl.col("relativity").alias("territory_factor"),
    pl.col("lower").alias("territory_factor_lower_95"),
    pl.col("upper").alias("territory_factor_upper_95"),
]).sort("postcode_sector")

print("Territory factor table (first 15 rows):")
print(factor_table.head(15))
print()
print("Usage in rating formula:")
print("  premium = base_rate * age_factor * ncd_factor * territory_factor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 11: Write results to Unity Catalog Delta tables
# MAGIC
# MAGIC Three tables:
# MAGIC - `bym2_relativities` — territory factors with credibility intervals
# MAGIC - `bym2_diagnostics` — convergence metrics for model governance
# MAGIC - `territory_factors` — the factor lookup table for the rating engine
# MAGIC
# MAGIC We include the run date and model version in every row. This lets you maintain
# MAGIC multiple vintages of territory factors in the same table and join on the version
# MAGIC that was live at any given date.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# -----------------------------------------------------------------------
# Write territory relativities
# -----------------------------------------------------------------------
rels_to_write = rels.with_columns([
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(MODEL_VERSION).alias("model_version"),
])

(spark.createDataFrame(rels_to_write.to_pandas())
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(TABLES["relativities"]))

print(f"Relativities written to {TABLES['relativities']}  ({len(rels)} rows)")

# COMMAND ----------

from pyspark.sql import Row

# -----------------------------------------------------------------------
# Write diagnostics row
# -----------------------------------------------------------------------
diag_row = {
    "run_date":         RUN_DATE,
    "model_version":    MODEL_VERSION,
    "n_areas":          int(adj.n),
    "n_chains":         int(model.chains),
    "n_draws":          int(model.draws),
    "scaling_factor":   float(adj.scaling_factor),
    "max_rhat":         float(diag.convergence.max_rhat),
    "min_ess_bulk":     float(diag.convergence.min_ess_bulk),
    "min_ess_tail":     float(diag.convergence.min_ess_tail),
    "n_divergences":    int(diag.convergence.n_divergences),
    "converged":        bool(diag.convergence.converged),
    "rho_mean":         float(diag.rho_summary["mean"][0]),
    "rho_sd":           float(diag.rho_summary["sd"][0]),
    "rho_q025":         float(diag.rho_summary["q025"][0]),
    "rho_q975":         float(diag.rho_summary["q975"][0]),
    "sigma_mean":       float(diag.sigma_summary["mean"][0]),
    "morans_i_prefit":  float(test_pre.statistic),
    "morans_p_prefit":  float(test_pre.p_value),
    "morans_i_postfit": float(test_post.statistic),
    "morans_p_postfit": float(test_post.p_value),
    "bym2_mae":         float(bym2_error),
    "naive_mae":        float(naive_error),
}

(spark.createDataFrame([Row(**diag_row)])
 .write
 .format("delta")
 .mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(TABLES["diagnostics"]))

print(f"Diagnostics written to {TABLES['diagnostics']}")

# COMMAND ----------

# -----------------------------------------------------------------------
# Write factor lookup table for the rating engine
# -----------------------------------------------------------------------
factor_to_write = factor_table.with_columns([
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(MODEL_VERSION).alias("model_version"),
])

(spark.createDataFrame(factor_to_write.to_pandas())
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(TABLES["factor_table"]))

print(f"Factor table written to {TABLES['factor_table']}  ({len(factor_table)} rows)")

# COMMAND ----------

# Quick verification query
verify = spark.sql(f"""
SELECT
    COUNT(*)       AS n_areas,
    MIN(territory_factor)  AS min_factor,
    MAX(territory_factor)  AS max_factor,
    AVG(territory_factor)  AS mean_factor
FROM {TABLES['factor_table']}
WHERE model_version = '{MODEL_VERSION}'
""")
verify.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the complete BYM2 spatial territory rating workflow:
# MAGIC
# MAGIC | Step | Purpose | Key output |
# MAGIC |------|---------|-----------|
# MAGIC | Data generation | Realistic synthetic spatial structure | 100-area grid with north-south gradient |
# MAGIC | Adjacency matrix | Encode neighbourhood structure | Queen contiguity W matrix |
# MAGIC | Pre-fit Moran's I | Confirm spatial structure exists | I statistic + p-value |
# MAGIC | BYM2 fitting | Borrow strength across neighbours | MCMC posterior samples |
# MAGIC | Convergence diagnostics | Validate MCMC reliability | R-hat, ESS, divergences |
# MAGIC | Post-fit Moran's I | Confirm model absorbed spatial pattern | Residual I should be non-significant |
# MAGIC | Territory relativities | Multiplicative factors per area | Point estimate + 95% CI per area |
# MAGIC | Comparison to naive O/E | Quantify the smoothing benefit | MAE reduction vs. unsmoothed |
# MAGIC | Rating engine integration | Slot into production formula | ln_offset for GLM or lookup table |
# MAGIC | Delta persistence | Audit trail and governance | Three Unity Catalog tables |
# MAGIC
# MAGIC **rho interpretation:**
# MAGIC - rho near 1.0: strong spatial structure, BYM2 factors are well-supported
# MAGIC - rho near 0.0: no spatial structure, use Bühlmann-Straub credibility instead
# MAGIC - Wide rho CI: data are too sparse to identify the spatial structure reliably
# MAGIC
# MAGIC **When not to use BYM2:**
# MAGIC - Moran's I is not significant (p > 0.05)
# MAGIC - Fewer than 50 areas with data (not enough neighbours to learn from)
# MAGIC - You cannot explain the methodology to your pricing committee
# MAGIC - The model needs to update daily (MCMC is too slow; use Bühlmann-Straub)
# MAGIC
# MAGIC **Next step:** Replace the synthetic grid with real UK postcode sector boundaries
# MAGIC using `insurance_spatial.from_geojson()`. Part 15 of the tutorial covers this,
# MAGIC including island handling and the 11,200-sector computational considerations.

# COMMAND ----------

print("Module 12 notebook complete.")
print(f"  Relativities:  {TABLES['relativities']}")
print(f"  Diagnostics:   {TABLES['diagnostics']}")
print(f"  Factor table:  {TABLES['factor_table']}")
print()
print(f"  rho posterior mean:  {diag.rho_summary['mean'][0]:.3f}")
print(f"  sigma posterior mean: {diag.sigma_summary['mean'][0]:.3f}")
print(f"  Converged:           {diag.convergence.converged}")
print(f"  BYM2 MAE improvement over naive O/E: "
      f"{(1 - bym2_error / naive_error) * 100:.1f}%")
