## Part 7: Bayesian hierarchical models with PyMC

### The model we are building

We will fit a Poisson hierarchical model to the district-level claim counts. The model in full:

```
claims_i  ~  Poisson(lambda_i × exposure_i)
log(lambda_i)  =  alpha + u_district[i]
u_district[k]  ~  Normal(0, sigma_district)
alpha           ~  Normal(log(mu_portfolio), 0.5)
sigma_district  ~  HalfNormal(0.3)
```

**What each line means:**

`claims_i ~ Poisson(lambda_i × exposure_i)` — Claims for district i are Poisson-distributed with rate lambda_i per unit of exposure. This is the same distributional assumption as a Poisson GLM.

`log(lambda_i) = alpha + u_district[i]` — The log-rate for district i is the sum of a global intercept (alpha) and a district-specific deviation (u_district[i]).

`u_district[k] ~ Normal(0, sigma_district)` — District deviations are drawn from a Normal distribution centred at zero with standard deviation sigma_district. When sigma_district is large, the model allows districts to differ substantially from the global mean. When sigma_district is small, districts are heavily pooled toward the global mean. The data determine sigma_district.

`alpha ~ Normal(log(mu_portfolio), 0.5)` — The global intercept has a weakly informative prior centred on the log of the observed portfolio frequency. SD = 0.5 on the log scale allows the intercept to range from approximately 0.6× to 1.65× the portfolio mean at ±1 SD — not unduly tight.

`sigma_district ~ HalfNormal(0.3)` — The between-district standard deviation has a HalfNormal prior (constrained to be positive). HalfNormal(0.3) places most prior mass below about 0.6 log points, implying most districts sit within roughly 0.5× to 2.0× the portfolio mean — reasonable for UK motor postcodes.

### Why we need MCMC

Unlike GLMs, which have closed-form maximum likelihood solutions, this hierarchical model cannot be solved analytically. The posterior distribution over all parameters — alpha, all 120 u_district[k] values, and sigma_district — is a 122-dimensional distribution that has no closed form.

MCMC (Markov Chain Monte Carlo) is a family of algorithms that draw samples from this posterior distribution. PyMC uses NUTS (No-U-Turn Sampler), a particularly efficient variant. After sampling, we summarise the posterior with means and quantiles.

You do not need to understand NUTS in detail. You need to understand three things:
1. How to run it (the `pm.sample()` call below)
2. How to check that it worked (convergence diagnostics — covered below)
3. How to interpret the output (posterior means and credible intervals)

### Non-centered parameterisation — mandatory for hierarchical models

Before writing the model code, there is one implementation detail that is non-negotiable: non-centered parameterisation.

The obvious (but wrong) way to write the district random effects:

```python
# CENTERED — do not use this
u_district = pm.Normal("u_district", mu=0, sigma=sigma_district, dims="district")
```

The problem: when sigma_district is near zero (all districts are similar), u_district and sigma_district become highly correlated in the posterior. The posterior geometry forms a "funnel". NUTS cannot pick a step size that works in both the narrow neck and the wide mouth of the funnel simultaneously. The sampler under-samples the region near sigma → 0, biasing the variance component estimates downward. Your credibility factors will be systematically too low — more shrinkage than the data warrant — without any obvious warning.

The correct way — non-centered parameterisation:

```python
# NON-CENTERED — always use this for hierarchical models
u_district_raw = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
u_district = pm.Deterministic("u_district", u_district_raw * sigma_district, dims="district")
```

This decouples the raw offset (u_district_raw, which is standard Normal) from the scale (sigma_district). The posterior geometry is now approximately Gaussian, and NUTS samples it efficiently. This is not optional — it is the standard practice for hierarchical models in PyMC.

### Building and fitting the model

Create a new markdown cell:

```python
%md
## Part 7: Bayesian hierarchical model
```

Create the next cell:

```python
# Prepare numpy arrays for PyMC
# PyMC works with numpy arrays, not Polars DataFrames directly.
# We need to convert our Polars data to numpy for the likelihood.

# Use district-level totals (aggregated over all years)
# For the Bayesian model, we pass total claims and total earned years per district
segments = dist_totals.sort("postcode_district")   # sort for reproducibility

# Encode districts as integer indices
# PyMC identifies array positions by integer index, not by string name.
# We need a mapping from district name to integer.
districts_sorted = segments["postcode_district"].to_list()
district_to_idx = {d: i for i, d in enumerate(districts_sorted)}
n_districts_model = len(districts_sorted)

# Convert to numpy
district_idx_arr = np.array([district_to_idx[d] for d in districts_sorted])
claims_arr = segments["total_claims"].to_numpy().astype(int)
exposure_arr = segments["total_earned_years"].to_numpy()

# Portfolio log-rate: prior centre for alpha
log_mu_portfolio = np.log(claims_arr.sum() / exposure_arr.sum())

print(f"Districts in model:       {n_districts_model}")
print(f"Total claims in model:    {claims_arr.sum():,}")
print(f"Total exposure in model:  {exposure_arr.sum():,.0f} earned years")
print(f"Portfolio log-rate:       {log_mu_portfolio:.4f}  (= log({np.exp(log_mu_portfolio):.4f}))")
print()
print("Exposure range:")
print(f"  Min: {exposure_arr.min():,.0f} earned years  (thinnest district)")
print(f"  Max: {exposure_arr.max():,.0f} earned years  (densest district)")
```

**What this does:** Converts the Polars DataFrame to numpy arrays in the format PyMC expects. The `district_idx_arr` is an integer array that maps each observation to its district. Because `segments` is sorted by district name and `districts_sorted` is the same sorted list, `district_idx_arr` is simply `[0, 1, 2, ..., 119]` — but writing it explicitly makes the indexing transparent.

**Run this cell.**

**What you should see:** 120 districts, total claims around 30,000-50,000 (depending on the random seed), exposure spanning from a thin district with ~100 earned years to a dense district with potentially 20,000+ earned years.

Now define and fit the model:

```python
# PyMC uses a "with pm.Model() as model_name:" context manager.
# Everything inside the with block is part of the model definition.
# This is just Python convention - the model object collects all the
# random variables defined inside it.

coords = {"district": districts_sorted}   # named coordinates for the random effects

with pm.Model(coords=coords) as hierarchical_model:

    # --- Priors ---
    # alpha: global log-rate intercept.
    # Prior: Normal centred on the observed log portfolio frequency.
    # SD = 0.5 on log scale: allows prior range of ~0.6x to ~1.65x portfolio mean at ±1 SD.
    alpha = pm.Normal("alpha", mu=log_mu_portfolio, sigma=0.5)

    # sigma_district: between-district log-rate standard deviation.
    # Prior: HalfNormal(0.3).
    # This constrains sigma to be positive and places most mass below ~0.6 log points.
    # On the rate scale: most districts within ~0.5x to ~2.0x the portfolio mean.
    sigma_district = pm.HalfNormal("sigma_district", sigma=0.3)

    # --- Non-centered parameterisation (mandatory) ---
    # u_district_raw: raw district offsets, standard Normal.
    # u_district: actual district log-rate deviations = raw * scale.
    # The "dims='district'" argument labels the array with the district names
    # defined in coords — this makes results easier to extract later.
    u_district_raw = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
    u_district = pm.Deterministic(
        "u_district", u_district_raw * sigma_district, dims="district"
    )

    # --- Linear predictor ---
    # log(lambda_i) = alpha + u_district[district_index_i]
    # district_idx_arr maps each observation to its district's log-rate deviation.
    log_lambda = alpha + u_district[district_idx_arr]

    # --- Likelihood ---
    # Poisson distribution: claims_i ~ Poisson(lambda_i * exposure_i)
    # pm.math.exp(log_lambda) converts from log space to rate space.
    # Multiplying by exposure_arr gives the expected claim count E[claims_i].
    claims_obs = pm.Poisson(
        "claims_obs",
        mu=pm.math.exp(log_lambda) * exposure_arr,
        observed=claims_arr,
    )

# Print a model summary so we can check the structure before sampling.
# This shows all random variables and their shapes.
print("Model structure:")
print(hierarchical_model.debug())
```

**What this does:** Defines the hierarchical Poisson model. The `with pm.Model()` block is how PyMC knows which variables belong to the model. Nothing is computed yet — this is just the model definition.

**Run this cell.**

**What you should see:** A text description of the model showing the random variables (`alpha`, `sigma_district`, `u_district_raw`, `u_district`, `claims_obs`) and their shapes and distributions. If you see an error here, it is almost always a shape mismatch — check that `district_idx_arr`, `claims_arr`, and `exposure_arr` all have length 120.

Now sample. This is the slow step — it runs MCMC:

```python
# pm.sample() runs NUTS (No-U-Turn Sampler) to draw samples from the posterior.
#
# Parameters:
#   draws=1000:          Number of posterior samples per chain to keep.
#   tune=1000:           Number of warmup/adaptation steps before keeping samples.
#                        During tuning, NUTS adapts its step size and mass matrix.
#                        Tuning samples are discarded.
#   chains=4:            Number of independent Markov chains to run.
#                        Running 4 chains lets us check convergence with R-hat.
#   target_accept=0.90:  Target acceptance probability. Higher values → smaller
#                        step sizes → slower but more thorough exploration.
#                        0.90 is appropriate for most hierarchical models.
#   return_inferencedata=True: Return an ArviZ InferenceData object (recommended).
#   random_seed=42:      For reproducibility.
#
# This takes approximately 3-6 minutes on a Databricks Community Edition cluster.
# The progress bar shows chains running in parallel if cores > 1.
# Do not stop it early — incomplete chains cannot be used for diagnostics.

print("Fitting hierarchical Bayesian model via NUTS...")
print("Expected time: 3-6 minutes on Databricks Community Edition.")
print()

with hierarchical_model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
    )

print()
print("Sampling complete.")
```

**What this does:** Runs NUTS sampling. PyMC will print a progress bar showing the sampling status for each chain.

**Run this cell and wait.** On Databricks Community Edition (single node), this takes 3-6 minutes for 120 districts. A multi-core worker cluster would be faster — see the Databricks Deployment section later in this module.

**What you should see during sampling:** A progress bar like:
```
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 2 jobs)
NUTS: [alpha, sigma_district, u_district_raw]
 |████████████| 100.00% [8000/8000 00:03<00:00 Sampling 4 chains, 0 divergences]
```

The number of divergences should be 0. If you see divergences, it is not a catastrophic error, but it needs investigation — covered in the next section.