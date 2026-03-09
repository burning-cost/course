# Module 6 Exercises: Credibility and Bayesian Methods

Four exercises. Work through them in order — each one builds on variables and functions defined in earlier exercises and in the main tutorial notebook. The solutions are at the end of each exercise. Try the exercise first before reading the solution.

Before starting, make sure you have run all cells in the tutorial notebook and have the following in your session:
- `df` — the district-year DataFrame (600 rows: 120 districts × 5 years)
- `dist_totals` — district-level aggregates (120 rows)
- `dist_year` — the filtered district-year data passed to `buhlmann_straub()`
- `bs` — the Bühlmann-Straub results dictionary from the tutorial
- `bs_results` — `bs["results"]`, the per-district Polars DataFrame
- `hierarchical_model` — the fitted PyMC model
- `trace` — the MCMC trace (ArviZ InferenceData)
- `results` — the per-district posterior means and intervals DataFrame
- `buhlmann_straub()` — the function defined in the tutorial

If your cluster has timed out or restarted, re-run all tutorial cells from the top before attempting these exercises.

**Note:** the  function is defined in Part 4 of the tutorial. If you are starting fresh in a new notebook, copy the full function definition from Part 4 before attempting these exercises — it is not imported from any library.

---

## Exercise 1: Bühlmann-Straub for severity data

### What this teaches

The tutorial applied Bühlmann-Straub to claim frequency (claims per earned year). This exercise applies it to claim severity (average incurred cost per claim). The mechanics are identical — the difference is in what the input data represent and what the credibility estimate means for pricing.

Thin cells are just as problematic for severity as for frequency. A district with 8 claims has a noisy average severity dominated by the particular mix of large and small losses that happened to fall there in the observation period. A single £50,000 bodily injury claim in a district with 8 total claims doubles the apparent average severity. Credibility weighting pulls that estimate back toward the portfolio mean.

### Context: adding severity to the synthetic data

The tutorial's synthetic dataset only has claim counts, not claim amounts. We need to add severity. Create a new cell in your notebook:

```python
%md
## Exercise 1: Credibility weighting for severity
```

Create the next cell and run it to add synthetic severity to the existing data:

```python
import numpy as np
import polars as pl

# Add synthetic incurred amounts to the df DataFrame.
# We need a random seed that does not conflict with the tutorial's seed=42.
rng_sev = np.random.default_rng(seed=99)

# True mean severity per district — correlated with frequency but not identical.
# Districts with high claim frequency tend to have slightly lower mean severity
# (urban motor tends to have frequent small claims rather than rare large ones).
# sigma = 0.40 on log scale gives moderate between-district severity variation.
TRUE_SIGMA_SEV = 0.40
true_sev_log_deviations = rng_sev.normal(0, TRUE_SIGMA_SEV, size=120)

PORTFOLIO_MEAN_SEVERITY = 3800.0  # £3,800 mean claim cost (UK motor benchmark)

# Map district names to their severity deviations
district_names_sorted = sorted(df["postcode_district"].unique().to_list())
sev_dev_map = {d: true_sev_log_deviations[i] for i, d in enumerate(district_names_sorted)}

# Add incurred amounts to df
# For rows with zero claims, incurred = 0.
# For rows with claims, generate Gamma-distributed total costs.
# Gamma shape = 2.0 gives moderate positive skew (typical for motor severity).
incurred_list = []
for row in df.to_dicts():
    district = row["postcode_district"]
    n_claims = row["claim_count"]
    if n_claims == 0:
        incurred_list.append(0.0)
    else:
        mean_sev = PORTFOLIO_MEAN_SEVERITY * np.exp(sev_dev_map[district])
        # Gamma parameterised by shape and scale: mean = shape × scale
        shape = 2.0
        scale = mean_sev / shape
        # Total incurred = sum of individual claim amounts
        individual_amounts = rng_sev.gamma(shape=shape, scale=scale, size=n_claims)
        incurred_list.append(float(individual_amounts.sum()))

df = df.with_columns(pl.Series("incurred", incurred_list))

print(f"Dataset now has incurred column.")
print(f"\nSample of rows with claims:")
print(df.filter(pl.col("claim_count") > 0).head(8))
print()
print(f"Portfolio mean severity (claims-only rows):")
df_claims = df.filter(pl.col("claim_count") > 0)
total_incurred = df_claims["incurred"].sum()
total_claims = df_claims["claim_count"].sum()
print(f"  {total_incurred / total_claims:,.0f}  (target: £{PORTFOLIO_MEAN_SEVERITY:,.0f})")
```

**What this does:** Adds a column of simulated incurred claim amounts to the existing `df` DataFrame. Rows with zero claims get incurred = 0. Rows with claims get Gamma-distributed totals — a positively skewed distribution that matches the shape of motor claim cost distributions in practice.

**Run this cell.** You should see the portfolio mean severity close to £3,800.

### Setting up severity credibility data

```python
# Filter to rows with at least one claim - severity is undefined for zero-claim rows.
# Also filter out rows where claim_count > 0 but incurred = 0, which would indicate
# a data quality issue (notification record without payment record, common in
# UK claims systems when notification and payment are handled in separate databases).
claims_df = df.filter(
    (pl.col("claim_count") > 0) & (pl.col("incurred") > 0)
)

print(f"Rows with valid severity data: {claims_df.height}  (out of {df.height} total)")
print()

# Compute average severity per (district, accident_year)
# Weight by claim count — severity is averaged over claims, not policy-years.
# A district with 10 claims contributes 10 data points to the severity pool;
# a district with 1 claim contributes 1.
sev_year = (
    claims_df
    .group_by(["postcode_district", "accident_year"])
    .agg([
        # Total incurred ÷ total claims = weighted average severity for this group-year
        (pl.col("incurred").sum() / pl.col("claim_count").sum()).alias("avg_severity"),
        pl.col("claim_count").sum().alias("claim_count"),
    ])
    .sort(["postcode_district", "accident_year"])
)

print(f"District-year severity rows: {sev_year.height}")
print()
print("Preview:")
print(sev_year.head(10))
```

**Run this cell.** You should have fewer rows than 600 (some district-years have zero claims and are excluded). The `avg_severity` column shows the mean claim cost for that district in that year; `claim_count` is the exposure weight for the severity credibility calculation.

### Tasks

**Task 1:** Fit `buhlmann_straub` on severity data.

Call the function with `value_col="avg_severity"` and `weight_col="claim_count"`. For severity, the weight is the number of claims, not the number of policy-years — severity is averaged over individual claims, not over earned years. Use `log_transform=False` to work in additive space (average severity in pounds), or `log_transform=True` if you want multiplicative severity relativities.

Print the structural parameters. What does v represent for severity data (hint: it is different from what it represents for frequency data)? What does K represent in terms of claims needed, rather than earned years?

**Task 2:** Compare K values for frequency and severity.

The frequency B-S produced a K in earned years. The severity B-S produces a K in claims. Print both. What does the ratio of K values tell you about the relative stability of frequency versus severity across districts?

**Task 3:** Identify the three districts with the highest severity credibility factor Z.

Why do they have high Z? Is it because they have many policy-years, or because they have many claims? For severity, which quantity matters?

**Task 4:** Compute the credibility-weighted pure premium for each district.

Pure premium = frequency × severity. Use:
- `bs_results` (from the tutorial) for frequency credibility estimates
- Your severity credibility results for severity estimates

Multiply the two estimates to get a credibility-weighted pure premium. Then compute the naive observed pure premium (observed frequency × observed average severity). Find the five districts where the credibility-weighted pure premium deviates most from the naive observed pure premium. For each, determine whether the discrepancy is driven more by the frequency adjustment or the severity adjustment.

---

### Solution — Exercise 1

**Task 1: Fitting B-S on severity**

```python
# Fit B-S on severity data
# weight_col = "claim_count" because severity is averaged over claims
bs_sev = buhlmann_straub(
    data=sev_year,
    group_col="postcode_district",
    value_col="avg_severity",
    weight_col="claim_count",
    log_transform=False,   # additive: blend in £ space, not log-£ space
)

print("=== Severity structural parameters ===")
print()
print(f"Grand mean severity:  £{bs_sev['grand_mean']:,.0f}")
print(f"EPV (v):              {bs_sev['v_hat']:,.0f}")
print(f"  (within-district year-to-year variance in £ — how much average severity")
print(f"   fluctuates year to year within a district given its true average)")
print()
print(f"VHM (a):              {bs_sev['a_hat']:.2f}")
print(f"  (between-district variance in £ — how much true average severities")
print(f"   differ across districts)")
print()
print(f"K:                    {bs_sev['k']:.1f} claims")
print(f"  (claims needed for a district to achieve Z = 0.50 on severity)")
print()

sev_results = bs_sev["results"]
print("Per-district severity credibility (first 15):")
print(sev_results.head(15))
```

**Understanding v for severity:** For frequency, v was the within-district year-to-year variance of the *claim rate* (claims per earned year). For severity, v is the within-district year-to-year variance of the *average claim cost* (£ per claim). High v for severity means a district's average severity jumps around a lot from year to year — typically driven by occasional large bodily injury or theft claims that dramatically inflate one year's average.

**Task 2: Comparing K values**

```python
print(f"Frequency K:  {bs['k']:.0f} earned years  (years for Z = 0.50 on frequency)")
print(f"Severity K:   {bs_sev['k']:.0f} claims       (claims for Z = 0.50 on severity)")
print()
print("Interpretation:")
print(f"  A district needs {bs['k']:.0f} earned years for Z = 0.50 on frequency,")
print(f"  but {bs_sev['k']:.0f} claims for Z = 0.50 on severity.")
print()

# What does this ratio tell us?
# Severity K is typically much higher than frequency K because:
# - Within-district severity variance (EPV, v) is large: a single BI claim
#   can inflate the average by thousands of pounds
# - Between-district severity heterogeneity (VHM, a) is smaller than frequency
#   heterogeneity: districts differ more in how often claims happen than in
#   how costly those claims are
# The ratio K_sev / K_freq approximates how much harder it is to pin down
# a district's true average severity compared to its true frequency.

ratio = bs_sev['k'] / bs['k']
print(f"K_sev / K_freq ≈ {ratio:.1f}")
print(f"  Severity is about {ratio:.1f}× harder to pin down than frequency")
print(f"  for the same number of data points.")
```

**Task 3: Districts with highest severity Z**

```python
# Sort by Z descending — highest Z = most credibility on severity
top_z_sev = sev_results.sort("Z", descending=True).head(3)

print("Top 3 districts by severity credibility factor Z:")
print(top_z_sev.select([
    "postcode_district",
    "exposure",          # this is total claim count, not earned years, for severity B-S
    "obs_mean",
    "Z",
    "credibility_estimate",
]))
print()
print("Note: 'exposure' here is total claim COUNT, not earned years.")
print("A district has high severity Z because it has accumulated many claims,")
print("not because it has high exposure in policy-years.")
print()
print(f"For Z = 0.50 on severity, need K = {bs_sev['k']:.0f} claims.")
print(f"For Z = 0.80 on severity, need 4K = {4*bs_sev['k']:.0f} claims.")
```

The districts with high severity Z are not necessarily the districts with the most policy-years. They are the districts with the most *claims*. A district with 5,000 policy-years but only 50 claims (1% frequency) has only 50 data points for its severity estimate. A district with 2,000 policy-years and 200 claims (10% frequency) has 200 severity data points. For severity credibility, claim count is the correct exposure weight.

**Task 4: Credibility-weighted pure premium**

```python
# Step 1: Get frequency and severity credibility estimates per district
freq_cred = bs_results.select([
    "postcode_district",
    pl.col("credibility_estimate").alias("freq_cred"),
    pl.col("obs_mean").alias("obs_freq"),
])

sev_cred = sev_results.select([
    "postcode_district",
    pl.col("credibility_estimate").alias("sev_cred"),
    pl.col("obs_mean").alias("obs_sev"),
])

# Step 2: District-level observed pure premium
# Need districts with both frequency and severity estimates
# (districts with zero claims across all years have no severity estimate)
dist_pp = (
    dist_totals
    .filter(pl.col("total_earned_years") > 0.5)
    .with_columns([
        (pl.col("total_claims") / pl.col("total_earned_years")).alias("obs_freq_raw"),
    ])
)

# Join all together
pp = (
    dist_pp
    .join(freq_cred, on="postcode_district", how="inner")
    .join(sev_cred,  on="postcode_district", how="inner")
    .with_columns([
        # Naive observed pure premium
        (pl.col("obs_freq_raw") * pl.col("obs_sev")).alias("obs_pp"),
        # Credibility-weighted pure premium
        (pl.col("freq_cred") * pl.col("sev_cred")).alias("cred_pp"),
    ])
    .with_columns([
        # Percentage difference: how much does credibility change the pure premium?
        ((pl.col("cred_pp") - pl.col("obs_pp")) / pl.col("obs_pp") * 100).alias("pp_diff_pct"),
        # Contribution from frequency adjustment
        ((pl.col("freq_cred") - pl.col("obs_freq_raw")) / pl.col("obs_freq_raw") * 100).alias("freq_contribution_pct"),
        # Contribution from severity adjustment
        ((pl.col("sev_cred") - pl.col("obs_sev")) / pl.col("obs_sev") * 100).alias("sev_contribution_pct"),
    ])
)

# Five districts with largest absolute deviation
pp_divergent = (
    pp
    .with_columns(pl.col("pp_diff_pct").abs().alias("abs_diff"))
    .sort("abs_diff", descending=True)
    .head(5)
)

print("5 districts with largest credibility adjustment to pure premium:")
print(pp_divergent.select([
    "postcode_district", "total_earned_years", "total_claims",
    "obs_pp", "cred_pp", "pp_diff_pct",
    "freq_contribution_pct", "sev_contribution_pct",
]))
```

**What you should find:** For thin districts, frequency adjustments dominate — with few earned years, the frequency estimate is pulled strongly toward the portfolio mean. Severity adjustments are typically smaller in magnitude for the same district, because severity requires claims as the exposure weight (a thin district by earned years may still have accumulated enough claims for moderate severity credibility).

**An important caveat:** The credibility-weighted pure premium is not the same as applying credibility to pure premium directly. Separately credibility-weighting frequency and severity and then multiplying is an approximation. If a district's frequency and severity adjustments are correlated — both pulling in the same direction because the district is generally high-risk — the combined pure premium adjustment will not precisely equal the product of the two individual credibility factors. For the thinnest districts, where both frequency and severity have low Z, this approximation error is typically below 2%.

---

## Exercise 2: Two-level geographic hierarchy in PyMC

### What this teaches

The flat hierarchical model in the tutorial treats all 120 districts as independently exchangeable — each district's log-rate is drawn from the same Normal distribution. For UK motor, this is a simplification. Districts in the same postcode area (the two-letter prefix: SW, KT, EC, etc.) share road networks, demographics, parking conditions, and crime rates. They are more similar to each other than to districts in other areas.

This exercise builds a two-level hierarchical model: districts nest within areas, and each level gets its own partial pooling. A thin district in the SW area borrows strength from other SW districts, not from rural Scottish ones.

### Setting up the two-level data

Create a new cell:

```python
%md
## Exercise 2: Two-level geographic hierarchy
```

```python
import numpy as np
import polars as pl
import pymc as pm
import arviz as az

# Derive postcode area from the first two characters of the district name.
# In real UK postcode data this would be the alphabetic prefix (SW, KT, etc.).
# Our synthetic names are formatted as "SW1", "KT4", etc. — take first two chars.
df_with_area = df.with_columns(
    pl.col("postcode_district").str.slice(0, 2).alias("postcode_area")
)

# Aggregate to district level (summing across years, as before)
dist_totals_ex2 = (
    df_with_area
    .group_by(["postcode_area", "postcode_district"])
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
    .sort(["postcode_area", "postcode_district"])
)

n_districts_ex2 = dist_totals_ex2.height
n_areas_ex2 = dist_totals_ex2["postcode_area"].n_unique()

print(f"Districts: {n_districts_ex2}")
print(f"Areas:     {n_areas_ex2}")
print()
print("Number of districts per area:")
area_sizes = (
    dist_totals_ex2
    .group_by("postcode_area")
    .agg(pl.col("postcode_district").n_unique().alias("n_districts"))
    .sort("n_districts", descending=True)
)
print(area_sizes.describe())
print()
print("Areas with fewest districts:")
print(area_sizes.sort("n_districts").head(10))
```

**Run this cell.** This shows the area structure — how many districts fall within each two-letter postcode area in the synthetic dataset.

### Tasks

**Task 1:** Fit a two-level hierarchical PyMC model with area effects and district-within-area effects.

The model structure is:

```python
claims_i ~ Poisson(lambda_i × exposure_i)
log(lambda_i) = alpha + u_area[area[i]] + u_district[district[i]]

u_area[k]     ~ Normal(0, sigma_area)       [area-level deviations]
u_district[k] ~ Normal(0, sigma_district)   [district-within-area deviations]

alpha          ~ Normal(log(mu_portfolio), 0.5)
sigma_area     ~ HalfNormal(0.3)
sigma_district ~ HalfNormal(0.3)
```

Both random effects must use non-centered parameterisation. Below is the integer encoding setup — complete the model definition:

```python
# Encode areas and districts as integer indices for PyMC
areas_ex2 = dist_totals_ex2["postcode_area"].unique().sort().to_list()
districts_ex2 = dist_totals_ex2["postcode_district"].unique().sort().to_list()

area_to_idx_ex2 = {a: i for i, a in enumerate(areas_ex2)}
district_to_idx_ex2 = {d: i for i, d in enumerate(districts_ex2)}

area_idx_arr_ex2 = np.array([
    area_to_idx_ex2[a] for a in dist_totals_ex2["postcode_area"].to_list()
])
district_idx_arr_ex2 = np.array([
    district_to_idx_ex2[d] for d in dist_totals_ex2["postcode_district"].to_list()
])
claims_ex2 = dist_totals_ex2["claims"].to_numpy().astype(int)
exposure_ex2 = dist_totals_ex2["earned_years"].to_numpy()

log_mu_portfolio_ex2 = np.log(claims_ex2.sum() / exposure_ex2.sum())
coords_ex2 = {"area": areas_ex2, "district": districts_ex2}

print(f"Areas: {len(areas_ex2)},  Districts: {len(districts_ex2)}")
print(f"Portfolio log-rate: {log_mu_portfolio_ex2:.4f}")
```

**Run this cell.** Then define and fit the model:

```python
with pm.Model(coords=coords_ex2) as nested_model:

    alpha_n = pm.Normal("alpha", mu=log_mu_portfolio_ex2, sigma=0.5)

    # Area-level random effects (non-centered)
    sigma_area = pm.HalfNormal("sigma_area", sigma=0.3)
    u_area_raw = pm.Normal("u_area_raw", mu=0, sigma=1, dims="area")
    u_area = pm.Deterministic("u_area", u_area_raw * sigma_area, dims="area")

    # District-within-area random effects (non-centered)
    sigma_district_nested = pm.HalfNormal("sigma_district_nested", sigma=0.3)
    u_district_raw_ex2 = pm.Normal("u_district_raw_ex2", mu=0, sigma=1, dims="district")
    u_district_ex2 = pm.Deterministic(
        "u_district_ex2",
        u_district_raw_ex2 * sigma_district_nested,
        dims="district",
    )

    # Log-rate: global intercept + area effect + district-within-area effect
    log_lambda_ex2 = alpha_n + u_area[area_idx_arr_ex2] + u_district_ex2[district_idx_arr_ex2]

    claims_obs_ex2 = pm.Poisson(
        "claims_obs_ex2",
        mu=pm.math.exp(log_lambda_ex2) * exposure_ex2,
        observed=claims_ex2,
    )

print("Fitting two-level nested model. Expected time: 4-8 minutes...")

with nested_model:
    trace_nested = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.92,    # slightly higher than default: nested models need more care
        return_inferencedata=True,
        random_seed=42,
    )

print("Sampling complete.")
```

**What this does:** Fits a two-level hierarchical model where district effects are nested within area effects. The `target_accept=0.92` is slightly higher than the tutorial's 0.90 — nested models have more complex posterior geometry and benefit from a smaller step size. This takes 4-8 minutes.

**Task 2:** Check convergence for the two-level model.

Run the same convergence diagnostics as the tutorial. Pay particular attention to `sigma_area` — the area-level variance component. With few areas (our synthetic dataset has around 10-20 two-letter prefixes), the ESS for `sigma_area` may be lower than the ESS for `sigma_district_nested`. What does this tell you?

**Task 3:** Compare the two-level model estimates to the flat model from the tutorial.

For the same districts, compare `posterior_mean` (from `results` in the tutorial) to the two-level model estimates. Which districts diverge most? Are they thin districts or dense districts? Are they in areas that are distinctively high or low risk?

**Task 4:** Report the posterior for `sigma_area` and `sigma_district_nested`.

If `sigma_area > sigma_district_nested`, what does that tell you about the geographic structure of the portfolio? If `sigma_district_nested > sigma_area`, what does it tell you?

---

### Solution — Exercise 2

**Task 1:** See the model code above. The key implementation points:
- Two HalfNormal priors: one for area-level SD, one for district-within-area SD
- Both random effects use non-centered parameterisation
- `target_accept=0.92` to handle the more complex geometry of the nested model

**Task 2: Convergence for the nested model**

```python
rhat_nested = az.rhat(trace_nested)
ess_nested = az.ess(trace_nested, method="bulk")

max_rhat_nested = float(rhat_nested.max().to_array().max())
min_ess_nested = float(ess_nested.min().to_array().min())
n_div_nested = int(trace_nested.sample_stats["diverging"].sum())

print("=== Two-level model convergence ===")
print(f"Max R-hat:               {max_rhat_nested:.4f}  ({'OK' if max_rhat_nested < 1.01 else 'INVESTIGATE'})")
print(f"Min ESS (bulk):          {min_ess_nested:.0f}  ({'OK' if min_ess_nested > 400 else 'LOW'})")
print(f"Divergences:             {n_div_nested}  ({'OK' if n_div_nested == 0 else 'INVESTIGATE'})")
print()

# Area-level variance specifically
sigma_area_posterior = trace_nested.posterior["sigma_area"]
sigma_area_ess = float(ess_nested["sigma_area"])

print(f"sigma_area posterior:")
print(f"  Mean:   {float(sigma_area_posterior.mean()):.4f}")
print(f"  SD:     {float(sigma_area_posterior.std()):.4f}")
print(f"  P5:     {float(np.percentile(sigma_area_posterior.values, 5)):.4f}")
print(f"  P95:    {float(np.percentile(sigma_area_posterior.values, 95)):.4f}")
print(f"  ESS:    {sigma_area_ess:.0f}  ({'OK' if sigma_area_ess > 1000 else 'LOW — consider more draws'})")
print()

sigma_dist_posterior = trace_nested.posterior["sigma_district_nested"]
sigma_dist_ess = float(ess_nested["sigma_district_nested"])
print(f"sigma_district_nested posterior:")
print(f"  Mean:   {float(sigma_dist_posterior.mean()):.4f}")
print(f"  ESS:    {sigma_dist_ess:.0f}")
```

**Why ESS for sigma_area may be low:** The area-level variance component is estimated from fewer data points than the district-level component. With 10-20 areas, the posterior for `sigma_area` is harder to characterise precisely. If `sigma_area` ESS is below 1000, increase `draws=2000` for the final model run. This does not indicate a model flaw — it is an honest reflection of the limited information about between-area heterogeneity.

**Task 3: Compare flat vs nested estimates**

```python
# Extract posterior means from the nested model
alpha_nested_mean = float(trace_nested.posterior["alpha"].mean())
u_district_nested_mean = trace_nested.posterior["u_district_ex2"].mean(
    dim=("chain", "draw")
).values
u_area_nested_mean = trace_nested.posterior["u_area"].mean(
    dim=("chain", "draw")
).values

# The log-rate for each district = alpha + area effect + district-within-area effect
nested_log_rate = (
    alpha_nested_mean
    + u_area_nested_mean[area_idx_arr_ex2]
    + u_district_nested_mean[district_idx_arr_ex2]
)
nested_posterior_mean = np.exp(nested_log_rate)

# Build a DataFrame with nested model estimates
nested_results = pl.DataFrame({
    "postcode_district": dist_totals_ex2["postcode_district"].to_list(),
    "nested_estimate":   nested_posterior_mean.tolist(),
    "earned_years":      exposure_ex2.tolist(),
})

# Compare to flat model results
flat_for_join = results.select([
    "postcode_district",
    pl.col("posterior_mean").alias("flat_estimate"),
])

comparison_ex2 = (
    flat_for_join
    .join(nested_results, on="postcode_district", how="inner")
    .with_columns([
        ((pl.col("nested_estimate") - pl.col("flat_estimate")) / pl.col("flat_estimate") * 100)
        .alias("pct_difference"),
    ])
)

# Districts where nested and flat estimates diverge most
largest_divergence = (
    comparison_ex2
    .with_columns(pl.col("pct_difference").abs().alias("abs_diff"))
    .sort("abs_diff", descending=True)
    .head(10)
)

print("Districts where flat and nested estimates diverge most:")
print(largest_divergence.select([
    "postcode_district", "earned_years",
    "flat_estimate", "nested_estimate", "pct_difference",
]))
```

**What you should see:** The largest divergences will be in thin districts that happen to be in areas with distinctive risk profiles. A thin district in a uniformly high-risk area will be pulled upward by the nested model (which borrows strength from neighbouring high-risk districts) but only to the portfolio mean by the flat model. Dense districts will show minimal divergence regardless of area — they have enough data to determine their own rate.

**Task 4: Interpreting sigma_area vs sigma_district_nested**

```python
sigma_area_mean = float(trace_nested.posterior["sigma_area"].mean())
sigma_dist_mean = float(trace_nested.posterior["sigma_district_nested"].mean())
ratio = sigma_area_mean / sigma_dist_mean

print("=== Variance component comparison ===")
print(f"sigma_area:              {sigma_area_mean:.4f}  (between-area log-SD)")
print(f"sigma_district_nested:   {sigma_dist_mean:.4f}  (between-district-within-area log-SD)")
print(f"Ratio (area/district):   {ratio:.2f}")
print()

if ratio > 1.0:
    print("Area-level variance DOMINATES.")
    print("Most geographic variation is between areas (SW vs KT vs IV),")
    print("not between individual districts within the same area.")
    print("Implication: area-level rating is the appropriate granularity.")
    print("Adding district-level splits may be modelling noise, not signal.")
elif ratio > 0.5:
    print("Area and district variance are COMPARABLE.")
    print("Both levels of geography carry genuine pricing signal.")
    print("District-level rating is justified, and the two-level hierarchy is helping")
    print("smooth estimates for the thinnest districts.")
else:
    print("District-level variance DOMINATES.")
    print("Most geographic variation is between districts within the same area.")
    print("Risk is driven by local factors (road density, deprivation, parking)")
    print("that vary at district granularity, not area granularity.")
    print("District-level rating is strongly justified.")
```

**On exchangeability in the nested model:** The nested model assumes districts within an area are exchangeable — drawn from the same area-level distribution. This is more defensible than the flat model's assumption that all 120 districts are exchangeable. But it still assumes districts within SW are more similar to each other than to districts in KT, which is true of the labelling but not necessarily of the underlying risk (SW1 and SW19 are very different places). The nested model is a pragmatic improvement, not a complete solution to geographic correlation.

---

## Exercise 3: Posterior predictive checks

### What this teaches

MCMC convergence (good R-hat, high ESS, zero divergences) tells you the sampler has explored the posterior correctly. It does not tell you the model is right. A misspecified model can converge perfectly and produce posterior means that are systematically wrong. Posterior predictive checks (PPCs) are how you detect misspecification.

A PPC works like this:
1. Draw a sample of parameters from the posterior (we have 4,000 such samples)
2. For each parameter sample, simulate a new dataset from the model
3. Compare properties of the simulated datasets to the actual observed data

If the model is well-specified, the observed data should look like a typical sample from the posterior predictive distribution. If the observed data look unusual relative to the simulations — for example, the observed total is in the extreme tail of simulated totals — the model is misspecified in some way.

### Setting up

Create a new cell:

```python
%md
## Exercise 3: Posterior predictive checks
```

The exercise uses `hierarchical_model` and `trace` from the tutorial. These must be in your session.

### Task 1: Overall distribution check

Simulate replicated datasets from the posterior predictive distribution. For each simulation, compute the total number of claims. Plot the distribution of simulated totals and mark the observed total.

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Sample from the posterior predictive distribution.
# pm.sample_posterior_predictive() draws parameter samples from the posterior
# and then generates new claim counts from the Poisson likelihood.
# This gives us datasets "as if" the model had generated the data.
with hierarchical_model:
    ppc_samples = pm.sample_posterior_predictive(trace, random_seed=42)

# ppc_samples.posterior_predictive["claims_obs"] has shape:
# (chains, draws, districts) = (4, 1000, 120)
# Reshape to (total_samples, districts) = (4000, 120)
sim_claims = ppc_samples.posterior_predictive["claims_obs"].values.reshape(
    -1, n_districts_model
)

# Simulated total claims per replicated dataset
sim_totals = sim_claims.sum(axis=1)    # shape: (4000,)
obs_total = int(claims_arr.sum())

print(f"Observed total claims: {obs_total:,}")
print(f"Simulated total claims — mean: {sim_totals.mean():,.0f},  SD: {sim_totals.std():,.0f}")
print(f"Simulated P5:  {np.percentile(sim_totals, 5):,.0f}")
print(f"Simulated P95: {np.percentile(sim_totals, 95):,.0f}")
```

**Run this cell.** The observed total should be comfortably within the P5-P95 range of simulated totals. If it falls outside, the model's calibration is off — investigate the alpha prior.

Now plot it:

```python
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(sim_totals, bins=50, alpha=0.7, color="steelblue",
        label="Posterior predictive simulations")
ax.axvline(obs_total, color="crimson", lw=2.5, linestyle="-",
           label=f"Observed: {obs_total:,}")
ax.set_xlabel("Total simulated claims (all districts)")
ax.set_ylabel("Frequency across 4,000 simulations")
ax.set_title("Posterior predictive check: total claims")
ax.legend()
plt.tight_layout()
display(fig)
plt.close(fig)

# Posterior predictive p-value:
# Fraction of simulations that produce at least as many total claims as observed.
# Should be between 0.05 and 0.95 for a well-calibrated model.
pvalue = (sim_totals >= obs_total).mean()
print(f"\nPosterior predictive p-value (total claims): {pvalue:.3f}")
print("(0.05–0.95 indicates good calibration; outside this range indicates misspecification)")
```

**What you should see:** The observed total as a vertical red line sitting inside the bulk of the histogram. A p-value between 0.05 and 0.95. With synthetic Poisson data and a correctly specified model, the observed total should be near the centre of the distribution (p-value near 0.50).

### Task 2: Overdispersion check

The Poisson likelihood assumes variance equals mean. If the data are overdispersed (variance greater than mean), the Poisson model understates uncertainty. Check this with the variance-to-mean ratio (VMR):

```python
# Observed VMR across districts (each district is one data point)
obs_vmr = claims_arr.var() / claims_arr.mean()
print(f"Observed VMR across districts: {obs_vmr:.2f}")
print(f"  (A Poisson process would give VMR = 1.0)")
print(f"  (VMR >> 1 suggests overdispersion; consider Negative Binomial)")
print()

# Simulated VMR for each replicated dataset
sim_vmrs = sim_claims.var(axis=1) / sim_claims.mean(axis=1)
print(f"Simulated VMR — mean: {sim_vmrs.mean():.2f},  P5: {np.percentile(sim_vmrs, 5):.2f},  P95: {np.percentile(sim_vmrs, 95):.2f}")
print()

# Is the observed VMR within the simulated range?
if obs_vmr > np.percentile(sim_vmrs, 95):
    print("OVERDISPERSION DETECTED: observed VMR exceeds 95th percentile of simulations.")
    print("The Poisson model is understating claim count variance across districts.")
    print("Consider switching to Negative Binomial likelihood (see Task 4).")
elif obs_vmr < np.percentile(sim_vmrs, 5):
    print("UNDERDISPERSION: observed VMR is below 5th percentile. Unusual.")
else:
    print("VMR is within the simulated range. Poisson dispersion appears adequate.")
```

**What you should see with synthetic Poisson data:** The observed VMR within the simulated range, since the data were generated from a Poisson process. In real insurance data, moderate overdispersion is common — unobserved factors (individual driving behaviour, vehicle actual condition) create extra between-policy variance that the Poisson model does not capture.

### Task 3: Tail check on thin districts

Thin districts are the most likely to be misrepresented. Check whether the model's uncertainty intervals capture the observed data for the thinnest districts:

```python
# Identify thin districts: less than 100 earned years in total
thin_mask = exposure_arr < 100
n_thin = thin_mask.sum()
print(f"Thin districts (< 100 earned years): {n_thin}")

if n_thin == 0:
    print("No districts below 100 earned years threshold.")
    print("Try lowering the threshold to exposure_arr < 200.")
    thin_mask = exposure_arr < 200
    n_thin = thin_mask.sum()
    print(f"Thin districts (< 200 earned years): {n_thin}")

# 90% posterior predictive interval for each thin district
lower_5_thin = np.percentile(sim_claims[:, thin_mask], 5, axis=0)
upper_95_thin = np.percentile(sim_claims[:, thin_mask], 95, axis=0)
obs_thin = claims_arr[thin_mask]

# How many thin districts' observed claims fall within the 90% interval?
within_interval = (obs_thin >= lower_5_thin) & (obs_thin <= upper_95_thin)
coverage = within_interval.mean()

print(f"\n90% PPC coverage for thin districts: {coverage*100:.1f}%  (target: ~90%)")
print()

# Report districts outside the interval
outside_mask = ~within_interval
n_outside = outside_mask.sum()
if n_outside > 0:
    thin_district_names = [d for i, d in enumerate(districts_sorted) if thin_mask[i]]
    print(f"Thin districts outside 90% predictive interval ({n_outside} districts):")
    for idx in np.where(outside_mask)[0]:
        print(f"  {thin_district_names[idx]}: observed={obs_thin[idx]},  "
              f"interval=[{lower_5_thin[idx]:.0f}, {upper_95_thin[idx]:.0f}]")
else:
    print("All thin districts within 90% posterior predictive interval.")
```

**What you should see:** Coverage close to 90% (allow ±5%). If coverage is substantially below 90% — say, 70% — the model is systematically underestimating uncertainty for thin districts. This typically indicates overdispersion (the Negative Binomial is more appropriate) or that the prior on sigma_district is too tight (widening `HalfNormal(0.3)` to `HalfNormal(0.5)` would help).

### Task 4: Model comparison — Poisson versus Negative Binomial

Fit a Negative Binomial variant of the same model and compare using LOO (Leave-One-Out cross-validation):

```python
# Negative Binomial variant.
# The only changes from the Poisson model:
# 1. pm.NegativeBinomial instead of pm.Poisson
# 2. An additional overdispersion parameter alpha_disp
# alpha_disp controls how much extra variance the NB allows beyond Poisson.
# When alpha_disp → ∞, NB → Poisson. Small alpha_disp = high overdispersion.

with pm.Model(coords=coords) as nb_model:
    alpha_nb = pm.Normal("alpha", mu=log_mu_portfolio, sigma=0.5)
    sigma_district_nb = pm.HalfNormal("sigma_district", sigma=0.3)
    u_district_raw_nb = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
    u_district_nb = pm.Deterministic(
        "u_district", u_district_raw_nb * sigma_district_nb, dims="district"
    )
    log_lambda_nb = alpha_nb + u_district_nb[district_idx_arr]
    mu_nb = pm.math.exp(log_lambda_nb) * exposure_arr

    # Overdispersion parameter: HalfNormal(1.0) is weakly informative.
    # Large alpha_disp ≈ Poisson; small alpha_disp = high overdispersion.
    alpha_disp = pm.HalfNormal("alpha_disp", sigma=1.0)

    claims_obs_nb = pm.NegativeBinomial(
        "claims_obs_nb",
        mu=mu_nb,
        alpha=alpha_disp,
        observed=claims_arr,
    )

print("Fitting Negative Binomial model...")
with nb_model:
    trace_nb = pm.sample(
        draws=1000, tune=1000, chains=4,
        target_accept=0.90, return_inferencedata=True, random_seed=42,
    )

print("Sampling complete.")
```

**Run this cell.** This takes 3-6 minutes.

Now compare:

```python
# LOO (Leave-One-Out) cross-validation using ArviZ.
# LOO estimates how well each model predicts each district's claim count
# when that district is excluded from training.
# Higher ELPD (Expected Log Pointwise Predictive Density) is better.
# log_likelihood must be computed first — ArviZ can do this if not already stored.

try:
    loo_poisson = az.loo(trace, pointwise=True)
    loo_nb = az.loo(trace_nb, pointwise=True)

    print("LOO model comparison (higher ELPD is better):")
    comparison_loo = az.compare({"poisson": trace, "neg_binomial": trace_nb})
    print(comparison_loo[["elpd_loo", "p_loo", "dse", "warning"]].to_string())
    print()

    elpd_diff = float(loo_nb.elpd_loo) - float(loo_poisson.elpd_loo)
    se_diff = float(az.compare({"poisson": trace, "neg_binomial": trace_nb})["dse"].iloc[1])

    print(f"ELPD difference (NB - Poisson): {elpd_diff:.1f}  (SE of difference: {se_diff:.1f})")
    print()
    if abs(elpd_diff) < 2 * se_diff:
        print("Difference is within 2 SEs. Models perform comparably.")
        print("Stick with Poisson: it is simpler and easier to explain.")
    elif elpd_diff > 0:
        print(f"Negative Binomial has higher ELPD by {elpd_diff:.1f} points.")
        if elpd_diff > 10:
            print("Difference is meaningful. Consider switching to Negative Binomial.")
        else:
            print("Difference is modest. Poisson is defensible if it passes PPC checks.")
    else:
        print("Poisson is better. The overdispersion parameter is not warranted.")

    # Pareto k diagnostic
    # LOO is unreliable for observations where Pareto k > 0.7.
    # This happens most often for thin districts with very few claims,
    # where the district's own data has a large influence on the posterior.
    print()
    print("Pareto k diagnostic (LOO reliability):")
    if hasattr(loo_poisson, "pareto_k"):
        k_vals = loo_poisson.pareto_k.values
        n_high_k = (k_vals > 0.7).sum()
        print(f"  Districts with Pareto k > 0.7: {n_high_k} / {len(k_vals)}")
        if n_high_k > len(k_vals) * 0.1:
            print("  More than 10% of districts have unreliable LOO estimates.")
            print("  Use WAIC instead: az.waic() has the same interface as az.loo().")
        elif n_high_k > 0:
            print(f"  {n_high_k} districts have unreliable LOO estimates.")
            print("  These are typically the thinnest districts. LOO comparison is")
            print("  still informative overall but treat with caution for those districts.")
        else:
            print("  All Pareto k values acceptable. LOO comparison is reliable.")
    else:
        print("  Pareto k values not available. Check ArviZ version.")

except Exception as e:
    print(f"LOO computation failed: {e}")
    print("This can happen if log_likelihood was not stored in the trace.")
    print("Try: pm.sample(..., idata_kwargs={'log_likelihood': True})")
```

**What you should see with synthetic Poisson data:** The Poisson model should win or be tied, since the data were generated from a Poisson process. A small ELPD difference within 2 standard errors means the models are statistically indistinguishable — use Poisson for simplicity. In real insurance data, the Negative Binomial typically wins by a modest margin, reflecting unmodelled heterogeneity within segments.

---

## Exercise 4: Presenting credibility-weighted estimates to a pricing committee

### What this teaches

This exercise is not primarily about code — it is about communication. The calculations in Exercises 1-3 are accurate. They are useless if the pricing committee does not understand them well enough to make a decision.

A pricing committee in a UK insurance company typically includes a Chief Underwriting Officer, a Head of Pricing, and a Chief Actuary. They are comfortable with factor tables, Gini coefficients, and residual plots. They are not comfortable with "posterior distributions" or "NUTS sampler". Your job is to present the information they need, in the format they can act on, with the limitations stated clearly enough to survive a sceptical challenge.

### Context

The Head of Pricing wants to introduce district-level geographic rating for the first time, moving away from a current system of six broad area bands. She needs to know:
1. Which districts have rates that are meaningfully different from their current area band rate
2. How confident you are in those differences
3. What would happen to those rates as the district accumulates more experience

### Task 1: Build the factor table

Format the posterior output as a factor table for the pricing committee. Required columns: district, current area band, earned years, observed frequency, credibility-weighted frequency, credibility factor Z (Bühlmann-Straub), uncertainty band (90% posterior interval from the Bayesian model).

```python
# Current area bands — simplified 3-band system based on the first two characters
# of the district name (in a real portfolio, you would use your current rating structure)
def area_to_band(area_prefix: str) -> str:
    high_risk = {"SW", "SE", "N", "E", "W", "EC", "WC"}
    medium_risk = {"BR", "CR", "DA", "EN", "HA", "IG", "KT", "RM", "SM", "TW", "UB", "WD", "SL"}
    if area_prefix in high_risk:
        return "Band 1 (High)"
    elif area_prefix in medium_risk:
        return "Band 2 (Medium)"
    else:
        return "Band 3 (Other)"

# Build the factor table
bs_for_table = bs_results.sort("postcode_district")
bayes_for_table = results.sort("postcode_district")

factor_table = (
    bs_for_table
    .join(
        bayes_for_table.select([
            "postcode_district", "posterior_mean", "lower_90", "upper_90", "earned_years"
        ]),
        on="postcode_district",
        how="inner",
    )
    .with_columns([
        # Derive area prefix for band lookup
        pl.col("postcode_district").str.slice(0, 2).alias("area_prefix"),
    ])
    .with_columns([
        pl.col("area_prefix")
        .map_elements(area_to_band, return_dtype=pl.Utf8)
        .alias("current_band"),
    ])
    .select([
        pl.col("postcode_district").alias("District"),
        pl.col("current_band").alias("Current Band"),
        pl.col("earned_years").round(0).alias("Earned Years"),
        pl.col("obs_mean").round(4).alias("Observed Freq"),
        pl.col("credibility_estimate").round(4).alias("B-S Credibility Freq"),
        pl.col("Z").round(2).alias("Z"),
        pl.col("posterior_mean").round(4).alias("Bayesian Posterior Mean"),
        pl.col("lower_90").round(4).alias("Lower 90%"),
        pl.col("upper_90").round(4).alias("Upper 90%"),
    ])
    .sort("Bayesian Posterior Mean", descending=True)
)

print("Factor table — top 20 districts by credibility-weighted frequency:")
print(factor_table.head(20))
```

**Run this cell.**

### Task 2: Flag material differences

For each district, compare the credibility-weighted frequency to the current area band average. Flag districts where the district rate deviates more than 15% from the area band average — these are the districts where introducing district-level rating would have a material impact on premiums:

```python
# Compute area band average frequencies (weighted by earned years)
band_avg = (
    factor_table
    .group_by("Current Band")
    .agg([
        (
            (pl.col("Bayesian Posterior Mean") * pl.col("Earned Years")).sum()
            / pl.col("Earned Years").sum()
        ).alias("band_avg_freq")
    ])
)

factor_table_flagged = (
    factor_table
    .join(band_avg, on="Current Band", how="left")
    .with_columns([
        (pl.col("Bayesian Posterior Mean") / pl.col("band_avg_freq")).alias("vs_band_ratio"),
    ])
    .with_columns([
        pl.when(pl.col("vs_band_ratio") > 1.15).then(pl.lit("HIGH"))
        .when(pl.col("vs_band_ratio") < (1 / 1.15)).then(pl.lit("LOW"))
        .otherwise(pl.lit(""))
        .alias("material_flag")
    ])
)

n_flagged = factor_table_flagged.filter(pl.col("material_flag") != "").height
n_high = factor_table_flagged.filter(pl.col("material_flag") == "HIGH").height
n_low = factor_table_flagged.filter(pl.col("material_flag") == "LOW").height

print(f"Districts with >15% deviation from area band rate: {n_flagged} / {factor_table.height}")
print(f"  Flagged HIGH (>15% above band): {n_high}")
print(f"  Flagged LOW  (>15% below band): {n_low}")
print()
print("Districts flagged HIGH:")
print(
    factor_table_flagged
    .filter(pl.col("material_flag") == "HIGH")
    .sort("vs_band_ratio", descending=True)
    .select(["District", "Current Band", "Earned Years", "Z",
             "Bayesian Posterior Mean", "band_avg_freq", "vs_band_ratio"])
)
```

**Run this cell.** These are the districts where moving from area-band rating to district-level rating would produce the biggest premium changes. For each flagged district, check the Z value — if Z is low (thin district), the credibility-weighted rate is still close to the portfolio mean, and the deviation from the area band may be more about the area band being poorly calibrated than about the district being genuinely extreme.

### Task 3: Write a three-sentence methodology summary

Write a paragraph suitable for the first slide of the pricing committee pack. The constraints:
- Three sentences maximum
- No mention of MCMC, PyMC, posterior distributions, or NUTS
- Jargon allowed: Bühlmann-Straub, credibility factor, actuarial
- Must explain: what credibility weighting is, what the uncertainty bands represent, why thin cells deserve more pooling than dense cells

**Model answer:**

> Credibility weighting is an established actuarial technique that blends each district's own claims experience with the broader portfolio average, with the blend determined statistically by the amount of data that district has accumulated. Districts with many policy-years of experience follow their own observed claim frequency closely; districts with few policy-years are pulled toward the portfolio mean, preventing thin-cell sampling volatility from driving large and unjustified rate departures. The uncertainty bands show the 90% range of plausible true claim frequencies for each district: for well-populated districts this band is narrow and the rate is well-supported by evidence; for thin districts the band is wider, which is an honest representation of what the data can and cannot tell us.

If you write a different version, review it against these criteria before reading the model answer: Does it explain the blend? Does it explain why thin cells get more shrinkage? Does it explain what the bands mean? Does it use any jargon that a CFAS-qualified actuary would not understand?

### Task 4: Anticipate the three hardest questions

The Chief Actuary is sceptical. She has read the 2024 IFoA working paper on credibility methods and knows the assumptions. Write responses to the three questions she is most likely to ask.

**Question 1: "How is this different from just smoothing with a spline?"**

Write your own response, then check against the model answer below.

*Model answer:*

A spline smoother imposes a shape — smooth, continuous, and often monotone — and fits it to the data. Credibility weighting imposes no shape. It shrinks each district independently toward the portfolio mean, with the amount of shrinkage determined by two quantities: how much districts differ from each other genuinely (the between-district variance, VHM), and how much a single district's experience fluctuates from year to year due to Poisson sampling noise alone (the within-district variance, EPV). A spline requires you to decide that geographic risk varies smoothly across space. Credibility weighting makes no such assumption. UK postcode district risk does not vary smoothly — the boundary between KT4 and SM4 can reflect a sharp demographic break, an A-road barrier, or a change in local authority crime policy. Credibility handles discontinuous geographic risk patterns correctly; splines distort them toward continuity.

---

**Question 2: "The uncertainty bands are very wide for thin districts — does that mean we cannot rate those districts individually?"**

*Model answer:*

The wide bands tell you that the data cannot pin down the true rate for a thin district precisely — that is true. You can still rate those districts individually. The credibility-weighted estimate is the best available estimate given the data: it is not a useless one, but it will sit close to the portfolio mean for thin districts, which is the appropriate pricing decision given the evidence available. What the wide band means is that we should not treat a thin district's observed rate as reliable enough to justify a large departure from the band average, because the observed rate may be driven by one or two unlucky large claims rather than systematic elevated risk. As the district accumulates exposure, the estimate will move toward its own observed experience if that experience is persistently different from the mean — the credibility factor Z approaches 1 as earned years increase. In the meantime, we apply soft caps and floors to district relativities (no district moves more than 40% from its area band rate in a single review cycle), so the wide uncertainty bands on thin cells do not translate to extreme premium swings.

---

**Question 3: "What happens if the true distribution of risks across districts is not log-Normal, as your model assumes?"**

*Model answer:*

The Bayesian hierarchical model assumes district log-rates are Normally distributed, which corresponds to a log-Normal distribution of rates. Log-Normal is reasonable for multiplicative insurance rating factors: decades of GLM work suggest most geographic relativities sit in the range 0.5× to 2.0× the portfolio mean for individual postcode districts, which is consistent with a log-Normal with the standard deviation our model estimates. If the true distribution has heavier tails — for example, a handful of genuinely extreme-risk micro-areas driven by local deprivation or specific road infrastructure — the log-Normal will produce too much shrinkage for the extreme districts, pulling them toward the portfolio mean when the data actually support a more extreme rate. The practical check is the shrinkage plot from Part 10 of the tutorial: if we see dense districts (high Z) being pulled substantially away from their observed rates, the model is over-shrinking. The mitigation is to replace the Normal prior on district log-rates with a Student-t prior with degrees of freedom ν = 4-6: `pm.StudentT("u_district_raw", nu=4, ...)`. This allows heavier tails while maintaining the partial pooling structure. For our current data, the posterior predictive checks in Exercise 3 show adequate calibration — the log-Normal appears sufficient.

---

## What you have built across this module

By completing the tutorial and all four exercises, you have:

1. **Implemented Bühlmann-Straub credibility from first principles** in NumPy and Polars — no black-box library. You understand what v, a, and K mean and why they are estimated from data rather than specified in advance.

2. **Fitted a Bayesian hierarchical Poisson model** in PyMC 5 with non-centered parameterisation, run mandatory convergence diagnostics, and extracted posterior means and credible intervals.

3. **Applied credibility to severity data** and computed credibility-weighted pure premiums — the combination of frequency and severity credibility adjustments that feeds into pricing.

4. **Built a two-level geographic hierarchy** that allows districts to borrow strength from neighbouring districts in the same postcode area, rather than from the full portfolio.

5. **Validated the model** with posterior predictive checks — confirming that convergence does not imply correctness, and that calibration checks are mandatory.

6. **Formatted outputs for a pricing committee** — factor tables, uncertainty bands, material deviation flags, and responses to the hardest regulatory questions.

The next module covers pricing for new schemes and thin portfolios without district-level geographic data — where the thin-cell problem operates at the scheme level rather than the geographic level, and where the credibility methods from this module combine with prior elicitation from underwriting expertise.
