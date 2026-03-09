## Part 9: Extracting posterior estimates

Once convergence is confirmed, extract the posterior means and credible intervals:

```python
%md
## Part 9: Extracting posterior estimates
```

```python
# Extract posterior samples for alpha and u_district
# trace.posterior is an xarray Dataset.
# trace.posterior["alpha"] has shape (chains, draws) = (4, 1000)
# trace.posterior["u_district"] has shape (chains, draws, districts) = (4, 1000, 120)

# Posterior means: average over chains and draws
alpha_post_mean = float(trace.posterior["alpha"].mean())
u_post_mean = trace.posterior["u_district"].mean(dim=("chain", "draw")).values  # shape: (120,)

# Posterior mean log-rate per district
log_rate_post_mean = alpha_post_mean + u_post_mean

# Convert to rate space (anti-log)
posterior_mean_rate = np.exp(log_rate_post_mean)

print(f"alpha posterior mean: {alpha_post_mean:.4f}")
print(f"Portfolio rate from alpha: {np.exp(alpha_post_mean):.4f}  ({np.exp(alpha_post_mean)*100:.2f}%)")
print()
print(f"sigma_district posterior mean: {float(trace.posterior['sigma_district'].mean()):.4f}")
print(f"  (True sigma_district was:   {TRUE_SIGMA_DISTRICT:.4f})")
print()
```

**What this does:** Extracts the posterior means for the global parameters. The `mean(dim=("chain", "draw"))` call averages across all 4 chains × 1000 draws = 4,000 samples to give a single posterior mean per district.

**Run this cell.**

**What you should see:** The posterior mean for sigma_district should be close to 0.35 (the true value we used to simulate the data). If it is far off, investigate the convergence diagnostics. Alpha posterior mean should be close to `log(0.07) ≈ -2.66`.

Now extract per-district posterior intervals:

```python
# Credible intervals: 90% posterior interval for each district's rate
# We need to sample the full posterior for each district, not just the mean.

# Stack all posterior samples into a 2D array: (total_samples, districts)
# Shape: (4 chains × 1000 draws, 120 districts) = (4000, 120)
alpha_samples = trace.posterior["alpha"].values.flatten()          # shape: (4000,)
u_samples = trace.posterior["u_district"].values.reshape(-1, n_districts_model)  # (4000, 120)

# Add alpha to each district's u_district samples
log_rate_samples = alpha_samples[:, np.newaxis] + u_samples         # (4000, 120)
rate_samples = np.exp(log_rate_samples)                              # (4000, 120)

# 5th and 95th percentile → 90% credible interval
lower_90 = np.percentile(rate_samples, 5, axis=0)   # shape: (120,)
upper_90 = np.percentile(rate_samples, 95, axis=0)  # shape: (120,)
posterior_sd = rate_samples.std(axis=0)              # shape: (120,)

observed_rate = claims_arr / exposure_arr

# Build results DataFrame
results = pl.DataFrame({
    "postcode_district": districts_sorted,
    "total_claims":      claims_arr.tolist(),
    "earned_years":      exposure_arr.tolist(),
    "observed_rate":     observed_rate.tolist(),
    "posterior_mean":    posterior_mean_rate.tolist(),
    "posterior_sd":      posterior_sd.tolist(),
    "lower_90":          lower_90.tolist(),
    "upper_90":          upper_90.tolist(),
    "interval_width":    (upper_90 - lower_90).tolist(),
})

print("Posterior estimates — first 15 districts:")
print(results.head(15))
```

**What this does:** Extracts the full posterior distribution for each district's claim rate, computes the 90% credible interval, and assembles a clean results DataFrame.

**Run this cell.**

**What you should see:** A 120-row DataFrame with each district's observed rate, posterior mean, posterior standard deviation, and 90% credible interval. Notice that thin districts have wide intervals (large `interval_width`) and dense districts have narrow intervals. This is honest: the model is correctly representing its own uncertainty.

### Approximate credibility factor from the Bayesian model

```python
# Compute an approximate Z from the Bayesian posterior.
# The Bayesian model does not directly output a Z value — Z is a B-S concept.
# But we can estimate it from how much the posterior mean was pulled toward
# the grand mean relative to the observed rate.

grand_mean_rate = float(np.exp(alpha_post_mean))

z_bayes_approx = np.where(
    np.abs(observed_rate - grand_mean_rate) > 1e-6,
    1.0 - np.abs(posterior_mean_rate - grand_mean_rate)
        / np.abs(observed_rate - grand_mean_rate),
    1.0,
).clip(0, 1)

results = results.with_columns(
    pl.Series("z_bayes_approx", z_bayes_approx)
)

print(f"Grand mean rate (exp(alpha)): {grand_mean_rate:.4f}")
print()
print("Approximate Bayesian Z vs Bühlmann-Straub Z:")
comparison = bs_results.join(
    results.select(["postcode_district", "posterior_mean", "z_bayes_approx"]),
    on="postcode_district",
    how="inner",
)
print(comparison.select([
    "postcode_district", "exposure", "Z", "z_bayes_approx",
    "credibility_estimate", "posterior_mean",
]).sort("exposure").head(20))
```

**Run this cell.**

**What you should see:** The B-S Z and Bayesian approximate Z should be broadly similar. They will agree closely for dense districts. For thin districts, the Bayesian model typically produces slightly more shrinkage (lower Z) because it propagates uncertainty in sigma_district — the B-S estimate plugs in a_hat as a fixed value, ignoring the uncertainty in the between-district variance estimate.