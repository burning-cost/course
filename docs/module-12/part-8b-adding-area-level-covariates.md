## Part 8b: Adding area-level covariates

The basic BYM2 model captures all geographic variation in the spatial random effect. But some geographic variation is not spatial autocorrelation -- it is explained by observable area-level covariates. If you include a covariate like Index of Multiple Deprivation (IMD) score in the model, the spatial random effect b only needs to capture the variation that remains after accounting for deprivation. This produces cleaner, more stable territory factors and makes the model easier to explain: "Here is the deprivation effect. Here is the additional spatial variation on top of deprivation."

### What counts as a useful area-level covariate

For UK personal lines:

- **Motor frequency:** IMD score (vehicle crime correlates with deprivation), road network density (urban areas have more minor collisions), vehicle crime rate per LSOA from data.police.uk
- **Home:** flood risk index (Environment Agency), subsidence risk, IMD
- **Pet:** none that are well-established; territory is largely capturing a residual

The covariate must be available at the same geographic granularity as your territory model (sector or district). It should be centred and scaled before passing to the model -- the prior on beta is Normal(0,1), which is appropriate for standardised covariates but too tight for variables on arbitrary scales.

### Adding IMD as a covariate

We simulate an IMD score for each synthetic area. True IMD data is available free from the Ministry of Housing, Communities and Local Government for England at LSOA level; aggregate to sector by exposure-weighted average.

```python
# Simulate IMD score: correlated with north-south gradient (urban areas are
# typically more deprived in this simplified model)
rng_cov = np.random.default_rng(seed=88)
imd_raw = 30 + 20 * (row_idx / (NROWS - 1)) + rng_cov.normal(0, 5, N)
# Standardise: BYM2 prior on beta is Normal(0,1), so covariates must be scaled
imd_scaled = (imd_raw - imd_raw.mean()) / imd_raw.std()

print(f"IMD raw range:    [{imd_raw.min():.1f}, {imd_raw.max():.1f}]")
print(f"IMD scaled mean:  {imd_scaled.mean():.4f}  (should be ~0)")
print(f"IMD scaled SD:    {imd_scaled.std():.4f}   (should be ~1)")
print(f"IMD-risk corr:    {np.corrcoef(imd_scaled, true_log_effect)[0,1]:.4f}")
```

### Fitting with covariates

Pass the scaled covariate as a 2D array of shape (N, P) where P is the number of covariates. Here P=1.

```python
model_cov = BYM2Model(
    adjacency=adj,
    draws=1000,
    chains=4,
    target_accept=0.9,
    tune=1000,
)

result_cov = model_cov.fit(
    claims=claims,
    exposure=exposure.astype(float),
    covariates=imd_scaled[:, np.newaxis],  # shape (N, 1)
    random_seed=42,
)

print("Covariate model fitted.")
```

### Interpreting the covariate coefficient

After fitting, the `beta` parameter in the trace is the log-scale coefficient for IMD. A posterior mean of 0.15 means that a one-standard-deviation increase in IMD score is associated with exp(0.15) = 1.16, or 16%, higher expected claim frequency, holding the spatial random effect constant.

```python
import arviz as az

# Extract beta posterior
beta_samples = result_cov.trace.posterior["beta"].values.ravel()
print(f"Beta (IMD effect):")
print(f"  Posterior mean:  {beta_samples.mean():.4f}")
print(f"  Posterior SD:    {beta_samples.std():.4f}")
print(f"  95% CI:          [{np.quantile(beta_samples, 0.025):.4f}, {np.quantile(beta_samples, 0.975):.4f}]")
print(f"  Multiplicative:  {np.exp(beta_samples.mean()):.4f} per SD increase in IMD")
```

If the 95% credibility interval on beta excludes zero, the covariate is adding genuine explanatory power. If it spans zero, the data do not support a covariate effect and you should drop it from the model.

### Does adding the covariate change the territory factors?

`rels` is the relativity DataFrame from Part 11 (the model without covariates). If you have not run Part 11 yet, compute it now: `rels = result.territory_relativities()`.

```python
rels_cov = result_cov.territory_relativities()

# Compare the range of b_mean (log-scale spatial effects)
# rels was computed in Part 11 from the no-covariate model
print(f"Without covariate: b range [{rels['b_mean'].min():.4f}, "
      f"{rels['b_mean'].max():.4f}]")
print(f"With IMD covariate: b range [{rels_cov['b_mean'].min():.4f}, "
      f"{rels_cov['b_mean'].max():.4f}]")
```

Adding a covariate that is correlated with the spatial pattern typically *reduces* the range of the spatial random effect b. The territory factors become smaller in magnitude because part of what looked like spatial variation was actually explained by deprivation. This is exactly what you want: the territory factor now represents genuine spatial variation, not a proxy for deprivation.

This matters for regulatory work. If your territory factor is partly a proxy for deprivation, and deprivation is correlated with a protected characteristic (e.g., ethnicity), the territory factor may be indirectly discriminatory. Removing the deprivation effect from territory before filing the factors is sound practice.
