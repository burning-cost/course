## Part 9: MCMC diagnostics -- how to know it worked

Fitting a Bayesian model via MCMC is not the same as fitting a GLM. A GLM optimiser either converges to a solution or fails. MCMC always produces samples -- the question is whether those samples are reliable. You must check diagnostics before trusting any output.

### R-hat (Gelman-Rubin statistic)

R-hat compares variance within chains to variance between chains. If all chains have converged to the same distribution, R-hat is close to 1.0. We use the threshold from Vehtari et al. (2021): R-hat < 1.01 for all parameters.

An R-hat of 1.05 for the `rho` parameter means the chains disagree about where rho lives. The posterior estimates are unreliable. Do not use the relativities until convergence is achieved.

### Effective sample size (ESS)

MCMC samples are correlated within a chain. The effective sample size is the number of *independent* samples the chain is equivalent to. With 4,000 total samples (4 chains x 1,000 draws), if ESS_bulk for `rho` is 850, that 4,000 samples are equivalent to 850 independent draws. We want ESS > 400 for all parameters.

### Running the diagnostics

```python
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
```

**Interpreting rho_summary.** The `mean` column gives the posterior mean of rho -- the average proportion of territory variance that is spatially structured. A mean of 0.72 with a 95% credibility interval [0.45, 0.93] means: we estimate that roughly 72% of the geographic variation in this dataset is spatially correlated, but there is substantial uncertainty. This is informative: it says spatial smoothing is doing real work, but the IID component is not negligible.

**Interpreting sigma_summary.** Sigma is on the log scale. A posterior mean of 0.28 means territory effects have a standard deviation of about 0.28 on the log scale, corresponding to a multiplicative spread of roughly exp(0.28) = 1.32 above the mean.

### Trace plots

A trace plot shows the sampler trajectory for each chain over time. Healthy chains look like "hairy caterpillars" -- fast mixing, all chains overlapping, no trend. A problematic chain looks like a slug: slow, drifting, stuck in one region.

```python
from insurance_spatial.plots import plot_trace

fig = plot_trace(result, params=["alpha", "sigma", "rho"])
plt.tight_layout()
plt.show()
```

If any chain shows a trend (systematically drifting upwards or downwards over the draw index), the model has not converged. You need more tuning steps: increase `tune=2000` and refit.

### Divergences: what they mean and how to fix them

Divergent transitions indicate that the NUTS sampler hit a region of the posterior where the geometry is very curved and it could not step accurately. Divergences are not just a computational nuisance -- they indicate that the posterior samples in those regions may be wrong. Even a handful of divergences can bias estimates.

If you see divergences:

1. **Increase `target_accept` to 0.95.** This makes the sampler take smaller steps, which handles more curved geometry at the cost of slower sampling.
2. **Check the model specification.** The BYM2 model is occasionally sensitive to the prior on sigma when data are very sparse. Try `sigma ~ HalfNormal(0.5)` for sparser data.
3. **If divergences persist above 10**, consider whether the data have enough spatial signal to identify both rho and sigma simultaneously. For very sparse data, rho and sigma can be weakly identified -- the sampler cannot distinguish "high rho, low sigma" from "low rho, high sigma". In this case, fix rho at 0.5 (a reasonable default for most UK personal lines books) and estimate only sigma.