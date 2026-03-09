## Part 8: Convergence diagnostics — do not skip this

MCMC sampling can fail silently. A model can complete sampling, print no errors, and produce posteriors that are systematically wrong. Convergence diagnostics are not bureaucratic box-ticking — they are how you detect this.

### What convergence means

MCMC generates a Markov chain — a sequence of samples where each sample depends only on the previous one. If the chain has been running long enough and mixing well across the posterior, the distribution of samples approximates the true posterior. "Convergence" means this approximation is good.

We run 4 independent chains starting from different initial points. If the chains have converged, their distributions should look the same. If they look different, something is wrong.

### R-hat: between-chain versus within-chain variance

R-hat (also written R̂) compares the variance between chains to the variance within chains. A value of 1.0 means the chains are identical — perfect convergence. Values above 1.01 indicate the chains have not converged: some region of the posterior is not being explored consistently.

### ESS: effective sample size

Because MCMC samples are autocorrelated (each sample is correlated with the previous one), 4,000 samples do not give you 4,000 independent pieces of information. ESS adjusts for this autocorrelation. Low ESS (below 400) means the posterior estimate for that parameter is unreliable.

For variance components (sigma_district), the ESS requirement is higher — we recommend 1,000+ — because sigma_district drives the credibility factors for all districts. A poorly sampled sigma_district propagates errors into all district estimates.

### Divergences

A divergence occurs when NUTS takes an extremely large step and the trajectory diverges numerically. Divergences indicate regions of the posterior where the sampler cannot explore correctly. Non-zero divergences require investigation, even if convergence diagnostics look otherwise acceptable.

### Running the diagnostics

```python
%md
## Part 8: Convergence diagnostics
```

```python
# R-hat: should be < 1.01 for all parameters
rhat = az.rhat(trace)

# az.rhat() returns an xarray Dataset. Convert to a single maximum value.
max_rhat = float(rhat.max().to_array().max())
print(f"Max R-hat across all parameters: {max_rhat:.4f}")

if max_rhat < 1.01:
    print("  Status: OK — chains have converged.")
elif max_rhat < 1.05:
    print("  Status: WARNING — some parameters have not fully converged.")
    print("  Consider increasing draws to 2000 or tune to 2000.")
else:
    print("  Status: FAILED — chains have not converged. Do not use these results.")
    print("  Check for model misspecification or try non-centered parameterisation.")

print()

# ESS (bulk): effective sample size for interior of distribution
ess_bulk = az.ess(trace, method="bulk")
min_ess_bulk = float(ess_bulk.min().to_array().min())
print(f"Min ESS (bulk) across all parameters: {min_ess_bulk:.0f}")

if min_ess_bulk > 400:
    print("  Status: OK")
else:
    print("  Status: LOW — increase draws or check for slow mixing.")

print()

# Variance component ESS — especially important
sigma_ess = float(ess_bulk["sigma_district"])
print(f"ESS for sigma_district (the key variance component): {sigma_ess:.0f}")
if sigma_ess > 1000:
    print("  Status: OK — variance component is well-sampled.")
elif sigma_ess > 400:
    print("  Status: ACCEPTABLE — consider 2000 draws for final results.")
else:
    print("  Status: LOW — increase draws to 2000+ for variance component reliability.")

print()

# Divergences: should be 0
n_div = int(trace.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}")
if n_div == 0:
    print("  Status: OK — no divergences.")
elif n_div < 10:
    print("  Status: FEW — investigate but may be acceptable.")
    print("  Try increasing target_accept to 0.95.")
else:
    print("  Status: MANY — model has posterior geometry problems.")
    print("  Verify non-centered parameterisation is correctly implemented.")
    print("  If divergences persist, the model may be misspecified.")
```

**What this does:** Runs the three standard MCMC convergence checks. The ESS check for `sigma_district` specifically is important — the between-district variance drives all the credibility factors, and it needs to be well-sampled.

**Run this cell.**

**What you should see with the synthetic data and the model as written:** Max R-hat below 1.01, min ESS above 400, ESS for sigma_district above 1000, zero divergences. If any check fails, the interpretations above tell you what to try next.

### Trace plots: visual convergence check

```python
# Trace plot for the key parameters
# Shows the chain values over iterations (should look like white noise)
# and the marginal distribution (should be smooth bell-shaped)

az.plot_trace(
    trace,
    var_names=["alpha", "sigma_district"],  # plot the global parameters
    figsize=(10, 5),
)
plt.suptitle("Trace plots: alpha and sigma_district", y=1.02)
plt.tight_layout()
display(plt.gcf())
plt.close()
```

**What this does:** Shows the trace of the MCMC chains over iterations. Good mixing looks like a "hairy caterpillar" — the chain moves freely across its range, all four chains overlap, and there are no obvious trends or stuck periods.

**Run this cell.**

**What you should see:** For alpha: four chains (different colours) that overlap heavily, all fluctuating around the same mean. For sigma_district: same pattern, with the four chains' marginal distributions (right panel) all showing the same shape. If any chain looks like it is stuck in one region or trending, convergence has not been achieved.

### Checkpoint 3: Convergence check

```python
# Hard gate: if convergence criteria are not met, do not proceed with results.
# The results downstream are meaningless if the sampler has not converged.
print("=== CHECKPOINT 3: CONVERGENCE ===")
print()
convergence_ok = (max_rhat < 1.01) and (min_ess_bulk > 400) and (n_div == 0)
print(f"R-hat OK (< 1.01):      {'YES' if max_rhat < 1.01 else 'NO'}")
print(f"Min ESS OK (> 400):     {'YES' if min_ess_bulk > 400 else 'NO'}")
print(f"Divergences OK (= 0):   {'YES' if n_div == 0 else 'NO'}")
print()
if convergence_ok:
    print("All convergence criteria met. Proceeding to extract results.")
else:
    print("CONVERGENCE FAILURE: Do not interpret the following results.")
    print("Investigate the failing criteria before proceeding.")
```

If any convergence check fails here, stop. Fix the issue before proceeding. Results from a model that has not converged are not just uncertain — they are wrong in ways that are hard to detect downstream.