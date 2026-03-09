## Part 10: Moran's I post-fit -- confirming the model absorbed the spatial structure

After fitting, we run Moran's I again on the posterior predictive residuals. If the model has correctly captured the spatial structure, the post-fit residuals should show no significant autocorrelation.

```python
# Posterior predictive mean of claims per area
# result.trace.posterior["mu"] has shape (chains, draws, N)
mu_samples = result.trace.posterior["mu"].values
mu_mean = mu_samples.mean(axis=(0, 1))  # shape (N,)

# Post-fit residuals: observed vs. posterior predictive mean
postfit_log_oe = np.log((claims + 0.5) / (mu_mean + 0.5))

test_post = moran_i(postfit_log_oe, adj, n_permutations=999)

print("=== Pre-fit Moran's I ===")
print(f"  I = {test.statistic:.4f}, p = {test.p_value:.4f}")
print()
print("=== Post-fit Moran's I ===")
print(f"  I = {test_post.statistic:.4f}, p = {test_post.p_value:.4f}")
print()
print(test_post.interpretation)
```

**What you want to see:** The pre-fit I is significantly positive. The post-fit I is non-significant (p > 0.05). This means the model has absorbed the spatial structure -- what remains in the residuals is effectively white noise.

If the post-fit Moran's I is *still* significant, the model has not fully captured the spatial pattern. Possible causes:

- The model needs more MCMC draws to properly characterise the posterior
- The spatial pattern is more complex than a single-scale ICAR can capture (e.g., multi-scale structure with both local and regional gradients)
- There is a missing covariate with a strong spatial pattern (e.g., flood risk, deprivation) that should be included as a fixed effect