## Part 1: What SHAP values are and why they work for relativities

Before writing any code, we need to understand what SHAP values actually are. This is not optional background - if you do not understand the maths, you cannot know when the output is wrong.

### The decomposition

The CatBoost Poisson frequency model produces predictions in log space. For a policy `i`, the model computes:

```sql
log(mu_i) = log(exposure_i) + phi(x_i)
```

where `phi(x_i)` is the sum of all tree outputs for that observation. SHAP (specifically TreeSHAP) decomposes `phi(x_i)` into a sum of per-feature contributions plus a constant:

```sql
phi(x_i) = expected_value + SHAP_1(x_i) + SHAP_2(x_i) + ... + SHAP_p(x_i)
```

The `expected_value` is the average prediction across the training set. Each `SHAP_j(x_i)` is the contribution of feature `j` to the difference between this observation's prediction and the average.

This decomposition satisfies the Shapley efficiency axiom: the contributions sum exactly to the difference between the prediction and the average. Every unit of the log-prediction is accounted for.

### Why this gives multiplicative relativities

Because the decomposition is additive in log space, the prediction in the original count scale factors as:

```sql
mu_i = exp(expected_value) × exp(SHAP_1(x_i)) × exp(SHAP_2(x_i)) × ... × exp(SHAP_p(x_i))
```

That is a multiplicative model. Each term `exp(SHAP_j(x_i))` is the factor contribution of feature `j` for this specific observation.

For a categorical feature like `area`, every observation in area B has a SHAP value for that feature. Those values vary slightly from observation to observation because tree splits interact - a GBM learns context-dependent effects, not pure main effects. But they cluster around a centre that represents the average log-contribution of being in area B.

The relativity for area B relative to area A is:

```sql
relativity(B vs A) = exp(mean_SHAP(area=B) - mean_SHAP(area=A))
```

where the mean is exposure-weighted across all observations at each level. This is directly analogous to `exp(beta_B - beta_A)` from a GLM.

### Confidence intervals

The central limit theorem gives us standard errors on the mean SHAP values within each level:

```bash
SE(level k) = shap_std(k) / sqrt(n_obs(k))
```

The 95% confidence interval on the relativity for level `k` relative to base level `0` is:

```bash
CI = exp( (mean_SHAP(k) - mean_SHAP(0)) ± 1.96 × sqrt(SE(k)^2 + SE(0)^2) )
```

The full formula accounts for uncertainty at both the level of interest and the base level. For a large, well-populated base level like area A, `SE(0)` is tiny and the formula simplifies. For a sparse base level, it matters.

These intervals capture **data uncertainty** - how precisely we have estimated the mean SHAP contribution given the observations we have. They do not capture model uncertainty: the fact that a different training split would produce a different GBM with different SHAP values. Be explicit about this distinction when presenting to regulators.

**Approximation note:** The CLT-based standard error formula assumes that the SHAP values within each level are approximately normally distributed around their mean, and that the SHAP estimates for the target level and the base level are independent. In practice, these assumptions are reasonable but not exact: SHAP values within a level can be skewed (especially for small, high-risk levels), and the base level and target level SHAP values share the same model, so they are not truly independent. The intervals are best treated as indicative rather than exact 95% bounds.