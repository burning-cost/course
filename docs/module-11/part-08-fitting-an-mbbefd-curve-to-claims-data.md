## Part 8: Fitting an MBBEFD curve to claims data

### Loading the synthetic claims data

We use the `insurance-datasets` library for a synthetic UK commercial property dataset. This dataset is designed to be realistic but has no real policyholder data: it is generated from the MBBEFD family with added noise and realistic censoring.

```python
%md
## Part 8: Fitting MBBEFD to claims data
```

```python
# Load the synthetic commercial property claims
# The dataset includes loss amounts, sums insured (= MPL for this dataset),
# and policy details.

try:
    from insurance_datasets import load_commercial_property_claims
    claims_raw = load_commercial_property_claims()
    print(f"Loaded {len(claims_raw):,} claims from insurance-datasets")
    use_synthetic_fallback = False
except (ImportError, Exception):
    # Fallback: generate synthetic MBBEFD data directly
    print("insurance-datasets not available; generating synthetic data")
    use_synthetic_fallback = True

if use_synthetic_fallback:
    # Generate claims from a Y3-like distribution with noise
    rng = np.random.default_rng(42)
    N_CLAIMS = 600
    true_dist = MBBEFDDistribution.from_c(2.8)   # close to Y3

    destruction_rates_raw = true_dist.rvs(N_CLAIMS, rng=rng)
    mpl_values = rng.choice(
        [500_000, 1_000_000, 2_000_000, 5_000_000],
        N_CLAIMS,
        p=[0.35, 0.35, 0.20, 0.10],
    )
    loss_amounts = destruction_rates_raw * mpl_values

    claims_df = pl.DataFrame({
        "loss_amount": loss_amounts,
        "mpl":         mpl_values.astype(float),
    })
    print(f"Generated {N_CLAIMS:,} synthetic claims")
    print(f"True distribution: c ≈ 2.8  (between Y2 and Y3)")
else:
    # Convert to Polars if needed
    if hasattr(claims_raw, "to_pandas"):
        claims_df = claims_raw  # already Polars
    else:
        claims_df = pl.from_pandas(claims_raw)
    print(claims_df.schema)

print(f"\nClaims shape: {claims_df.shape}")
```

### Computing destruction rates and exploring the data

```python
# Compute destruction rates (z = loss / MPL)
claims_df = claims_df.with_columns(
    (pl.col("loss_amount") / pl.col("mpl")).clip(0.0, 1.0).alias("z")
)

# Summary statistics on z
z_vals = claims_df["z"].to_numpy()
total_losses = (z_vals >= 1.0 - 1e-9).sum()

print(f"Destruction rate summary:")
print(f"  n claims:           {len(z_vals):,}")
print(f"  total losses (z=1): {total_losses:,}  ({total_losses/len(z_vals):.1%})")
print(f"  mean z:             {z_vals.mean():.4f}")
print(f"  median z:           {np.median(z_vals):.4f}")
print(f"  90th percentile z:  {np.percentile(z_vals, 90):.4f}")
print(f"  max z:              {z_vals.max():.4f}")
```

Before fitting, it is worth thinking about what you expect. If the true curve is between Y2 and Y3, you would expect:
- Total loss frequency: between 4% and 13%
- Mean z: roughly 0.15 to 0.25
- The bulk of claims below z = 0.40

### Fitting with MLE

```python
from insurance_ilf import fit_mbbefd

# Fit MBBEFD using MLE
result = fit_mbbefd(z_vals)

print(result)
print()
print(result.summary())
```

**What you should see (with the synthetic fallback):**

```
FittingResult(g=22.4321, b=3.1847, loglik=-312.4, aic=628.8, converged=True)

      g      b  total_loss_prob    mean   loglik      aic       bic  n_obs method  converged
 22.432  3.185           0.0446  0.1891  -312.39  628.78  638.17    600    mle       True
```

The fitted g implies a total loss probability of about 4.5%, and b ≈ 3.2 places this between Y2 (b ≈ 9.0) and Y3 (b ≈ 2.7). We know the true distribution was c ≈ 2.8 (between Y2 and Y3), so the fit is recovering something close to the truth. You would not expect an exact match from 600 observations.

```python
# Access the fitted distribution
fitted_dist = result.dist
print(f"Fitted: g = {fitted_dist.g:.4f}, b = {fitted_dist.b:.4f}")
print(f"Total loss prob: {fitted_dist.total_loss_prob():.2%}")

# Where on the c-parameter scale does this sit?
c_approx = fitted_dist.to_c()
if c_approx is not None:
    print(f"Nearest Swiss Re c: {c_approx:.2f}")
else:
    print("Fitted parameters do not correspond to a standard Swiss Re curve")
```

### Understanding the multi-start optimisation

The log-likelihood surface for MBBEFD is non-convex. There are multiple local optima, and a single-start optimiser will often get stuck. The library uses multi-start optimisation: it tries 6 starting points (the five Swiss Re c-parameter values plus a moment-matching start) and runs L-BFGS-B from each. The best result across all starts is returned.

```python
# Show what the five c-parameter starts give individually
from insurance_ilf.fitting import _starting_params_for_c, _neg_loglik
import warnings

print("Log-likelihoods from each Swiss Re c starting point:")
print(f"{'c':>5} {'start loglik':>15} {'converged NLL':>15}")
print("-" * 38)

from scipy.optimize import minimize

for c in [1.5, 2.0, 3.0, 4.0, 5.0]:
    start = _starting_params_for_c(c)
    res = minimize(
        _neg_loglik,
        x0=start,
        args=(z_vals, None, None),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    print(f"  {c:>3.1f}  start_nll={_neg_loglik(start, z_vals, None, None):>12.2f}  "
          f"converged_nll={res.fun:>12.2f}")
```

You will see that different starting points converge to different local optima, with different negative log-likelihoods. The library picks the minimum. This is why `n_starts` matters: for unusual datasets, six starts may not be enough, and you can set `n_starts=12` to try more random perturbations.