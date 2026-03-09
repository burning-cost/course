## Part 9: Truncated and censored data

### The problem

The claims in your database are not the complete severity distribution. They are observations subject to at least two distortions:

1. **Truncation (deductible)**: if the policy has a deductible, only claims above the deductible enter the claims system. Claims below the deductible are never reported. In destruction-rate terms, if the deductible is 3% of MPL, then all z < 0.03 are unobserved.

2. **Censoring (policy limit)**: if the claim exceeds the policy limit, the insurer pays the policy limit and the excess is unrecovered (or goes to a reinsurer). In destruction-rate terms, if the policy limit is 70% of MPL, then all z > 0.70 are recorded as z = 0.70.

If you ignore these distortions when fitting, the fitted curve will be wrong. Specifically:
- Ignoring truncation overstates the mean destruction rate (you fitted only to the larger losses)
- Ignoring censoring understates the tail of the distribution (all the mass above the policy limit is compressed into the point at the limit)

The `fit_mbbefd` function handles both via the `truncation` and `censoring` arguments. These are scalars (same for all observations) expressed as fractions of MPL. For per-observation truncation or censoring at different levels, you would need to model each observation separately -- the library does not currently support this, and you would need to construct the likelihood manually (see Exercise 8).

### Fitting with truncation and censoring

```python
%md
## Part 9: Truncated and censored data
```

```python
# Suppose our data has:
# - A deductible of 5% of MPL (claims below 5% of MPL not reported)
# - A policy limit of 75% of MPL (claims above 75% are capped)

# First, create a version of the data that reflects these constraints
z_truncated_censored = z_vals[(z_vals >= 0.05)]  # remove claims below deductible
z_truncated_censored = np.where(
    z_truncated_censored > 0.75,
    0.75,   # cap at policy limit
    z_truncated_censored
)

print(f"Original n:    {len(z_vals):,}")
print(f"After truncation (z >= 0.05):  {len(z_truncated_censored):,} "
      f"({len(z_truncated_censored)/len(z_vals):.0%} retained)")
print(f"Capped at z=0.75: {(z_truncated_censored == 0.75).sum()} observations")

# Naive fit: ignore truncation and censoring
result_naive = fit_mbbefd(z_truncated_censored)

# Correct fit: tell the fitter about the data-generating constraints
result_correct = fit_mbbefd(
    z_truncated_censored,
    truncation=0.05,   # deductible = 5% of MPL
    censoring=0.75,    # policy limit = 75% of MPL
)

print()
print("Naive fit (ignoring truncation/censoring):")
print(f"  g = {result_naive.params['g']:.4f},  b = {result_naive.params['b']:.4f}")
print(f"  Total loss prob: {result_naive.dist.total_loss_prob():.2%}")
print(f"  Mean z:          {result_naive.dist.mean():.4f}")

print()
print("Correct fit (with truncation=0.05, censoring=0.75):")
print(f"  g = {result_correct.params['g']:.4f},  b = {result_correct.params['b']:.4f}")
print(f"  Total loss prob: {result_correct.dist.total_loss_prob():.2%}")
print(f"  Mean z:          {result_correct.dist.mean():.4f}")
```

The naive fit will typically produce a lower mean z (because small losses are missing and the model does not know they exist) and will misestimate the total loss probability. The correct fit adjusts for both distortions by modifying the likelihood following Bernegger (1997), extended to the truncated-censored case following standard survival likelihood arguments.

The truncated-censored likelihood adjusts each observation's log-density:

- For uncensored observations: log f(z_i; g, b) -- log[1 - F(T; g, b)]
- For censored observations (z_i = M): log[1 - F(M; g, b)] -- log[1 - F(T; g, b)]
- For total losses (z_i = 1): log[1/g] -- log[1 - F(T; g, b)]

where T is the truncation point and M is the censoring point. The subtraction of log[1 - F(T)] in each term is the correction for truncation: we condition on the observation being above the deductible.

In practice: always pass `truncation` and `censoring` when they apply. The difference in fitted parameters matters materially for XL pricing.