# Module 11 Exercises: Exposure Curves and Increased Limits Factors

Ten exercises, progressive difficulty. Work through them in order: each one builds on the data and results from the previous. The solutions are at the end of each exercise, collapsed. Try each exercise before looking at the solution.

Before starting: read all 17 parts of the tutorial. All concepts required appear there.

---

## Exercise 1: Understanding the exposure curve formula

**Reference:** Tutorial Parts 4 and 5

**What you will do:** Verify the MBBEFD exposure curve formula manually, then explore how changes to g and b shift the curve.

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from insurance_ilf import MBBEFDDistribution, swiss_re_curve
from insurance_ilf.curves import all_swiss_re_curves

# We will use Y2 as our reference throughout this exercise
y2 = swiss_re_curve(2.0)
```

### Tasks

**Task 1.** Implement the MBBEFD exposure curve formula from scratch using numpy, without calling `dist.exposure_curve()`. Use the formula from the tutorial:

```
G(x) = ln[A + C * b^x] / ln(g*b)
where A = (g-1)*b / (1-b),  C = (1-g*b) / (1-b)
```

Verify your implementation against `y2.exposure_curve()` at x values of 0.0, 0.1, 0.25, 0.5, 0.75, and 1.0. The maximum absolute error should be below 1e-10 for x in (0, 1).

Note: your formula will not automatically enforce G(0) = 0 and G(1) = 1 -- you will need to handle the boundary cases explicitly.

**Task 2.** Plot G(x) for MBBEFDDistribution objects with the following (g, b) pairs on a single chart:

- (3.0, 5.0)
- (7.69, 9.02) -- this is approximately Y2
- (20.0, 3.0)
- (100.0, 0.9)
- (500.0, 0.3)

Label each curve with its g and b values. Add the diagonal reference line. Describe in 2-3 sentences what the g and b parameters control visually.

**Task 3.** Compute `total_loss_prob()` and `mean()` for each of the five (g, b) pairs from Task 2. Arrange them in a table. What is the relationship between g and total loss probability? What is the relationship between b and the mean destruction rate?

**Task 4.** The Y2 distribution has mean destruction rate 0.214. Verify this by numerical integration:

```python
from scipy import integrate
# E[Z] = integral of x * f(x) dx + 1.0 * P(X=1)
```

Use `y2.pdf()` for the density and `y2.total_loss_prob()` for the point mass. If your numerical result differs from `y2.mean()` by more than 0.001, explain why.

<details>
<summary>Solution -- Exercise 1</summary>

```python
# Task 1: manual G(x) formula
def mbbefd_exposure_curve_manual(x: np.ndarray, g: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    A = (g - 1.0) * b / (1.0 - b)
    C = (1.0 - g * b) / (1.0 - b)
    log_gb = np.log(g * b)
    log_b  = np.log(b)
    bx     = np.exp(x * log_b)
    inner  = A + C * bx
    inner  = np.where(inner <= 0.0, 1e-300, inner)
    result = np.log(inner) / log_gb
    # Enforce boundaries
    result = np.where(x <= 0.0, 0.0, result)
    result = np.where(x >= 1.0, 1.0, result)
    result = np.clip(result, 0.0, 1.0)
    return result

# Verify against library
g, b = y2.g, y2.b
xs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
manual  = mbbefd_exposure_curve_manual(xs, g, b)
library = y2.exposure_curve(xs)
max_err = np.max(np.abs(manual - library))
print(f"Max absolute error: {max_err:.2e}")   # should be < 1e-10

for x_val, m, lib in zip(xs, manual, library):
    print(f"  x={x_val:.2f}: manual={m:.6f}, library={lib:.6f}")
```

```python
# Task 2: varying (g, b)
pairs = [(3.0, 5.0), (7.69, 9.02), (20.0, 3.0), (100.0, 0.9), (500.0, 0.3)]
x_grid = np.linspace(0, 1, 500)

fig, ax = plt.subplots(figsize=(9, 6))
for (g, b) in pairs:
    dist = MBBEFDDistribution(g=g, b=b)
    ax.plot(x_grid, dist.exposure_curve(x_grid),
            label=f"g={g:.0f}, b={b:.2f}  (total loss={dist.total_loss_prob():.1%})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Diagonal")
ax.set_xlabel("Fraction of MPL")
ax.set_ylabel("G(x)")
ax.set_title("Effect of g and b on the exposure curve")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Visual observations: g controls the height of the curve at x close to 1 -- higher g means the total loss probability 1/g is smaller, so the curve approaches 1.0 more gradually in the upper tail. b controls the curvature: high b (>1) concentrates the curve in the lower-left (many small losses), while b < 1 flattens the curve toward the lower-right (losses spread more uniformly across the severity range).

```python
# Task 3: table
print(f"{'g':>8} {'b':>8} {'P(total)':>10} {'mean z':>8}")
print("-" * 40)
for (g, b) in pairs:
    d = MBBEFDDistribution(g=g, b=b)
    print(f"{g:>8.1f} {b:>8.2f} {d.total_loss_prob():>10.2%} {d.mean():>8.4f}")
```

The relationship: `P(total loss) = 1/g` exactly, so higher g gives lower total loss probability. b and mean destruction rate are inversely related for b > 1 (higher b = more small losses = lower mean). For b < 1 the relationship reverses.

```python
# Task 4: numerical integration
from scipy import integrate

def integrand(x: float) -> float:
    return x * float(y2.pdf(np.array([x]))[0])

e_x_continuous, _ = integrate.quad(integrand, 0.0, 1.0 - 1e-9)
e_x_total = 1.0 * y2.total_loss_prob()      # contribution from z=1
mean_numerical = e_x_continuous + e_x_total

print(f"Numerical mean:  {mean_numerical:.6f}")
print(f"Analytic mean:   {y2.mean():.6f}")
print(f"Difference:      {abs(mean_numerical - y2.mean()):.2e}")
```

The difference should be well below 0.001. Any larger discrepancy indicates numerical issues with the integration bounds (integrate.quad avoids the discontinuity at x=1 by setting the upper limit to 1 - 1e-9, which excludes the point mass; we add it explicitly via `total_loss_prob()`).

</details>

---

## Exercise 2: Curve selection by class of business

**Reference:** Tutorial Part 6

**What you will do:** Apply the Swiss Re curve selection guidelines to a realistic portfolio of UK commercial property risks, and quantify the pricing error from using the wrong curve.

### Setup

You are the pricing actuary for a UK commercial lines insurer. You have three portfolios to price for a per-risk XL layer of £500k xs £250k:

| Portfolio | Description | Count | Avg SI | Total SP |
|-----------|-------------|-------|--------|----------|
| A | Mixed retail, restaurants, hair salons | 850 | £350,000 | £420,000 |
| B | Light industrial units, sprinkler-protected | 320 | £800,000 | £310,000 |
| C | Heavy engineering, construction yards | 95 | £2,500,000 | £280,000 |

```python
import pandas as pd
import numpy as np
from insurance_ilf import swiss_re_curve, per_risk_xl_rate

portfolio_a = pd.DataFrame({
    "sum_insured": [350_000], "premium": [420_000], "count": [850]
})
portfolio_b = pd.DataFrame({
    "sum_insured": [800_000], "premium": [310_000], "count": [320]
})
portfolio_c = pd.DataFrame({
    "sum_insured": [2_500_000], "premium": [280_000], "count": [95]
})

attachment = 250_000
limit      = 500_000
```

### Tasks

**Task 1.** For each portfolio, select the most appropriate Swiss Re curve and justify your choice in one sentence each. Then compute the technical rate and rate on line for each portfolio using your chosen curves.

**Task 2.** For Portfolio B (light industrial, sprinkled), compute the technical rate using all five Swiss Re curves. Produce a sensitivity table showing: curve name, c-parameter, technical rate (% SP), and rate on line. What is the ratio between the highest and lowest rates in the table?

**Task 3.** Suppose your company has been using Y2 for all three portfolios without differentiation. Compute the repricing impact: the difference in technical rate between the Y2 rate and your correctly-selected curve rate, for each portfolio. Express as both a percentage-point difference and as a sterling amount on the actual subject premium.

**Task 4.** For Portfolio C (heavy engineering), plot the expected layer loss as a function of attachment point, from AP = 0 to AP = £2,000,000 in steps of £100,000, using the curve you selected. Label the current attachment (£250k) on the plot. What does the shape of this curve tell you about where in the severity range most losses are occurring?

<details>
<summary>Solution -- Exercise 2</summary>

```python
# Task 1: Curve selection
# Portfolio A: mixed retail/restaurants -> Y1 (light commercial, higher total loss probability)
# Portfolio B: light industrial sprinkled -> Y2 (sprinklers reduce total loss probability;
#              sprinkler-protected risks have lower probability of fires spreading to full
#              destruction compared to unprotected light industrial -- use Y2, not Y1)
# Portfolio C: heavy engineering -> Y3 (robust construction, low total loss probability)

selections = {"A": 1.5, "B": 2.0, "C": 3.0}
portfolios = {"A": portfolio_a, "B": portfolio_b, "C": portfolio_c}

print("Technical rates with selected curves:")
for name, profile in portfolios.items():
    dist = swiss_re_curve(selections[name])
    r = per_risk_xl_rate(profile, dist, attachment, limit)
    print(f"  Portfolio {name} (Y{[1,2,3][list(selections.values()).index(selections[name])]}): "
          f"rate={r['technical_rate']:.3%}, ROL={r['rol']:.4%}")
```

```python
# Task 2: Sensitivity table for Portfolio B
print("\nPortfolio B sensitivity to curve choice:")
print(f"{'Curve':8} {'c':>5} {'Tech rate':>12} {'ROL':>10}")
print("-" * 40)
c_labels = {1.5: "Y1", 2.0: "Y2", 3.0: "Y3", 4.0: "Y4", 5.0: "Lloyds"}
rates_b = {}
for c, label in c_labels.items():
    dist = swiss_re_curve(c)
    r = per_risk_xl_rate(portfolio_b, dist, attachment, limit)
    rates_b[label] = r["technical_rate"]
    print(f"{label:8} {c:>5.1f} {r['technical_rate']:>12.3%} {r['rol']:>10.4%}")

ratio = max(rates_b.values()) / min(rates_b.values())
print(f"\nRatio max/min: {ratio:.1f}x")
```

The ratio from Y1 to Lloyd's is typically 8-12x for this layer structure.

```python
# Task 3: Repricing impact vs Y2 baseline
print("\nRepricing impact vs Y2 (universal):")
for name, profile in portfolios.items():
    y2_rate    = per_risk_xl_rate(profile, swiss_re_curve(2.0), attachment, limit)["technical_rate"]
    corr_rate  = per_risk_xl_rate(profile, swiss_re_curve(selections[name]), attachment, limit)["technical_rate"]
    sp         = float(profile["premium"].sum())
    diff_pct   = corr_rate - y2_rate
    diff_gbp   = diff_pct * sp
    print(f"  Portfolio {name}:  Y2={y2_rate:.3%},  correct={corr_rate:.3%},  "
          f"diff={diff_pct:+.3%}  (£{diff_gbp:+,.0f})")
```

```python
# Task 4: Expected layer loss vs attachment point for Portfolio C
y3 = swiss_re_curve(3.0)
attachments = np.arange(0, 2_100_000, 100_000)
el_vals = []
si_c = float(portfolio_c["sum_insured"].iloc[0])
sp_c = float(portfolio_c["premium"].iloc[0])
count_c = int(portfolio_c["count"].iloc[0])

for ap in attachments:
    r = per_risk_xl_rate(
        portfolio_c, y3, attachment=float(ap), limit=limit
    )
    el_vals.append(r["total_expected_loss"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(attachments / 1e6, el_vals, "b-", linewidth=2)
ax.axvline(x=attachment / 1e6, color="r", linestyle="--",
           label=f"Current attachment £{attachment/1e3:.0f}k")
ax.set_xlabel("Attachment point (£m)")
ax.set_ylabel("Expected layer loss (£)")
ax.set_title(f"Expected layer loss vs attachment -- Portfolio C (Y3, £{limit/1e3:.0f}k limit)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

The expected layer loss falls steeply as the attachment rises above the mean sum insured, because fewer and fewer risks have their SI above the attachment. The shape reflects both the loss severity distribution (Y3 exposure curve) and the concentration of risks in the lower SI bands.

</details>

---

## Exercise 3: Fitting your own curve

**Reference:** Tutorial Parts 8 and 9

**What you will do:** Fit an MBBEFD distribution to a sample of claims data, check convergence, and compare the result to the standard Swiss Re curves.

### Setup

Generate a synthetic claims dataset. In production you would read from a claims table; here we simulate data from a known distribution so we can check recovery.

```python
import numpy as np
from insurance_ilf import MBBEFDDistribution, fit_mbbefd
from insurance_ilf.curves import all_swiss_re_curves

rng = np.random.default_rng(seed=1234)

# True distribution: c = 2.5 (between Y2 and Y3)
true_c   = 2.5
true_dist = MBBEFDDistribution.from_c(true_c)
N_CLAIMS  = 400

# Sample
z_true = true_dist.rvs(N_CLAIMS, rng=rng)
print(f"True distribution: c={true_c}, g={true_dist.g:.3f}, b={true_dist.b:.3f}")
print(f"True total loss prob: {true_dist.total_loss_prob():.2%}")
print(f"True mean z:          {true_dist.mean():.4f}")
print(f"Sample: n={N_CLAIMS}, total losses={( z_true >= 1.0-1e-9).sum()}")
```

### Tasks

**Task 1.** Fit MBBEFD using MLE (`method='mle'`). Print the fitted parameters g and b, the log-likelihood, the AIC, and whether the fit converged. How close are the fitted parameters to the true values (g and b from `MBBEFDDistribution.from_c(2.5)`)?

**Task 2.** Fit using the method of moments (`method='mme'`). Compare the fitted parameters and AIC between MLE and MME. Which gives a better AIC? Why would you prefer MLE in practice despite both converging?

**Task 3.** Compute the fitted distribution's equivalent c-parameter using `fitted_dist.to_c()`. If it returns `None`, explain why. If it returns a value, compare it to the true c = 2.5.

**Task 4.** Use `GoodnessOfFit` to run the KS and Anderson-Darling tests. Then produce a three-panel figure: QQ plot, PP plot, and exposure curve comparison (fitted vs empirical, with Y2 and Y3 standard curves for context). Comment briefly on each panel.

**Task 5.** Fit the same data three times with different `n_starts` values: 2, 6, and 12. Compare the AIC across the three fits. Does increasing the number of starts improve the result? What does this tell you about the practical choice of `n_starts`?

<details>
<summary>Solution -- Exercise 3</summary>

```python
# Task 1: MLE fit
from insurance_ilf import GoodnessOfFit, compare_curves
from insurance_ilf.curves import empirical_exposure_curve

result_mle = fit_mbbefd(z_true, method="mle")
fitted_mle  = result_mle.dist

print("MLE fit:")
print(f"  g = {result_mle.params['g']:.4f}  (true: {true_dist.g:.4f})")
print(f"  b = {result_mle.params['b']:.4f}  (true: {true_dist.b:.4f})")
print(f"  loglik = {result_mle.loglik:.2f}")
print(f"  AIC    = {result_mle.aic:.2f}")
print(f"  BIC    = {result_mle.bic:.2f}")
print(f"  converged: {result_mle.converged}")
```

With N = 400, fitted g and b should be within 20-30% of the true values. The log-likelihood surface is relatively flat in the g-b plane, so exact recovery is not expected.

```python
# Task 2: MME fit
result_mme = fit_mbbefd(z_true, method="mme")
fitted_mme  = result_mme.dist

print("\nMME fit:")
print(f"  g = {result_mme.params['g']:.4f},  b = {result_mme.params['b']:.4f}")
print(f"  AIC = {result_mme.aic:.2f}")
print(f"\nAIC comparison: MLE={result_mle.aic:.2f}, MME={result_mme.aic:.2f}")
print(f"MLE improvement: {result_mme.aic - result_mle.aic:.2f} AIC points")
```

MLE will typically have a lower AIC by 2-10 points. Prefer MLE: it is statistically efficient (asymptotically achieves the Cramér-Rao lower bound for variance of parameter estimates) and its AIC is comparable to other parametric models. MME is a useful starting point or sanity check, not a production method.

```python
# Task 3: c-parameter recovery
c_fitted = fitted_mle.to_c()
if c_fitted is not None:
    print(f"Fitted c ≈ {c_fitted:.2f}  (true c = {true_c})")
else:
    print("Fitted (g, b) do not sit on the Swiss Re c-parameter path.")
    print("This is expected: MLE optimises over all of (g, b) space,")
    print("not just the Swiss Re c-parameter manifold.")
```

`to_c()` returns `None` when the fitted (g, b) do not correspond to a Swiss Re standard curve within tolerance. This is common: MLE is free to find the best (g, b) anywhere in the parameter space, not just on the c-parameter path.

```python
# Task 4: Goodness of fit
gof = GoodnessOfFit(z_true, fitted_mle)
print(f"KS test: {gof.ks_test()}")
print(f"AD test: {gof.ad_test()}")

ec_empirical = empirical_exposure_curve(
    losses=z_true,       # here z = loss/MPL = loss (since MPL=1 for rvs output)
    mpl=np.ones(N_CLAIMS),
    n_points=50,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
gof.qq_plot(ax=axes[0])
gof.pp_plot(ax=axes[1])
gof.exposure_curve_plot(ax=axes[2], empirical_ec=ec_empirical)
plt.suptitle(f"GOF: MBBEFD(g={fitted_mle.g:.2f}, b={fitted_mle.b:.2f})  --  true c=2.5")
plt.tight_layout()
plt.show()
```

QQ plot: points should follow the diagonal. Deviations in the upper tail (above the last decile) are expected with 400 observations -- sampling noise is large there. PP plot: similar diagnostic, more sensitive to the bulk. Exposure curve: the fitted smooth curve should pass through the cloud of empirical points.

```python
# Task 5: n_starts sensitivity
for n_st in [2, 6, 12]:
    r = fit_mbbefd(z_true, n_starts=n_st)
    print(f"  n_starts={n_st:2d}:  AIC={r.aic:.2f},  g={r.params['g']:.4f},  b={r.params['b']:.4f}")
```

In most cases the AIC will be similar across n_starts. The practical choice: 6 is adequate for well-behaved data. Use 12 if you see `converged=False` or if the AIC from two separate runs differs by more than 1 unit (which would indicate the optimiser is finding different local optima on different runs, which should not happen with the same data and the same seeded starting points -- if it does, the data may be pathological).

</details>

---

## Exercise 4: Handling truncated and censored data

**Reference:** Tutorial Part 9

**What you will do:** Demonstrate that ignoring data constraints biases the fitted curve, and quantify the effect on layer pricing.

### Setup

```python
import numpy as np
from insurance_ilf import MBBEFDDistribution, fit_mbbefd, swiss_re_curve, layer_expected_loss

rng = np.random.default_rng(seed=42)
true_dist = MBBEFDDistribution.from_c(2.0)   # Y2 true curve

# Generate 1000 untruncated destruction rates
N = 1000
z_full = true_dist.rvs(N, rng=rng)

# Apply data constraints
DEDUCTIBLE    = 0.05    # 5% of MPL; claims below not reported
POLICY_LIMIT  = 0.80    # 80% of MPL; claims above capped

# What we observe in the claims system
z_observed = z_full[z_full >= DEDUCTIBLE]                # remove sub-deductible
z_observed = np.where(z_observed > POLICY_LIMIT,
                      POLICY_LIMIT, z_observed)          # cap at policy limit

print(f"Full sample:     {len(z_full):,}  claims")
print(f"Observed sample: {len(z_observed):,}  claims "
      f"({len(z_observed)/len(z_full):.0%} retained after truncation)")
print(f"Capped at 0.80:  {(z_observed == POLICY_LIMIT).sum()}  observations")
```

### Tasks

**Task 1.** Fit three models:
- **Naive**: fit to `z_observed` with no truncation/censoring arguments
- **Truncation-only**: fit with `truncation=0.05`, no censoring
- **Correct**: fit with both `truncation=0.05` and `censoring=0.80`

Print a table of g, b, total loss probability, mean, and AIC for each model.

**Task 2.** For each of the three fitted curves, compute the expected layer loss for a layer of £500k xs £500k on a risk with MPL = £2,000,000 and subject premium = £30,000. State which fitted curve gives the result closest to the true curve (Y2) and by how much.

**Task 3.** Plot all three fitted exposure curves plus the true Y2 curve on a single chart. Which of the three is closest to the truth? How does this explain your Task 2 findings?

**Task 4.** Repeat Task 1 using only the top 200 observations (i.e., the 200 largest destruction rates from the observed sample). How does fit quality degrade with fewer observations? Does the ordering of naive vs correct fit change?

<details>
<summary>Solution -- Exercise 4</summary>

```python
# Task 1: Three fits
result_naive  = fit_mbbefd(z_observed)
result_trunc  = fit_mbbefd(z_observed, truncation=DEDUCTIBLE)
result_correct = fit_mbbefd(z_observed, truncation=DEDUCTIBLE, censoring=POLICY_LIMIT)

print(f"{'Model':15} {'g':>8} {'b':>8} {'P(total)':>10} {'mean_z':>8} {'AIC':>10}")
print("-" * 60)
for label, r in [("Naive", result_naive), ("Trunc only", result_trunc), ("Correct", result_correct)]:
    print(f"{label:15} {r.params['g']:>8.4f} {r.params['b']:>8.4f} "
          f"{r.dist.total_loss_prob():>10.2%} {r.dist.mean():>8.4f} {r.aic:>10.2f}")

# True values for reference
print(f"{'True Y2':15} {true_dist.g:>8.4f} {true_dist.b:>8.4f} "
      f"{true_dist.total_loss_prob():>10.2%} {true_dist.mean():>8.4f} {'--':>10}")
```

Expected findings: the naive fit will overestimate mean z (because small losses below 5% are missing and the model treats the sample as complete). The truncation-only fit corrects for this. The correct fit further adjusts for the censoring at 80%, which without correction compresses all the mass above 80% into the z=0.80 point rather than letting it spread into the total-loss atom at z=1.

```python
# Task 2: Layer pricing
attachment = 500_000
limit      = 500_000
policy_limit = 2_000_000
mpl          = 2_000_000
subject_premium = 30_000

true_el = layer_expected_loss(true_dist, attachment, limit, policy_limit, mpl, subject_premium)
print(f"\nTrue (Y2):     £{true_el:,.0f}")

for label, r in [("Naive", result_naive), ("Trunc only", result_trunc), ("Correct", result_correct)]:
    el = layer_expected_loss(r.dist, attachment, limit, policy_limit, mpl, subject_premium)
    err = (el - true_el) / true_el
    print(f"{label:15}: £{el:,.0f}   error {err:+.1%}")
```

```python
# Task 3: Plot
x_grid = np.linspace(0, 1, 500)
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_grid, true_dist.exposure_curve(x_grid), "k-", linewidth=2.5, label="True Y2")
for label, r, ls in [
    ("Naive", result_naive, "--"),
    ("Trunc only", result_trunc, "-."),
    ("Correct", result_correct, ":"),
]:
    ax.plot(x_grid, r.dist.exposure_curve(x_grid), ls, linewidth=1.8, label=label)

ax.axvline(x=DEDUCTIBLE, color="gray", linestyle=":", alpha=0.6, label="Deductible")
ax.axvline(x=POLICY_LIMIT, color="gray", linestyle="-.", alpha=0.6, label="Policy limit")
ax.set_xlabel("Fraction of MPL")
ax.set_ylabel("G(x)")
ax.set_title("Effect of ignoring truncation and censoring")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# Task 4: Thin data
z_top200 = np.sort(z_observed)[-200:]

result_naive_200   = fit_mbbefd(z_top200)
result_correct_200 = fit_mbbefd(z_top200, truncation=DEDUCTIBLE, censoring=POLICY_LIMIT)

print("\nWith only 200 largest observations:")
print(f"{'Model':15} {'g':>8} {'b':>8} {'mean_z':>8} {'AIC':>10}")
for label, r in [("Naive (200)", result_naive_200), ("Correct (200)", result_correct_200)]:
    print(f"{label:15} {r.params['g']:>8.4f} {r.params['b']:>8.4f} "
          f"{r.dist.mean():>8.4f} {r.aic:>10.2f}")
```

With only 200 observations (and further selection bias -- these are the 200 largest), the naive fit will be severely upward-biased in mean z. The correct fit still adjusts for the known constraints but has much wider uncertainty. The message: thin data makes the correct/naive distinction even more important, not less.

</details>

---

## Exercise 5: ILF tables in practice

**Reference:** Tutorial Part 11

**What you will do:** Build ILF tables for a commercial liability portfolio and use them to rate policies at multiple limit levels.

### Context

You are pricing a UK employer's liability portfolio. The basic limit is £5 million. You need ILF tables at limits of £10m, £20m, £25m (the Employers' Liability (Compulsory Insurance) Act 1969 minimum is £5m, but most larger employers buy higher limits).

The MPL for employers' liability is difficult to define precisely -- some actuaries use the legal cap on damages as a proxy, others use the reinsurance treaty limit. For this exercise, use MPL = £50 million as a conservative upper bound.

```python
import pandas as pd
import numpy as np
from insurance_ilf import swiss_re_curve, ilf_table

limits_el = [5_000_000, 10_000_000, 20_000_000, 25_000_000, 50_000_000]
basic_limit = 5_000_000
mpl         = 50_000_000
```

### Tasks

**Task 1.** Build ILF tables for Y2 and Y3 at the limits above. Print both tables side by side. Which column is more sensitive to curve choice: the ILF at £10m or the ILF at £25m?

**Task 2.** You have a portfolio of 500 employers' liability policies, all at the £5m basic limit. The total premium at £5m is £1,200,000. Suppose 80 of these policies upgrade to £10m limits. Using both Y2 and Y3, compute:
- The additional premium for the 80 upgrading policies
- The new total portfolio premium
- The difference in total portfolio premium between Y2 and Y3

**Task 3.** The marginal ILF column shows the incremental factor from the previous limit. Confirm that the sum of all marginal ILFs equals the ILF at the highest limit. Then explain in two sentences what the marginal ILF tells an underwriter about pricing high limits.

**Task 4.** A competitor is offering £25m policies at 1.80 times the basic limit rate. Based on your ILF tables, are they under- or over-pricing the high limits? Which Swiss Re curve would justify their 1.80 ILF?

<details>
<summary>Solution -- Exercise 5</summary>

```python
# Task 1: ILF tables
y2 = swiss_re_curve(2.0)
y3 = swiss_re_curve(3.0)

table_y2 = ilf_table(y2, limits_el, basic_limit, mpl)
table_y3 = ilf_table(y3, limits_el, basic_limit, mpl)

# Merge for comparison
merged = table_y2[["limit", "ilf"]].copy().rename(columns={"ilf": "ilf_Y2"})
merged["ilf_Y3"] = table_y3["ilf"].values
merged["ratio"]  = merged["ilf_Y3"] / merged["ilf_Y2"]
print(merged.to_string(index=False))

# Which limit is more sensitive?
# The ratio ilf_Y3/ilf_Y2 increases with limit, so high limits are more sensitive.
```

```python
# Task 2: Premium calculation
per_policy_premium = 1_200_000 / 500   # £2,400 average at £5m limit

print("\nAdditional premium for 80 policies upgrading to £10m:")
for label, tbl in [("Y2", table_y2), ("Y3", table_y3)]:
    ilf_10m = float(tbl.loc[tbl["limit"] == 10_000_000, "ilf"])
    additional = per_policy_premium * (ilf_10m - 1.0) * 80
    new_total  = 1_200_000 + additional
    print(f"  {label}: ILF={ilf_10m:.3f},  additional=£{additional:,.0f},  "
          f"new total=£{new_total:,.0f}")

# Difference between Y2 and Y3 totals
for (label_a, tbl_a), (label_b, tbl_b) in [
    (("Y2", table_y2), ("Y3", table_y3))
]:
    ilf_y2 = float(tbl_a.loc[tbl_a["limit"] == 10_000_000, "ilf"])
    ilf_y3 = float(tbl_b.loc[tbl_b["limit"] == 10_000_000, "ilf"])
    add_y2 = per_policy_premium * (ilf_y2 - 1.0) * 80
    add_y3 = per_policy_premium * (ilf_y3 - 1.0) * 80
    print(f"\nPremium difference (Y3 vs Y2): £{add_y3 - add_y2:,.0f}")
```

```python
# Task 3: Marginal ILF identity
# Sum of marginal ILFs should equal the ILF at the highest limit
sum_marginal = table_y2["marginal_ilf"].sum()
ilf_max      = float(table_y2.loc[table_y2["limit"] == max(limits_el), "ilf"])
print(f"\nSum of marginal ILFs: {sum_marginal:.4f}")
print(f"ILF at max limit:     {ilf_max:.4f}")
print(f"Match: {abs(sum_marginal - ilf_max) < 0.001}")
```

The marginal ILF tells an underwriter: how much extra does each additional tranche of limit cost? Because the exposure curve is concave, each successive £5m of limit costs less than the previous one. The marginal ILF at £25m is smaller than at £10m, reflecting that the uppermost part of the severity range is very thin.

```python
# Task 4: Competitor pricing
competitor_ilf = 1.80
print("\nCurve-implied ILF at £25m for each Swiss Re curve:")
from insurance_ilf.curves import all_swiss_re_curves
for name, dist in all_swiss_re_curves().items():
    tbl = ilf_table(dist, [25_000_000], basic_limit, mpl)
    ilf_25m = float(tbl.loc[tbl["limit"] == 25_000_000, "ilf"])
    verdict = "below" if ilf_25m > competitor_ilf else "above"
    print(f"  {name:8}: {ilf_25m:.3f}  (competitor at {competitor_ilf:.2f} is {verdict} this)")
```

A competitor charging ILF = 1.80 for £25m is pricing below Y1 (the softest curve). That implies they believe the account has higher total loss probability than Y1, which for employer's liability is unusual. More likely they are using a flat-file ILF derived from market benchmark without fitting to their own data. The correct curve (probably Y2 or Y3 for most UK EL accounts) implies a substantially higher ILF.

</details>

---

## Exercise 6: Pricing a per-risk XL treaty

**Reference:** Tutorial Part 13

**What you will do:** Price a per-risk XL treaty from a realistic cedant risk profile, run sensitivity analysis, and produce an underwriting memorandum summary.

### Context

You are quoting a per-risk XL treaty for a UK commercial property MGA. The MGA writes primarily retail property in the North West of England. The layer under consideration is £2m xs £1m.

```python
import pandas as pd
import numpy as np
from insurance_ilf import swiss_re_curve, per_risk_xl_rate, layer_expected_loss

# Risk profile provided by the MGA
risk_profile = pd.DataFrame({
    "sum_insured":   [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000,
                      3_000_000, 5_000_000],
    "premium":       [95_000, 130_000, 185_000, 160_000, 140_000,
                      90_000, 60_000],
    "count":         [190, 173, 185, 107, 70, 30, 12],
})

attachment = 1_000_000
limit      = 2_000_000
```

### Tasks

**Task 1.** Price the layer using Y1, Y2, and Y3. Produce a table showing, for each curve: total subject premium, total expected loss, technical rate (% SP), and rate on line.

**Task 2.** The cedant claims the book is "standard retail property" and that Y2 is the appropriate curve. Your underwriting team suspects the true c-parameter is closer to Y1 based on the property types. Quantify the financial impact of this disagreement:
- If Y1 is correct and you price at Y2, how much premium are you leaving on the table (or how much are you overcharging)?
- Assume the cedant buys at your Y2 technical rate. Compute the loss ratio the treaty would run at if the true curve is Y1.

**Task 3.** Run a sweep of the attachment point from £500k to £3m in steps of £250k, keeping the limit at £2m. Plot the technical rate as a function of attachment for both Y1 and Y2. At what attachment does the rate drop below 5%? Below 2%?

**Task 4.** The cedant's risk profile shows 190 risks with SI = £500,000. These cannot pierce the £1m attachment and therefore contribute zero to the layer. Recompute the treaty assuming the cedant agrees to exclude sub-£1m risks from the subject premium entirely (i.e., the subject premium base is reduced). How does this change the technical rate? Is this a better or worse deal for the cedant?

<details>
<summary>Solution -- Exercise 6</summary>

```python
# Task 1: Three curve pricing
print(f"Layer: £{limit/1e6:.0f}m xs £{attachment/1e6:.0f}m\n")
print(f"{'Curve':8} {'Total SP':>14} {'Total EL':>12} {'Tech rate':>12} {'ROL':>10}")
print("-" * 58)

results = {}
for name, c in [("Y1", 1.5), ("Y2", 2.0), ("Y3", 3.0)]:
    dist = swiss_re_curve(c)
    r = per_risk_xl_rate(risk_profile, dist, attachment, limit)
    results[name] = r
    print(f"{name:8} £{r['subject_premium']:>12,.0f} £{r['total_expected_loss']:>10,.0f} "
          f"{r['technical_rate']:>12.3%} {r['rol']:>10.4%}")
```

```python
# Task 2: Financial impact of curve disagreement
y1_rate = results["Y1"]["technical_rate"]
y2_rate = results["Y2"]["technical_rate"]
total_sp = results["Y2"]["subject_premium"]

# If you charge Y2 rate but true is Y1:
y2_premium_charged = y2_rate * total_sp
y1_expected_loss   = y1_rate * total_sp   # true expected loss

print(f"\nY2 rate charged:        {y2_rate:.3%}   (premium £{y2_premium_charged:,.0f})")
print(f"Y1 expected loss:       {y1_rate:.3%}   (expected £{y1_expected_loss:,.0f})")
print(f"Difference:             {(y2_rate - y1_rate):+.3%}  (£{y2_premium_charged - y1_expected_loss:+,.0f})")

treaty_lr = y1_expected_loss / y2_premium_charged
print(f"\nTreaty loss ratio if Y1 true, priced at Y2: {treaty_lr:.1%}")
```

If Y1 is true and you price at Y2, you are undercharging: the expected loss (Y1) exceeds the premium you collected (Y2-rated). The treaty runs at a loss ratio above 100%, which is the exposure-rated equivalent of inadequate pricing.

```python
# Task 3: Attachment sweep
attachments = np.arange(500_000, 3_250_000, 250_000)
rates_y1 = []
rates_y2 = []

for ap in attachments:
    for name, rates_list in [("Y1", rates_y1), ("Y2", rates_y2)]:
        dist = swiss_re_curve(1.5 if name == "Y1" else 2.0)
        r = per_risk_xl_rate(risk_profile, dist, float(ap), limit)
        rates_list.append(r["technical_rate"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(attachments / 1e6, [r * 100 for r in rates_y1], "b-", label="Y1", linewidth=2)
ax.plot(attachments / 1e6, [r * 100 for r in rates_y2], "r-", label="Y2", linewidth=2)
ax.axhline(y=5, color="gray", linestyle="--", alpha=0.6, label="5% level")
ax.axhline(y=2, color="gray", linestyle=":",  alpha=0.6, label="2% level")
ax.set_xlabel("Attachment point (£m)")
ax.set_ylabel("Technical rate (% of SP)")
ax.set_title("Technical rate vs attachment -- Y1 and Y2")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find crossover attachments
for threshold in [0.05, 0.02]:
    for name, rates_list in [("Y1", rates_y1), ("Y2", rates_y2)]:
        cross = [(attachments[i]/1e6, attachments[i+1]/1e6)
                 for i in range(len(rates_list)-1)
                 if rates_list[i] > threshold >= rates_list[i+1]]
        if cross:
            print(f"{name} rate drops below {threshold:.0%} between £{cross[0][0]:.2f}m and £{cross[0][1]:.2f}m")
```

```python
# Task 4: Exclude sub-£1m risks from SP base
profile_above_ap = risk_profile[risk_profile["sum_insured"] >= attachment].copy()

print("\nOriginal profile (all risks):")
r_all = per_risk_xl_rate(risk_profile, swiss_re_curve(2.0), attachment, limit)
print(f"  SP = £{r_all['subject_premium']:,.0f},  tech rate = {r_all['technical_rate']:.3%}")

print("\nReduced profile (SI >= £1m only):")
r_reduced = per_risk_xl_rate(profile_above_ap, swiss_re_curve(2.0), attachment, limit)
print(f"  SP = £{r_reduced['subject_premium']:,.0f},  tech rate = {r_reduced['technical_rate']:.3%}")
```

The expected layer loss is identical (sub-£1m risks contribute zero anyway). But the technical rate rises when the SP base shrinks. The cedant pays the same absolute premium but it is expressed as a higher percentage of a smaller base. For a net-cost-focused cedant, the absolute premium is what matters; the technical rate is how the reinsurer measures adequacy. Excluding sub-attachment risks from the subject premium base is standard practice in treaty work and produces a cleaner rate.

</details>

---

## Exercise 7: The Lee diagram as a communication tool

**Reference:** Tutorial Part 14

**What you will do:** Use the Lee diagram to identify whether a portfolio has unusual loss concentration, and communicate the finding to a non-technical underwriter.

### Setup

Generate two synthetic portfolios: one well-fitted by MBBEFD, one with an unusually heavy tail.

```python
import numpy as np
import matplotlib.pyplot as plt
from insurance_ilf import MBBEFDDistribution, lee_diagram, fit_mbbefd

rng = np.random.default_rng(seed=99)

# Portfolio A: standard Y2 losses
dist_a = MBBEFDDistribution.from_c(2.0)
z_a = dist_a.rvs(300, rng=rng)
mpl_a = np.ones(300)  # normalised

# Portfolio B: heavy-tailed (unusual -- add some very large partial losses)
dist_b_base = MBBEFDDistribution.from_c(2.0)
z_b = dist_b_base.rvs(300, rng=rng)
# Inject 10 catastrophic partial losses (destruction rates 0.85-0.99)
z_b[rng.choice(300, size=10, replace=False)] = rng.uniform(0.85, 0.99, 10)
mpl_b = np.ones(300)
```

### Tasks

**Task 1.** Fit MBBEFD to both portfolios. For each, produce a Lee diagram overlaid with the fitted G(x). Where does the empirical Lee curve deviate from the fitted G(x) in Portfolio B? What does this deviation mean for XL pricing?

**Task 2.** Compute the total expected loss in a layer of £800k xs £200k (as a fraction of total expected loss, per unit of MPL = 1) using:
- The fitted MBBEFD for Portfolio B
- The Swiss Re Y2 standard curve

Compare the two estimates. Which is higher? Why?

**Task 3.** Write a three-sentence paragraph suitable for an underwriter, explaining what the Lee diagram for Portfolio B shows and why it should affect the reinsurance pricing. Avoid actuarial jargon: no "MBBEFD," no "G(x)."

**Task 4.** Produce a combined 2x2 figure with:
- Top left: Lee diagram for Portfolio A with fitted curve
- Top right: Lee diagram for Portfolio B with fitted curve
- Bottom left: Fitted exposure curve for A vs Y2 reference
- Bottom right: Fitted exposure curve for B vs Y2 reference

<details>
<summary>Solution -- Exercise 7</summary>

```python
# Task 1: Fit and Lee diagrams
result_a = fit_mbbefd(z_a)
result_b = fit_mbbefd(z_b)

print(f"Portfolio A fit:  g={result_a.params['g']:.2f}, b={result_a.params['b']:.2f}, "
      f"AIC={result_a.aic:.1f}")
print(f"Portfolio B fit:  g={result_b.params['g']:.2f}, b={result_b.params['b']:.2f}, "
      f"AIC={result_b.aic:.1f}")

# KS tests
from insurance_ilf import GoodnessOfFit
for name, z, r in [("A", z_a, result_a), ("B", z_b, result_b)]:
    gof = GoodnessOfFit(z, r.dist)
    ks = gof.ks_test()
    print(f"Portfolio {name} KS: stat={ks['statistic']:.4f}, p={ks['p_value']:.4f}")
```

For Portfolio B, the KS p-value should be lower (possibly < 0.05) because the injected extreme losses push the empirical Lee curve above the fitted curve in the upper tail.

```python
# Task 2: Layer pricing comparison
from insurance_ilf import layer_expected_loss
attachment_t2 = 0.2
limit_t2      = 0.8

el_fitted_b = layer_expected_loss(result_b.dist, attachment_t2, limit_t2,
                                   policy_limit=1.0, mpl=1.0, subject_premium=1.0)
el_y2       = layer_expected_loss(MBBEFDDistribution.from_c(2.0), attachment_t2, limit_t2,
                                   policy_limit=1.0, mpl=1.0, subject_premium=1.0)

print(f"\nExpected layer loss (£0.8 xs £0.2 per unit MPL):")
print(f"  Fitted B: {el_fitted_b:.4f}")
print(f"  Y2 ref:   {el_y2:.4f}")
print(f"  Fitted/Y2 ratio: {el_fitted_b/el_y2:.3f}")
```

The fitted curve for Portfolio B will have a higher expected layer loss than Y2 because the fitting algorithm responds to the extreme losses by flattening the curve -- shifting more expected loss into the upper severity range.

```python
# Task 3: Non-technical explanation
explanation = """
Portfolio B contains a cluster of losses where properties have suffered severe
damage reaching 85-99% of their insured value -- fires or flood events that
nearly (but did not quite) total the property. When we look at how losses are
distributed across the portfolio, 10% of the claims account for more than 50%
of the total loss amount. This is unusually concentrated compared to a normal
commercial property book. When pricing the reinsurance layer, this concentration
means the layer attaches more often and more severely than the standard tables
would predict. The correct reinsurance rate is therefore higher than the standard
starting point.
"""
print(explanation)
```

```python
# Task 4: Combined figure
x_grid = np.linspace(0, 1, 500)
y2_ref = MBBEFDDistribution.from_c(2.0)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top left: Lee A
lee_diagram(z_a, mpl_a, dist=result_a.dist, ax=axes[0, 0])
axes[0, 0].set_title("Portfolio A -- Lee Diagram")

# Top right: Lee B
lee_diagram(z_b, mpl_b, dist=result_b.dist, ax=axes[0, 1])
axes[0, 1].set_title("Portfolio B -- Lee Diagram (heavy tail)")

# Bottom left: Exposure curve A
axes[1, 0].plot(x_grid, y2_ref.exposure_curve(x_grid), "k--", label="Y2 reference", alpha=0.6)
axes[1, 0].plot(x_grid, result_a.dist.exposure_curve(x_grid), "b-", linewidth=2,
                label=f"Fitted A (g={result_a.dist.g:.1f})")
axes[1, 0].set_xlabel("Fraction of MPL")
axes[1, 0].set_ylabel("G(x)")
axes[1, 0].set_title("Portfolio A -- Exposure Curve")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Bottom right: Exposure curve B
axes[1, 1].plot(x_grid, y2_ref.exposure_curve(x_grid), "k--", label="Y2 reference", alpha=0.6)
axes[1, 1].plot(x_grid, result_b.dist.exposure_curve(x_grid), "r-", linewidth=2,
                label=f"Fitted B (g={result_b.dist.g:.1f})")
axes[1, 1].set_xlabel("Fraction of MPL")
axes[1, 1].set_ylabel("G(x)")
axes[1, 1].set_title("Portfolio B -- Exposure Curve (heavy tail effect)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

</details>

---

## Exercise 8: Fitting from an empirical exposure curve

**Reference:** Tutorial Part 8; `fit_exposure_curve` in the library

**What you will do:** Fit an MBBEFD distribution to a pre-computed empirical exposure curve rather than to raw claims data. This is the workflow when you receive summary data (e.g., from the R `mbbefd` package or from a colleague's Excel model) rather than individual claim records.

### Context

A London market cedant has provided you with an empirical exposure curve (pre-computed by their internal team) rather than individual claim records. The curve is given as a table of G(x) values at 20 evenly-spaced x-points.

```python
import numpy as np
from insurance_ilf import MBBEFDDistribution, fit_mbbefd
from insurance_ilf.fitting import fit_exposure_curve
from insurance_ilf.curves import empirical_exposure_curve, EmpiricalEC

# Empirical G(x) values provided by cedant (pre-computed)
# True underlying distribution: c = 3.5 (between Y3 and Y4)
rng = np.random.default_rng(55)
true_c = 3.5
true_dist = MBBEFDDistribution.from_c(true_c)

x_points   = np.linspace(0.0, 1.0, 20)
g_true     = true_dist.exposure_curve(x_points)
# Add a small amount of noise to simulate a real empirical curve
g_noisy    = np.clip(g_true + rng.normal(0, 0.01, len(x_points)), 0.0, 1.0)
g_noisy[0] = 0.0    # enforce G(0) = 0
g_noisy[-1] = 1.0   # enforce G(1) = 1
```

### Tasks

**Task 1.** Wrap the cedant's empirical curve in an `EmpiricalEC` object and use its `.fit()` method to fit an MBBEFD distribution. Alternatively, call `fit_exposure_curve(empirical_x=x_points, empirical_g=g_noisy)` directly. Compare the fitted parameters to the true distribution (c = 3.5).

**Task 2.** Also fit the MBBEFD distribution to the empirical curve using the standard Swiss Re curves Y3 and Y4 as starting guesses (manually set `n_starts=2` and see if initialising from nearby curves helps). Compare AIC to the fit from Task 1.

**Task 3.** The cedant's empirical curve was computed from 80 claims. Simulate a scenario where it was computed from only 20 claims by increasing the noise standard deviation to 0.04. Fit MBBEFD to both the low-noise (n=80) and high-noise (n=20) empirical curves. How does the uncertainty in the empirical curve propagate to uncertainty in the fitted MBBEFD parameters?

**Task 4.** Plot the true G(x), the noisy empirical curve (both noise levels), and the two fitted MBBEFD curves on a single chart. Label the Swiss Re c-parameter nearest to each fitted distribution.

<details>
<summary>Solution -- Exercise 8</summary>

```python
# Task 1: Fit from empirical curve
from insurance_ilf.fitting import fit_exposure_curve

result_ec = fit_exposure_curve(empirical_x=x_points, empirical_g=g_noisy, n_starts=8)
fitted_ec = result_ec.dist

print("Fit from empirical curve:")
print(f"  g = {result_ec.params['g']:.4f}  (true: {true_dist.g:.4f})")
print(f"  b = {result_ec.params['b']:.4f}  (true: {true_dist.b:.4f})")
print(f"  method: {result_ec.method},  converged: {result_ec.converged}")

# Equivalent c
c_fitted = fitted_ec.to_c()
if c_fitted:
    print(f"  Equivalent c ≈ {c_fitted:.2f}  (true c = {true_c})")

# Using EmpiricalEC wrapper
ec_obj = EmpiricalEC(x_points=x_points, g_values=g_noisy, n_losses=80)
result_via_obj = ec_obj.fit()
print(f"\nVia EmpiricalEC.fit(): g={result_via_obj.params['g']:.4f}, b={result_via_obj.params['b']:.4f}")
```

```python
# Task 2: AIC comparison -- both methods use the same curve_objective internally,
# so the AIC should be the same given the same data and n_starts.
# But let us try with different n_starts to show robustness.
for n_st in [2, 6, 12]:
    r = fit_exposure_curve(x_points, g_noisy, n_starts=n_st)
    print(f"n_starts={n_st:2d}: AIC={r.aic:.4f},  g={r.params['g']:.4f}")
```

Note: the "AIC" reported by `fit_exposure_curve` is based on negative SSE (sum of squared errors), not a true log-likelihood. It should not be compared directly to AIC from `fit_mbbefd` (which uses the true log-likelihood). It is used only for internal comparison between runs.

```python
# Task 3: Low-noise vs high-noise
rng2 = np.random.default_rng(56)

# High noise (simulate n=20 claims)
g_high_noise = np.clip(g_true + rng2.normal(0, 0.04, len(x_points)), 0.0, 1.0)
g_high_noise[0]  = 0.0
g_high_noise[-1] = 1.0

r_low  = fit_exposure_curve(x_points, g_noisy,      n_starts=8)
r_high = fit_exposure_curve(x_points, g_high_noise, n_starts=8)

print("\nFit quality by noise level:")
print(f"{'Noise':12} {'g':>8} {'b':>8} {'g error':>10} {'b error':>10}")
print("-" * 48)
for label, r in [("Low (n=80)", r_low), ("High (n=20)", r_high)]:
    g_err = r.params['g'] - true_dist.g
    b_err = r.params['b'] - true_dist.b
    print(f"{label:12} {r.params['g']:>8.4f} {r.params['b']:>8.4f} "
          f"{g_err:>+10.4f} {b_err:>+10.4f}")
```

Increased noise in the empirical curve propagates directly to wider uncertainty in the fitted parameters. This illustrates why using `fit_mbbefd` on individual claim records is preferred over `fit_exposure_curve` on a summary curve when the raw data is available: the former uses all the statistical information in the data, the latter discards some.

```python
# Task 4: Plot
x_grid = np.linspace(0, 1, 500)
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_grid, true_dist.exposure_curve(x_grid), "k-", linewidth=2.5, label=f"True (c={true_c})")
ax.scatter(x_points, g_noisy,      s=30, color="blue",  alpha=0.7, label="Empirical (low noise)")
ax.scatter(x_points, g_high_noise, s=30, color="red",   alpha=0.7, marker="x", label="Empirical (high noise)")
ax.plot(x_grid, r_low.dist.exposure_curve(x_grid),  "b--", linewidth=1.5,
        label=f"Fitted low-noise (g={r_low.dist.g:.2f})")
ax.plot(x_grid, r_high.dist.exposure_curve(x_grid), "r--", linewidth=1.5,
        label=f"Fitted high-noise (g={r_high.dist.g:.2f})")

ax.set_xlabel("Fraction of MPL")
ax.set_ylabel("G(x)")
ax.set_title("Fitting from empirical exposure curve")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

</details>

---

## Exercise 9: Integrating with the GLM/GBM pipeline

**Reference:** Tutorial Part 16; Modules 2-4

**What you will do:** Combine per-risk expected loss from a GLM-style model with exposure curve pricing to produce per-policy layer rates. This connects the ground-up pricing work from earlier modules to the reinsurance pricing work in this one.

### Context

You have a portfolio of 200 commercial property risks. Each risk has:
- A sum insured (which equals MPL)
- A GLM-predicted annual expected loss frequency and severity
- A class of business (retail, light industrial, or heavy industrial)

You want to compute the expected loss to a per-risk XL layer of £750k xs £500k for each risk.

```python
import numpy as np
import polars as pl
from insurance_ilf import swiss_re_curve, layer_expected_loss

rng = np.random.default_rng(seed=2026)
N = 200

# Synthetic portfolio with GLM-style predictions
sum_insured = rng.choice(
    [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000],
    N, p=[0.20, 0.20, 0.25, 0.15, 0.15, 0.05]
).astype(float)

class_of_business = rng.choice(
    ["retail", "light_industrial", "heavy_industrial"],
    N, p=[0.50, 0.35, 0.15]
)

# GLM predictions: expected frequency and severity per risk per year
freq  = rng.uniform(0.02, 0.15, N)                        # annual claim frequency
sev   = sum_insured * rng.uniform(0.05, 0.30, N)          # expected severity if claim

# Expected loss = frequency * severity
expected_loss = freq * sev

portfolio = pl.DataFrame({
    "risk_id":     [f"R{i:04d}" for i in range(N)],
    "cob":         class_of_business.tolist(),
    "sum_insured": sum_insured.tolist(),
    "freq":        freq.tolist(),
    "sev":         sev.tolist(),
    "expected_loss": expected_loss.tolist(),
})

attachment = 500_000
limit      = 750_000
```

### Tasks

**Task 1.** Assign a Swiss Re curve to each risk based on its class of business:
- retail -> Y1 (c = 1.5)
- light_industrial -> Y2 (c = 2.0)
- heavy_industrial -> Y3 (c = 3.0)

For each risk, compute the expected loss to the layer using `layer_expected_loss()`, using the risk's `expected_loss` as the subject premium and its `sum_insured` as both the policy limit and the MPL.

Add the result as a column `layer_el` to the portfolio DataFrame.

**Task 2.** Compute the total expected layer loss by class of business. Which class contributes the most in aggregate? Which has the highest average layer rate (layer_el / expected_loss) per risk?

**Task 3.** Find the 10 risks with the highest expected layer loss. Print their risk_id, cob, sum_insured, expected_loss, and layer_el. What do these risks have in common?

**Task 4.** Compute the portfolio-aggregate layer rate: sum(layer_el) / sum(expected_loss). Then compute the same rate using only Y2 for all risks (ignoring the class-differentiated curves). How much does curve differentiation by class change the aggregate rate?

<details>
<summary>Solution -- Exercise 9</summary>

```python
# Task 1: Class-differentiated curve assignment and per-risk layer pricing
curve_map = {
    "retail":           swiss_re_curve(1.5),
    "light_industrial": swiss_re_curve(2.0),
    "heavy_industrial": swiss_re_curve(3.0),
}

layer_el_list = []
for row in portfolio.iter_rows(named=True):
    dist = curve_map[row["cob"]]
    si   = row["sum_insured"]
    el_risk = layer_expected_loss(
        dist=dist,
        attachment=attachment,
        limit=limit,
        policy_limit=si,   # policy limit = sum insured
        mpl=si,            # MPL = sum insured
        subject_premium=row["expected_loss"],
    )
    layer_el_list.append(el_risk)

portfolio = portfolio.with_columns(pl.Series("layer_el", layer_el_list))

print(f"Total expected loss:       £{portfolio['expected_loss'].sum():,.0f}")
print(f"Total expected layer loss: £{portfolio['layer_el'].sum():,.0f}")
print(f"Aggregate layer rate:      {portfolio['layer_el'].sum() / portfolio['expected_loss'].sum():.3%}")
```

```python
# Task 2: Summary by class
summary = portfolio.group_by("cob").agg([
    pl.len().alias("count"),
    pl.col("expected_loss").sum().alias("total_el"),
    pl.col("layer_el").sum().alias("total_layer_el"),
]).with_columns(
    (pl.col("total_layer_el") / pl.col("total_el")).alias("layer_rate")
).sort("cob")
print(summary)
```

```python
# Task 3: Top 10 risks
top10 = (
    portfolio
    .sort("layer_el", descending=True)
    .head(10)
    .select(["risk_id", "cob", "sum_insured", "expected_loss", "layer_el"])
)
print(top10)
```

Common characteristics: high sum insured (risks above the attachment), high expected loss (high frequency or severity), and often a lighter class of business (retail/Y1) where more expected loss sits in the upper severity range.

```python
# Task 4: Differentiated vs uniform curve
y2_uniform = swiss_re_curve(2.0)

layer_el_uniform = []
for row in portfolio.iter_rows(named=True):
    si = row["sum_insured"]
    el_u = layer_expected_loss(
        dist=y2_uniform,
        attachment=attachment,
        limit=limit,
        policy_limit=si,
        mpl=si,
        subject_premium=row["expected_loss"],
    )
    layer_el_uniform.append(el_u)

portfolio = portfolio.with_columns(pl.Series("layer_el_uniform", layer_el_uniform))

rate_differentiated = portfolio["layer_el"].sum() / portfolio["expected_loss"].sum()
rate_uniform        = portfolio["layer_el_uniform"].sum() / portfolio["expected_loss"].sum()

print(f"Differentiated rate (class curves): {rate_differentiated:.3%}")
print(f"Uniform rate (Y2 for all):          {rate_uniform:.3%}")
print(f"Difference:                         {(rate_differentiated - rate_uniform):+.3%}")
```

The direction of the difference depends on the mix: if retail (Y1) is the largest class, the differentiated rate will be higher than Y2-uniform (because Y1 is a softer curve that places more expected loss in the upper severity range). If the portfolio is dominated by heavy industrial (Y3), the differentiated rate will be lower.

</details>

---

## Exercise 10: Extension -- fitting with a fixed g

**Reference:** Tutorial Parts 5, 8; library source code

**What you will do:** Implement a constrained fitting routine that fixes the total loss probability (and therefore g) at a known value from claims data, and estimates only b using MLE. This arises when you trust the empirical total loss frequency but want to fit the partial loss distribution more carefully.

### Context

You are working with a commercial property book where the total loss frequency is known precisely from many years of data: 8.5% of risks that have claims experience a total loss. However, the partial loss data is sparse. You want to fix g = 1/0.085 ≈ 11.76 and estimate only b.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from insurance_ilf import MBBEFDDistribution, fit_mbbefd
from insurance_ilf.fitting import _log_density

rng = np.random.default_rng(seed=77)
TRUE_TOTAL_LOSS_PROB = 0.085
g_fixed = 1.0 / TRUE_TOTAL_LOSS_PROB   # ≈ 11.76
true_b  = 5.0                           # unknown, to be estimated

true_dist = MBBEFDDistribution(g=g_fixed, b=true_b)
N = 300
z_data = true_dist.rvs(N, rng=rng)
```

### Tasks

**Task 1.** Implement a single-parameter MLE that fixes g and optimises only over log(b) using `minimize_scalar`. The objective is the negative log-likelihood using `_log_density` from the library's fitting module. Recover b from the optimisation and compare to the true value (b = 5.0).

**Task 2.** Compare the constrained fit (fixed g) against the unconstrained MLE (free g and b). Which gives a better log-likelihood? Which gives a better AIC? Explain why the unconstrained MLE should always have an equal or better log-likelihood, but why the constrained fit might be preferred in practice.

**Task 3.** Compute the expected layer loss for a layer of £500k xs £250k on a risk with MPL = £2m and subject premium = £20,000 using: (a) the constrained fit, (b) the unconstrained fit, (c) the true distribution. Which fit gives the result closer to the true value?

**Task 4.** Plot the likelihood function for b over the range [0.5, 50] with g fixed at the true value (g = 1/0.085). Mark the MLE estimate and the true value b = 5.0. How flat is the likelihood surface? What does this say about the statistical precision of the estimate?

<details>
<summary>Solution -- Exercise 10</summary>

```python
# Task 1: Single-parameter MLE (fixed g)
def neg_loglik_fixed_g(log_b: float, z: np.ndarray, g: float) -> float:
    b = np.exp(np.clip(log_b, -10.0, 5.0))
    if abs(g * b - 1.0) < 1e-8:
        return 1e12
    try:
        contributions = _log_density(z, g, b)
        total = float(-np.sum(contributions))
        return total if np.isfinite(total) else 1e12
    except Exception:
        return 1e12

result_constrained = minimize_scalar(
    neg_loglik_fixed_g,
    bounds=(-10.0, 5.0),
    method="bounded",
    args=(z_data, g_fixed),
)

b_estimated = float(np.exp(result_constrained.x))
dist_constrained = MBBEFDDistribution(g=g_fixed, b=b_estimated)
loglik_constrained = float(-result_constrained.fun)

print(f"True b:           {true_b:.4f}")
print(f"Estimated b (MLE, fixed g): {b_estimated:.4f}")
print(f"Log-likelihood:   {loglik_constrained:.2f}")
print(f"AIC (k=1):        {-2 * loglik_constrained + 2 * 1:.2f}")
```

```python
# Task 2: Compare to unconstrained MLE
result_unc = fit_mbbefd(z_data)

print(f"\nUnconstrained MLE:")
print(f"  g = {result_unc.params['g']:.4f}  (true: {g_fixed:.4f})")
print(f"  b = {result_unc.params['b']:.4f}  (true: {true_b:.4f})")
print(f"  loglik = {result_unc.loglik:.2f}")
print(f"  AIC (k=2) = {result_unc.aic:.2f}")
print(f"\nConstrained MLE:")
print(f"  AIC (k=1) = {-2 * loglik_constrained + 2 * 1:.2f}")
```

The unconstrained MLE has a higher (better) log-likelihood because it has an additional degree of freedom -- it can also adjust g to fit the data. But if we genuinely know g from external data (many years of total loss experience), the constrained model is more parsimonious: it uses fewer parameters, so the AIC penalty is smaller (1 parameter vs 2). When the data constraint is valid, prefer the constrained fit.

```python
# Task 3: Layer pricing comparison
from insurance_ilf import layer_expected_loss

ap = 250_000; lim = 500_000; pl_val = mpl_val = 2_000_000; sp = 20_000

el_true   = layer_expected_loss(true_dist, ap, lim, pl_val, mpl_val, sp)
el_const  = layer_expected_loss(dist_constrained, ap, lim, pl_val, mpl_val, sp)
el_unc    = layer_expected_loss(result_unc.dist, ap, lim, pl_val, mpl_val, sp)

print(f"\nExpected layer loss:")
print(f"  True distribution:   £{el_true:,.0f}")
print(f"  Constrained fit:     £{el_const:,.0f}  ({(el_const-el_true)/el_true:+.2%})")
print(f"  Unconstrained fit:   £{el_unc:,.0f}  ({(el_unc-el_true)/el_true:+.2%})")
```

```python
# Task 4: Likelihood profile for b
import matplotlib.pyplot as plt

log_b_grid = np.linspace(-1.0, 4.0, 300)   # b from exp(-1) ≈ 0.37 to exp(4) ≈ 54.6
b_grid     = np.exp(log_b_grid)
ll_grid    = np.array([-neg_loglik_fixed_g(lb, z_data, g_fixed) for lb in log_b_grid])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(b_grid, ll_grid, "b-", linewidth=2)
ax.axvline(x=b_estimated, color="r", linestyle="--", label=f"MLE estimate b={b_estimated:.2f}")
ax.axvline(x=true_b,      color="g", linestyle=":",  label=f"True b={true_b:.2f}")
ax.set_xlabel("b parameter")
ax.set_ylabel("Log-likelihood")
ax.set_title("Log-likelihood profile for b (g fixed)")
ax.legend()
ax.set_xscale("log")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Flatness: find the region where LL is within 2 units of the maximum
ll_max = ll_grid.max()
credible_b = b_grid[ll_grid >= ll_max - 2.0]
print(f"\nApproximate 95% confidence interval for b: [{credible_b.min():.2f}, {credible_b.max():.2f}]")
print(f"(Based on LL drop of 2 units from maximum)")
```

The likelihood profile is typically flat and asymmetric for b: there is a long plateau on the right side (large b values give similar likelihoods). This reflects the fundamental difficulty of estimating the shape of the partial loss distribution from a limited number of claims. The width of the 95% interval tells you how precisely b is determined by your data. If the interval is wide, the choice between Y2 (b ≈ 9) and Y3 (b ≈ 2.7) is not statistically resolved by your data alone.

</details>

---

## Summary of what you have built

After completing these ten exercises, you can:

- Implement the MBBEFD exposure curve formula from scratch and verify it against the library
- Select Swiss Re standard curves by class of business and quantify the pricing error from wrong selection
- Fit MBBEFD distributions using MLE and method of moments, with correct handling of truncated and censored data
- Build and interpret ILF tables and use them to rate policies at multiple limit levels
- Price per-risk XL treaties from cedant risk profiles
- Use Lee diagrams to assess loss concentration and communicate it to underwriters
- Fit MBBEFD from empirical exposure curves (for when raw claims data is unavailable)
- Integrate exposure curve pricing with GLM/GBM expected loss predictions at the individual risk level
- Implement constrained MLE when external information pins one parameter

The next module extends this framework to catastrophe modelling and aggregate excess-of-loss structures.
