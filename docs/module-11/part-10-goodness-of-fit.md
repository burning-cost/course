## Part 10: Goodness of fit

After fitting, you must check whether the MBBEFD family is an adequate fit for your data. Three tools are available.

```python
%md
## Part 10: Goodness of fit
```

```python
from insurance_ilf import GoodnessOfFit
from insurance_ilf.curves import empirical_exposure_curve
from insurance_ilf import compare_curves

# Use the result from the unconstrained fit (Part 8)
gof = GoodnessOfFit(z_vals, fitted_dist)

# KS test (continuous part only)
ks = gof.ks_test()
print(f"KS test:  statistic = {ks['statistic']:.4f},  p-value = {ks['p_value']:.4f}")

# Anderson-Darling (continuous part only)
ad = gof.ad_test()
print(f"AD test:  statistic = {ad['statistic']:.4f}  (critical 5%: 2.492)")

print()
if ks['p_value'] > 0.05:
    print("KS: no evidence to reject MBBEFD fit at 5%")
else:
    print(f"KS: marginal or failed fit (p={ks['p_value']:.3f})")
```

**Interpreting these tests:**

The KS test compares the empirical CDF to the theoretical CDF on the continuous part of the data (z < 1). The total-loss atom is excluded because the KS test requires a continuous CDF. A p-value above 0.05 says there is no significant evidence the data is not MBBEFD. With 600 observations, the test has reasonable power to detect misfit.

The Anderson-Darling statistic gives more weight to the tails. A value below 2.492 is a pass at 5%. For insurance data, AD > 5 usually means the tail behaviour is substantially different from MBBEFD.

Neither test is the end of the story. The diagnostic plots matter more.

```python
# Diagnostic plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. QQ plot
gof.qq_plot(ax=axes[0])

# 2. PP plot
gof.pp_plot(ax=axes[1])

# 3. Exposure curve: fitted vs empirical
ec_empirical = empirical_exposure_curve(
    losses=claims_df["loss_amount"].to_numpy(),
    mpl=claims_df["mpl"].to_numpy(),
    n_points=50,
)
gof.exposure_curve_plot(ax=axes[2], empirical_ec=ec_empirical)

plt.suptitle(f"Goodness of Fit: MBBEFD(g={fitted_dist.g:.2f}, b={fitted_dist.b:.2f})")
plt.tight_layout()
plt.show()
```

**Reading the plots:**

The QQ plot compares observed quantiles (y-axis) against theoretical quantiles (x-axis). Points close to the 45-degree line indicate good fit. Points above the line in the upper tail say the data has heavier tails than the fitted distribution. Points below the line say the fitted distribution has heavier tails than the data.

The PP plot compares empirical probabilities against theoretical probabilities. This plot is more sensitive to the bulk of the distribution, less sensitive to extremes.

The exposure curve comparison overlays the fitted parametric G(x) against the non-parametric empirical estimate. The empirical curve is computed from observed destruction rates: for each x, it is the proportion of expected loss below x estimated from the data. A good fit shows the smooth fitted curve running through the scatter of empirical points.

```python
# Compare the fitted curve against the five Swiss Re standards
curves = all_swiss_re_curves()
all_dists = list(curves.values()) + [fitted_dist]
all_labels = list(curves.keys()) + [f"Fitted (g={fitted_dist.g:.1f}, b={fitted_dist.b:.1f})"]

fig, ax = plt.subplots(figsize=(9, 6))
compare_curves(dists=all_dists, labels=all_labels, empirical=ec_empirical, ax=ax)
ax.set_title("Fitted curve vs Swiss Re standards")
plt.tight_layout()
plt.show()
```

This comparison is the most useful picture for communicating the fit to an underwriter. Show them where your fitted curve sits relative to the standard curves, and use that to justify (or question) the choice of curve for the layer pricing.