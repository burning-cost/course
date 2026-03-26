## Part 13: Application 3 - Portfolio reserve range estimates

Individual prediction intervals aggregate to portfolio-level range estimates. This section shows how, and is explicit about the assumptions that make the aggregation valid or invalid.

In a new cell:

```python
%md
## Part 13: Portfolio reserve ranges
```

```python
# Portfolio point estimate: sum of all individual predictions
portfolio_point = point.sum()

# --- Method 1: Naive (worst-case correlation) ---
# Assume all risks simultaneously hit their lower (or upper) bounds.
# This is the catastrophe scenario: all risks move together.
portfolio_lower_naive = lower.sum()
portfolio_upper_naive = upper.sum()

# --- Method 2: Independence (CLT approximation) ---
# Assume individual risks are independent. Sum the variances, take the square root.
# IMPORTANT WARNING (read before presenting to reserving team):
# - This uses a symmetric normal approximation to individual losses.
#   Tweedie losses are right-skewed with zero-capped lower bounds, so the
#   symmetric approximation OVERESTIMATES the portfolio SD: the lower bound is
#   floored at zero for many policies, but the symmetric formula treats it as
#   if the lower tail is as wide as the upper tail.
# - This assumes zero correlation across risks.
#   Systemic events (weather, economic shocks) violate this assumption.
# - The independence range is an optimistic lower bound, not a central estimate.
#
# For a 90% interval, width ≈ 2 × 1.645 × sd, so sd ≈ width / 3.29
approx_sd        = (upper - lower) / 3.29
portfolio_sd     = np.sqrt((approx_sd ** 2).sum())

portfolio_lower_indep = max(0, portfolio_point - 1.645 * portfolio_sd)
portfolio_upper_indep = portfolio_point + 1.645 * portfolio_sd

print("Portfolio reserve range estimates")
print("=" * 55)
print(f"Point estimate (sum of technical premiums): £{portfolio_point:,.0f}")
print()
print("90% Range (Naive - perfect correlation, worst case):")
print(f"  Lower: £{portfolio_lower_naive:,.0f}")
print(f"  Upper: £{portfolio_upper_naive:,.0f}")
print()
print("90% Range (Independence - CLT approximation, optimistic):")
print(f"  Lower: £{portfolio_lower_indep:,.0f}")
print(f"  Upper: £{portfolio_upper_indep:,.0f}")
print()
diversification_benefit = portfolio_upper_naive / portfolio_upper_indep
print(f"Diversification benefit (naive / independence upper): {diversification_benefit:.1f}x")
print()
print("NOTE: True portfolio range lies between these bounds.")
print("Naive bound: relevant for catastrophe scenarios (correlated weather, economic shocks).")
print("Independence bound: relevant for idiosyncratic risk (individual accidents, theft).")
print("Add a catastrophe overlay separately for systemic events.")
```

**What you should see:** the naive upper bound is much larger than the independence upper bound. For a portfolio of 20,000 test policies, the diversification benefit is typically 3-10x - the independence bound is far lower than the naive bound because portfolio diversification reduces the aggregate uncertainty when risks are uncorrelated.

**How to present this to the reserving team:**

Present both bounds explicitly with their assumptions:

- "The independence bound (£X) is the expected 90th percentile reserve if all claims events are independent. This is appropriate for standard idiosyncratic risk."
- "The naive bound (£Y) is the expected 90th percentile reserve if all risks are simultaneously adversely affected. This is the catastrophe scenario."
- "The true range lies between these. For storm or flood events, the relevant bound is closer to the naive. For everyday claims events, it is closer to the independence."

```python
# Segmented reserve range by area: useful for the reinsurance conversation
area_col    = X_test.reset_index(drop=True)["area"]
seg_results = {}

for area_val in sorted(area_col.unique()):
    mask = area_col.values == area_val
    seg_results[area_val] = {
        "n_risks":     mask.sum(),
        "total_point": point[mask].sum(),
        "total_lower": lower[mask].sum(),
        "total_upper": upper[mask].sum(),
    }

print(f"\n{'Area':<6} {'Risks':>7} {'Point £':>12} {'Lower £':>12} {'Upper £':>12} {'Width ratio':>13}")
print("-" * 65)
for area_val, seg in seg_results.items():
    ratio = seg["total_upper"] / max(seg["total_lower"], 1)
    print(f"{area_val:<6} {seg['n_risks']:>7,} {seg['total_point']:>12,.0f} "
          f"{seg['total_lower']:>12,.0f} {seg['total_upper']:>12,.0f} {ratio:>13.2f}x")
```

Areas with a high upper/lower ratio have the most reserve uncertainty. These are candidates for area-specific stop-loss reinsurance cover.