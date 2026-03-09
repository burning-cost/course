## Part 12: Application 2 - Risk-specific minimum premium floors

### The problem with flat multipliers

Your current minimum premium policy probably reads something like: "Minimum premium = 1.3x technical premium, subject to a floor of £250."

The 1.3x multiplier has no principled basis in the distribution of losses. It is a rule of thumb chosen based on historical reserve adequacy experience. Applied uniformly, it:

- Overcharges stable, well-understood risks (their 95th percentile loss is well below 1.3x the technical premium)
- Undercharges volatile, uncertain risks (their 95th percentile loss may be 2x or 3x the technical premium)

Both create problems. Overcharging stable customers is a Consumer Duty fair value issue. Undercharging volatile risks understates the reserve requirement and creates solvency adequacy risk.

### Using the conformal upper bound as a floor

The 95% conformal upper bound is the loss level we expect to be exceeded only 5% of the time. Using it as a minimum premium floor means: "we price this risk so that it will be unprofitable for us no more than 5% of the time, as measured on the calibration data."

That is a principled, auditable justification. Not "we apply 1.3x because that is what we have always done," but "we apply a floor calibrated to the 95th percentile of predicted losses, validated on historical business, with documented coverage of 95% across all risk deciles."

In a new cell:

```python
%md
## Part 12: Minimum premium floors
```

```python
# Extract upper bounds at different alpha levels
upper_90 = intervals_90["upper"].to_numpy()   # 90% upper bound (10% exceedance)
upper_95 = intervals_95["upper"].to_numpy()   # 95% upper bound (5% exceedance)
upper_80 = intervals_80["upper"].to_numpy()   # 80% upper bound (20% exceedance)

# Three floor approaches
floor_conventional = np.maximum(1.3 * point, 250)    # current policy
floor_conformal_95 = upper_95                          # principled 95% upper bound
floor_practical    = np.maximum(1.5 * point, upper_80) # hybrid: 1.5x vs 80% upper

print(f"{'Approach':<35} {'Median':>10} {'Mean':>10} {'95th pctile':>14}")
print("-" * 72)
for label, floor in [
    ("Conventional (1.3x, floor £250)",     floor_conventional),
    ("Conformal 95% upper bound",           floor_conformal_95),
    ("Practical (1.5x vs 80% upper)",       floor_practical),
]:
    print(f"{label:<35} {np.median(floor):>10.2f} {np.mean(floor):>10.2f} {np.quantile(floor, 0.95):>14.2f}")
```

**What you should see:** the three approaches produce similar medians but different tails. The conformal floor will be higher than the conventional floor for volatile risks (wide intervals) and lower for stable risks (narrow intervals).

Now find which risks the two approaches disagree on:

```python
# Where conformal floor is HIGHER than conventional: conventional undercharges these risks
higher_than_conventional = floor_conformal_95 > floor_conventional
print(f"\nRisks where conformal floor > conventional floor: {higher_than_conventional.sum():,} ({higher_than_conventional.mean():.1%})")
if higher_than_conventional.any():
    sub = X_test_pl.filter(pl.Series(higher_than_conventional))
    print(f"  Their profile: mean age {sub['driver_age'].mean():.1f}, "
          f"mean vehicle group {sub['vehicle_group'].mean():.1f}, "
          f"{(sub['conviction_points'] > 0).mean() * 100:.1f}% with convictions")

# Where conformal floor is LOWER than conventional: conventional overcharges these risks
lower_than_conventional = floor_conformal_95 < floor_conventional
print(f"\nRisks where conformal floor < conventional floor: {lower_than_conventional.sum():,} ({lower_than_conventional.mean():.1%})")
if lower_than_conventional.any():
    sub_lo = X_test_pl.filter(pl.Series(lower_than_conventional))
    print(f"  Their profile: mean age {sub_lo['driver_age'].mean():.1f}, "
          f"mean vehicle group {sub_lo['vehicle_group'].mean():.1f}, "
          f"mean NCD {sub_lo['ncd_years'].mean():.1f} years")
```

**Interpreting the results:**

- Risks where the conformal floor is higher: these are volatile risks (wide intervals) where the conventional 1.3x multiplier does not cover the genuine uncertainty. The conformal floor is the correct answer here - it reflects the actual 95th percentile of losses for that risk profile.

- Risks where the conformal floor is lower: these are stable risks (narrow intervals) where the conventional floor is excessive. On Consumer Duty grounds, the conformal floor is more defensible: you are not applying an arbitrary multiplier to a well-understood risk.

The FCA evidence: the coverage-by-decile diagnostic shows that the 95% intervals achieve 95% coverage across all risk deciles. This is the mathematical basis for the floor - not a rule of thumb, but a calibrated threshold validated on recent business.