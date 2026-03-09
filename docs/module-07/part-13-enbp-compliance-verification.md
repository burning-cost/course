## Part 13: ENBP compliance verification

After solving, always verify ENBP compliance per-policy rather than relying solely on the constraint having been active during the solve. This is a belt-and-braces check that catches implementation errors.

```python
# Compute adjusted premium and NB equivalent for each renewal policy
# Use Polars for the computation, then extract numpy arrays for the check

renewal_flag_np = df["renewal_flag"].to_numpy()

# factor_adj: the solved multipliers from Part 9
factor_adj  = result.factor_adjustments

adj_premium = df["current_premium"].to_numpy().copy()
nb_equiv    = df["current_premium"].to_numpy().copy()

for fname in FACTOR_NAMES:
    m = factor_adj.get(fname, 1.0)
    adj_premium = adj_premium * m
    if fname not in fs.renewal_factor_names:
        nb_equiv = nb_equiv * m

# Check: for every renewal policy, adjusted_renewal <= NB_equivalent
# Allow 1p tolerance for floating-point rounding
violations = (adj_premium[renewal_flag_np] > nb_equiv[renewal_flag_np] + 0.01)
n_renewals = renewal_flag_np.sum()

print("ENBP compliance verification:")
print(f"  Renewal policies checked: {n_renewals:,}")
print(f"  ENBP violations:          {violations.sum()}")

if violations.sum() == 0:
    print("  RESULT: All renewal premiums are at or below the NB equivalent.")
    print("  ENBP constraint satisfied per-policy.")
else:
    print("  RESULT: ENBP violations detected.")
    print("  Do not proceed to sign-off. Investigate the factor classification.")
    # Show the worst violations
    excess = adj_premium[renewal_flag_np] - nb_equiv[renewal_flag_np]
    top5 = sorted(excess[violations], reverse=True)[:5]
    print(f"  Top 5 violation amounts (£): {[f'{x:.2f}' for x in top5]}")
```

**What you should see:**

```python
ENBP compliance verification:
  Renewal policies checked: 3,250
  ENBP violations:          0
  RESULT: All renewal premiums are at or below the NB equivalent.
  ENBP constraint satisfied per-policy.
```

If you see violations, the most likely cause is that the tenure discount factor was accidentally included in the shared factors (not in `renewal_factor_names`), or that the optimiser found a way to increase the tenure discount that the ENBP constraint did not catch. Do not proceed to sign-off until this check passes.

The per-policy ENBP check is the compliance evidence for the FCA. Keep this output in the notebook and export it to a Unity Catalog table alongside the factor adjustments.