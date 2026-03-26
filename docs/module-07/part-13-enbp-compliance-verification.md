## Part 13: ENBP compliance verification

After solving, always verify ENBP compliance per-policy independently of the constraint. The solver enforces ENBP via multiplier upper bounds, but an independent check is belt-and-braces and is the compliance evidence for the FCA audit trail.

```python
import numpy as np

# ENBP check: for every renewal policy, new_premium <= enbp
new_premiums_arr = result.new_premiums
enbp_arr         = enbp
renewal_mask     = renewal_flag.astype(bool)

# Excess: how much each renewal premium exceeds its ENBP
enbp_excess  = new_premiums_arr[renewal_mask] - enbp_arr[renewal_mask]
violations   = enbp_excess > 0.01   # 1p tolerance for floating-point rounding

n_renewals   = renewal_mask.sum()
n_violations = violations.sum()

print("ENBP compliance verification:")
print(f"  Renewal policies checked: {n_renewals:,}")
print(f"  ENBP violations:          {n_violations}")

if n_violations == 0:
    print("  RESULT: All renewal premiums are at or below ENBP.")
    print("  ENBP constraint satisfied per-policy.")
else:
    print("  RESULT: ENBP violations detected.")
    print("  Do not proceed to sign-off. Investigate the enbp array.")
    # Show the worst violations
    excess_at_violations = enbp_excess[violations]
    top5 = np.sort(excess_at_violations)[::-1][:5]
    print(f"  Top 5 violation amounts (£): {[f'{x:.2f}' for x in top5]}")

print()
print("ENBP binding summary:")
binding_count = result.summary_df["enbp_binding"].sum()
print(f"  Policies at ENBP cap: {binding_count:,} ({binding_count/n_renewals:.1%} of renewals)")
print(f"  Max excess (should be <= 0): £{enbp_excess.max():.4f}")
```

**What you should see:**

```text
ENBP compliance verification:
  Renewal policies checked: 3,250
  ENBP violations:          0
  RESULT: All renewal premiums are at or below ENBP.
  ENBP constraint satisfied per-policy.

ENBP binding summary:
  Policies at ENBP cap: 412 (12.7% of renewals)
  Max excess (should be <= 0): £-0.0002
```

If you see violations, the most likely cause is that the `enbp` array passed to the optimiser was not correctly computed. The most common error in production is computing ENBP as the prior year's premium without applying the current year's NB pricing model — resulting in ENBP values that are lower than they should be.

### What "binding" means

`enbp_binding = True` for a policy means its optimal multiplier exactly hits the ENBP upper bound. The optimiser would have liked to charge this customer more (the profit-maximising multiplier without ENBP would be higher), but the regulatory constraint prevents it.

These are the policies where ENBP is most costly in commercial terms. If a large fraction of your renewals are at the ENBP cap, it suggests your renewal tariff is systematically above new business, which is the scenario PS21/11 is designed to prevent. The right remedy is not to loosen ENBP — it is to review why renewals are being priced above new business equivalents.

### Saving the compliance record

```python
# Save the per-policy ENBP check to a DataFrame for the audit trail
enbp_compliance = (
    df
    .filter(pl.col("renewal_flag"))
    .with_columns([
        pl.Series("new_premium",   new_premiums_arr[renewal_mask].tolist()),
        pl.Series("enbp",          enbp_arr[renewal_mask].tolist()),
        pl.Series("enbp_excess",   enbp_excess.tolist()),
        pl.Series("compliant",     (enbp_excess <= 0.01).tolist()),
    ])
    .select(["policy_id", "new_premium", "enbp", "enbp_excess", "compliant"])
)

print(f"ENBP compliance record: {len(enbp_compliance):,} renewal policies")
print(f"All compliant: {enbp_compliance['compliant'].all()}")
```

Write this DataFrame to Unity Catalog alongside the factor adjustments (Part 15). This is the per-policy compliance evidence the FCA expects under PS21/11.
