## Part 12: Excess loss factors

The excess loss factor (ELF) at an attachment point AP is the complement of the exposure curve:

```sql
ELF(AP) = 1 - G(AP/MPL)
```

It is the proportion of expected loss that sits above the attachment point. For XL pricing, it answers: "what fraction of my book's expected losses does the reinsurer expect to see?"

```python
%md
## Part 12: Excess loss factors and per-risk XL layer pricing
```

```python
from insurance_ilf import excess_loss_factor, layer_expected_loss

# ELF at various attachment points for Y2, MPL = £2m
mpl = 2_000_000
y2  = swiss_re_curve(2.0)

print("Excess Loss Factors -- Y2, MPL = £2,000,000:")
for ap in [250_000, 500_000, 750_000, 1_000_000, 1_500_000]:
    elf = excess_loss_factor(dist=y2, attachment=ap, mpl=mpl)
    print(f"  ELF({ap/1e6:.2f}m) = {elf:.4f}  ({elf:.1%} of expected loss above this point)")
```

**Output:**

```python
Excess Loss Factors -- Y2, MPL = £2,000,000:
  ELF(0.25m) = 0.4183  (41.8% of expected loss above this point)
  ELF(0.50m) = 0.2606  (26.1% of expected loss above this point)
  ELF(0.75m) = 0.1319  (13.2% of expected loss above this point)
  ELF(1.00m) = 0.0476  (4.8% of expected loss above this point)
  ELF(1.50m) = 0.0035  (0.4% of expected loss above this point)
```

An attachment of £500,000 on a Y2 risk with £2m MPL absorbs 74% of expected loss below the attachment. Only 26% of expected loss is in the layer above £500k. This makes intuitive sense: most losses are small partial losses.

### Single-risk layer pricing

```python
# Price a specific per-risk XL layer for a single risk
mpl            = 2_000_000
policy_limit   = 2_000_000
attachment     = 500_000
limit          = 1_000_000
subject_premium = 25_000     # expected total loss from this risk

el = layer_expected_loss(
    dist=y2,
    attachment=attachment,
    limit=limit,
    policy_limit=policy_limit,
    mpl=mpl,
    subject_premium=subject_premium,
)

print(f"Layer: £{limit/1e6:.1f}m xs £{attachment/1e6:.1f}m")
print(f"MPL: £{mpl/1e6:.1f}m,  Policy limit: £{policy_limit/1e6:.1f}m")
print(f"Subject premium: £{subject_premium:,}")
print(f"Expected layer loss: £{el:,.0f}")
print(f"Technical rate (% of SP): {el/subject_premium:.2%}")
print(f"Rate on line: {el/limit:.3%}")
# Note: we assume MPL = sum insured here. For industrial risks (construction yards,
# petrochemical plants), the maximum possible loss from a single event is often less
# than the full sum insured -- the MPL factor may be 0.6-0.8. Adjust mpl_factor
# accordingly when using per_risk_xl_rate() for industrial portfolios.
```

### The policy limit correction

The `min(PL, ...)` terms in the Clark (2014) formula are not just algebraic tidying. They matter when the policy limit is below the layer ceiling.

```python
# Demonstrate the policy limit correction
# Same layer, but now the policy limit is only £1.2m
# The reinsurer can never recover more than 1.2m - 0.5m = 700k per risk,
# not the full £1m layer width.

policy_limit_restricted = 1_200_000

el_full  = layer_expected_loss(y2, attachment, limit, policy_limit, mpl, subject_premium)
el_restricted = layer_expected_loss(y2, attachment, limit, policy_limit_restricted, mpl, subject_premium)

print(f"PL = £{policy_limit/1e6:.1f}m:  expected layer loss = £{el_full:,.0f}")
print(f"PL = £{policy_limit_restricted/1e6:.1f}m:  expected layer loss = £{el_restricted:,.0f}")
print(f"Reduction: {(el_full - el_restricted)/el_full:.1%}")

# The formula:
# PL=2.0m: upper = min(2.0m, 1.5m)/2.0m = 0.75, lower = min(2.0m, 0.5m)/2.0m = 0.25
# PL=1.2m: upper = min(1.2m, 1.5m)/2.0m = 0.60, lower = min(1.2m, 0.5m)/2.0m = 0.25
# The upper limit is capped at 0.60 instead of 0.75
print()
print("Formula check:")
g_upper_full = y2.exposure_curve(min(policy_limit, attachment + limit) / mpl)
g_upper_rest = y2.exposure_curve(min(policy_limit_restricted, attachment + limit) / mpl)
g_lower      = y2.exposure_curve(min(policy_limit, attachment) / mpl)
print(f"PL=2.0m: G(upper)={g_upper_full:.4f} - G(lower)={g_lower:.4f} = {g_upper_full - g_lower:.4f}")
print(f"PL=1.2m: G(upper)={g_upper_rest:.4f} - G(lower)={g_lower:.4f} = {g_upper_rest - g_lower:.4f}")
```

Always use the policy limit in the formula. Ignoring it overprices reinsurance on risks where the policy limit sits below the layer ceiling.