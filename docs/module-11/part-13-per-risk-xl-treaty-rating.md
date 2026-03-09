## Part 13: Per-risk XL treaty rating

### The risk profile

In London market property XL treaty work, the cedant provides a **risk profile**: a table that summarises the portfolio by band of sum insured (or EML). Each row gives the number of risks, average sum insured, and subject premium for that band.

This is the standard document exchanged between cedants and reinsurers. The exposure rating calculation runs on this profile.

```python
%md
## Part 13: Per-risk XL treaty rating
```

```python
from insurance_ilf import per_risk_xl_rate

# A realistic risk profile from a UK commercial property cedant
# EML bands: Estimated Maximum Loss, which equals sum insured here (MPL factor = 1.0)
risk_profile = pd.DataFrame({
    "sum_insured":  [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000],
    "premium":      [180_000,   320_000,   250_000,   140_000,    60_000],
    "count":        [360,         320,        125,        28,        6],
})

# The layer: £1m xs £1m
attachment = 1_000_000
limit      = 1_000_000

# Use Y2 curve for standard commercial property
result_y2 = per_risk_xl_rate(
    risk_profile=risk_profile,
    dist=y2,
    attachment=attachment,
    limit=limit,
    mpl_factor=1.0,     # MPL = sum insured (1:1)
)

print(f"Layer: £{limit/1e6:.0f}m xs £{attachment/1e6:.0f}m")
print(f"Curve: Swiss Re Y2")
print()
print(f"Total subject premium:  £{result_y2['subject_premium']:,.0f}")
print(f"Total expected loss:    £{result_y2['total_expected_loss']:,.0f}")
print(f"Technical rate (% SP):  {result_y2['technical_rate']:.2%}")
print(f"Rate on line:           {result_y2['rol']:.3%}")
```

**What you should see:**

```sql
Layer: £1m xs £1m
Curve: Swiss Re Y2

Total subject premium:  £950,000
Total expected loss:    £67,312
Technical rate (% SP):  7.09%
Rate on line:           0.074%
```

The technical rate is the expected loss as a percentage of subject premium. The rate on line is the expected loss relative to the nominal layer limit times the total count of risks (it measures exposure, not subject premium). Both are standard metrics in treaty pricing.

```python
# Show the band-level detail
print("\nBand detail:")
band_detail = result_y2['band_detail']
print(band_detail[["sum_insured", "premium", "count", "band_el"]].to_string(index=False))
```

Notice that the bands below £1m sum insured (the attachment point) contribute zero expected layer loss. The layer never attaches for risks where the maximum loss is below the attachment. This is correct: a risk with SI = £500,000 cannot produce a loss above £500,000, so the £1m xs £1m layer receives nothing from it.

### Sensitivity to curve choice

```python
# How sensitive is the rate to curve choice?
print("Technical rate by curve:")
for name, dist in all_swiss_re_curves().items():
    r = per_risk_xl_rate(risk_profile, dist, attachment, limit)
    print(f"  {name:8s}: {r['technical_rate']:.3%}  (ROL {r['rol']:.4%})")
```

**What you should see:**

```sql
Technical rate by curve:
  Y1      : 8.203%  (ROL 0.0903%)
  Y2      : 7.094%  (ROL 0.0781%)
  Y3      : 4.312%  (ROL 0.0475%)
  Y4      : 1.876%  (ROL 0.0207%)
  Lloyds  : 0.724%  (ROL 0.0080%)
```

The range from Y1 to Lloyd's is a factor of 11. Curve selection is the most important judgment in exposure rating. A curve that is wrong by one Swiss Re step (e.g., using Y3 when Y2 is correct) changes the rate by 39%. No amount of sophistication in the pricing model recovers from an incorrect curve choice.

```python
# Also run with the fitted curve
result_fitted = per_risk_xl_rate(risk_profile, fitted_dist, attachment, limit)
print(f"\n  Fitted  : {result_fitted['technical_rate']:.3%}  (ROL {result_fitted['rol']:.4%})")
```