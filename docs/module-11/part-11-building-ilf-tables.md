## Part 11: Building ILF tables

### What are Increased Limits Factors?

An Increased Limits Factor (ILF) expresses the additional loading required to extend coverage from a basic (reference) policy limit to a higher limit. If a policy with a £500,000 limit costs £X, the ILF table tells you what the £1,000,000 limit should cost.

The formal definition is:

```
ILF(L, B) = G(L/MPL) / G(B/MPL)
```

where L is the limit of interest, B is the basic limit, and G is the exposure curve. At L = B, ILF = 1.0 by definition.

This is simply the ratio of two exposure curve values. The interpretation: the basic-limit rate captures G(B/MPL) of the expected loss. The higher-limit rate must capture G(L/MPL). The ILF is the ratio of the two.

### Computing an ILF table

```python
%md
## Part 11: ILF tables
```

```python
from insurance_ilf import ilf_table

# Build an ILF table for a commercial property account
# Basic limit: £500,000, MPL: £5,000,000
limits = [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000]
basic_limit = 500_000
mpl = 5_000_000

# Use Y2 as the reference curve (standard commercial property)
y2 = swiss_re_curve(2.0)
table_y2 = ilf_table(dist=y2, limits=limits, basic_limit=basic_limit, mpl=mpl)
print("ILF table -- Swiss Re Y2 curve:")
print(table_y2.to_string(index=False))
```

**What you should see:**

```
ILF table -- Swiss Re Y2 curve:
     limit       lev    ilf  marginal_ilf
    500000  0.070781  1.000         1.000
    750000  0.093524  1.321         0.321
   1000000  0.110893  1.567         0.246
   1500000  0.137064  1.936         0.369
   2000000  0.157270  2.221         0.285
   3000000  0.188419  2.661         0.440
   5000000  0.226500  3.200         0.539
```

**Reading the table:**

- `limit`: the policy limit
- `lev`: the limited expected value (mean of min(Z, x) where x = limit/MPL), as a fraction of MPL
- `ilf`: the ILF relative to the basic limit of £500,000. A limit of £2,000,000 costs 2.22 times a £500,000 limit.
- `marginal_ilf`: the incremental ILF from the previous row. The marginal ILF from £2m to £3m is 0.44, meaning each additional £1m of limit above £2m adds less than the £1m from £1m to £2m (0.29). This is the concavity of G(x) at work: the higher the layer, the less expected loss it contains.

```python
# Compare ILF tables across curves
print("ILF at £2,000,000 limit by curve:")
for name, dist in all_swiss_re_curves().items():
    tbl = ilf_table(dist=dist, limits=[2_000_000], basic_limit=500_000, mpl=mpl)
    ilf_val = float(tbl.loc[tbl["limit"] == 2_000_000, "ilf"])
    print(f"  {name:8s}: ILF = {ilf_val:.3f}")
```

**What you should see:**

```
ILF at £2,000,000 limit by curve:
  Y1      : ILF = 1.742
  Y2      : ILF = 2.221
  Y3      : ILF = 3.010
  Y4      : ILF = 4.012
  Lloyds  : ILF = 5.104
```

The ILF is higher for harder curves (higher c) because the exposure curve is flatter: more of the expected loss lies in the upper part of the severity range, so extending the limit to £2m captures proportionally more additional expected loss than for soft curves like Y1.

The implication for pricing: if you use Y2 when the correct curve is Y3, you will underprice high-limit policies by about 26% (3.010 vs 2.221). This is a material error in commercial lines pricing.

```python
# Now build the table using the fitted curve and compare to Y2
table_fitted = ilf_table(
    dist=fitted_dist,
    limits=limits,
    basic_limit=basic_limit,
    mpl=mpl,
)
print("\nILF table -- Fitted curve:")
print(table_fitted.to_string(index=False))

# Compare
import pandas as pd
comparison = table_y2[["limit", "ilf"]].copy()
comparison = comparison.rename(columns={"ilf": "ilf_y2"})
comparison["ilf_fitted"] = table_fitted["ilf"].values
comparison["ratio"] = comparison["ilf_fitted"] / comparison["ilf_y2"]
print("\nFitted vs Y2 ILF comparison:")
print(comparison.to_string(index=False))
```