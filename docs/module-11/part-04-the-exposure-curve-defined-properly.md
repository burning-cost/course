## Part 4: The exposure curve, defined properly

### The formal definition

Let Z be the destruction rate random variable. The **exposure curve** is:

```sql
G(x) = E[min(Z, x)] / E[Z]
```

That is: the ratio of the limited expected value of Z (truncated at x) to the unrestricted expected value of Z. The curve is defined for x in [0, 1].

Three properties follow directly from the definition:

- **G(0) = 0**: if you limit losses at zero fraction of MPL, you recover nothing
- **G(1) = 1**: if you limit losses at 100% of MPL, you recover everything
- **G is concave**: the marginal contribution of each additional fraction of MPL decreases as x increases, because the density of losses thins out at high loss levels

The concavity property is critical. It means the exposure curve always lies above the 45-degree line (the curve for a distribution that places all its mass at a single point equal to the mean). The more spread the loss distribution, the more the curve bows toward the upper left.

### What the exposure curve tells you about layers

For a per-risk XL layer from attachment AP to AP + L, the fraction of expected loss that falls in the layer is:

```sql
layer fraction = G(AP + L / MPL) - G(AP / MPL)
```

Here we are assuming the policy limit equals MPL for now. We add the policy limit correction in Part 10.

This formula says: the layer captures the losses between two horizontal cuts on the x-axis of the exposure curve. The area under G(x) between AP/MPL and (AP+L)/MPL is the proportion of expected loss in the layer.

Multiply by the subject premium (expected total loss from the risk) to get the expected layer loss:

```sql
expected layer loss = subject_premium * [G((AP+L)/MPL) - G(AP/MPL)]
```

This is the core formula. Everything else in exposure rating is bookkeeping around this formula.

### A numerical example

Suppose we have a risk with:
- MPL = £2,000,000
- Subject premium = £40,000 (expected total loss)
- Layer: £500,000 excess of £500,000 (i.e., AP = £500k, L = £500k, ceiling = £1m)
- Curve: Swiss Re Y2

```python
from insurance_ilf import swiss_re_curve

y2 = swiss_re_curve(2.0)

mpl = 2_000_000
ap = 500_000
limit = 500_000
subject_premium = 40_000

# Layer fraction
g_upper = y2.exposure_curve(min(ap + limit, mpl) / mpl)
g_lower = y2.exposure_curve(ap / mpl)
layer_fraction = g_upper - g_lower

print(f"G(AP/MPL) = G({ap/mpl:.3f})        = {g_lower:.4f}")
print(f"G((AP+L)/MPL) = G({(ap+limit)/mpl:.3f})   = {g_upper:.4f}")
print(f"Layer fraction                         = {layer_fraction:.4f}")
print(f"Expected layer loss                    = £{layer_fraction * subject_premium:,.0f}")
```

**Output:**

```python
G(AP/MPL) = G(0.250)        = 0.5817
G((AP+L)/MPL) = G(0.500)   = 0.7394
Layer fraction                         = 0.1577
Expected layer loss                    = £6,307
```

The layer captures 15.77% of expected loss. On £40,000 subject premium, the expected layer loss is £6,307. The technical rate is £6,307 / £40,000 = 15.77%. That is the rate as a percentage of subject premium. The rate on line is £6,307 / £500,000 = 1.26%.