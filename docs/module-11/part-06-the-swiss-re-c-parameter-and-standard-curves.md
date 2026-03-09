## Part 6: The Swiss Re c-parameter and standard curves

### From two parameters to one

Fitting a two-parameter family means searching a 2D surface for the maximum likelihood. But for practical use -- when you want to choose a curve by class of business judgment, not by fitting -- you want a single number that you can dial up or down.

Swiss Re solved this by defining a one-dimensional path through the (g, b) space parametrised by c:

```sql
b = exp(3.1 - 0.15 * c * (1 + c))
g = exp(c * (0.78 + 0.12 * c))
```

These formulas come directly from Bernegger (1997). They are not derived from first principles -- they are calibrated to make the four standard curves (Y1-Y4) correspond to c = 1.5, 2.0, 3.0, 4.0.

The c-parameter has an intuitive direction: higher c means heavier, more industrial risks with lower total loss probability and more loss concentrated at the low end of the severity scale. As c increases from 1.5 to 5.0, the exposure curve flattens: G(0.5) falls, meaning less of the expected loss is captured below the midpoint of MPL.

### The five standard curves

```python
from insurance_ilf.curves import all_swiss_re_curves
import numpy as np

curves = all_swiss_re_curves()

print(f"{'Curve':<8} {'c':>4} {'g':>8} {'b':>8} {'P(total)':>10} {'G(0.25)':>9} {'G(0.50)':>9} {'G(0.75)':>9}")
print("-" * 70)

c_values = {"Y1": 1.5, "Y2": 2.0, "Y3": 3.0, "Y4": 4.0, "Lloyds": 5.0}
for name, dist in curves.items():
    c = c_values[name]
    print(
        f"{name:<8} {c:>4.1f} {dist.g:>8.2f} {dist.b:>8.4f} "
        f"{dist.total_loss_prob():>10.1%} "
        f"{float(dist.exposure_curve(0.25)):>9.4f} "
        f"{float(dist.exposure_curve(0.50)):>9.4f} "
        f"{float(dist.exposure_curve(0.75)):>9.4f}"
    )
```

**What you should see:**

```python
Curve       c        g       b   P(total)   G(0.25)   G(0.50)   G(0.75)
----------------------------------------------------------------------
Y1      1.5     4.22  22.1998      23.7%    0.7027    0.8560    0.9417
Y2      2.0     7.69   9.0196      13.0%    0.5817    0.7394    0.8681
Y3      3.0    24.53   2.7456       4.1%    0.3961    0.5660    0.7470
Y4      4.0    78.67   0.8825       1.3%    0.2536    0.4136    0.6215
Lloyds  5.0   251.65   0.3033       0.4%    0.1614    0.2961    0.5036
```

Read this table as follows. Y1 is the "softest" curve: 23.7% of claims are total losses, and already 85.6% of expected loss is captured at the halfway point of MPL. Lloyd's is the "hardest" curve: only 0.4% of claims are total losses, and at the halfway point of MPL only 29.6% of expected loss has been captured. The Lloyd's curve says: most of the loss is concentrated in the upper part of the severity range, because partial losses on industrial complexes tend to be large.

### When to use which curve

These guidelines follow London market practice and Bernegger's original paper:

- **Y1 (c=1.5)**: light manufacturing, sprinkler-protected commercial premises, retail with good fire separation. Highest total loss probability because sprinklers often fail to prevent total loss once a serious fire starts.
- **Y2 (c=2.0)**: standard commercial property, warehousing, mixed-use commercial. The default for UK commercial lines without specific engineering information.
- **Y3 (c=3.0)**: heavy commercial, large warehousing, construction. Risks where partial losses dominate because the structure is robust enough to survive most events.
- **Y4 (c=4.0)**: high-value industrial, petrochemical, power generation. Very low total loss probability; when these risks burn, they produce large partials rather than total losses.
- **Lloyd's (c=5.0)**: industrial complexes, special risks, offshore. Used at Lloyd's as a default for risks where Y4 is still too optimistic.

The choice of curve is a judgment call. In practice:
1. If you have 50+ large losses from the class, fit the curve from data (Part 8)
2. If you have fewer losses, choose the nearest standard curve by class description
3. For a mixed portfolio, fit a single curve across all classes and accept the averaging