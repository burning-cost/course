## Part 7: Exploring the standard curves in code

Before fitting anything, let us explore what the standard curves look like and confirm we understand the formulas.

Add a markdown cell:

```python
%md
## Part 7: Exploring the standard curves
```

Then paste and run this:

```python
# ---- Part 7: Swiss Re standard curves ----

from insurance_ilf import swiss_re_curve, MBBEFDDistribution
from insurance_ilf.curves import all_swiss_re_curves
import numpy as np
import matplotlib.pyplot as plt

curves = all_swiss_re_curves()
x_grid = np.linspace(0.0, 1.0, 500)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: exposure curves
ax = axes[0]
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
labels  = ["Y1 (c=1.5, light mfg)", "Y2 (c=2.0, standard commercial)",
           "Y3 (c=3.0, heavy commercial)", "Y4 (c=4.0, high-value industrial)",
           "Lloyd's (c=5.0, complexes)"]

for (name, dist), colour, label in zip(curves.items(), colours, labels):
    g_vals = dist.exposure_curve(x_grid)
    ax.plot(x_grid, g_vals, color=colour, label=label, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Diagonal (reference)")
ax.set_xlabel("Fraction of MPL  (x)")
ax.set_ylabel("G(x)")
ax.set_title("Swiss Re Standard Exposure Curves")
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# Right: PDF for the continuous part (x < 1)
ax = axes[1]
x_cont = np.linspace(0.001, 0.999, 500)
for (name, dist), colour, label in zip(curves.items(), colours, labels):
    pdf_vals = dist.pdf(x_cont)
    ax.plot(x_cont, pdf_vals, color=colour, label=label, linewidth=2)

ax.set_xlabel("Fraction of MPL  (x)")
ax.set_ylabel("f(x)")
ax.set_title("Partial Loss Density (continuous part)")
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

Look at the two panels carefully.

**Left panel (exposure curves):** Y1 bows furthest toward the upper-left corner. This says: at any fraction of MPL, Y1 captures the largest proportion of expected loss. The curve is "rich" at the low end because Y1 has many total losses (z = 1.0) and many small partial losses, both of which contribute to G(x) being high for small x. Lloyd's bows farthest toward the lower-right: losses are distributed toward the upper end of the severity range.

**Right panel (partial loss density):** Y1 shows very high density near x = 0 (many small partial losses) falling steeply. Lloyd's shows a flatter, fatter-tailed density that spreads further toward x = 1. The total loss probability (the point mass at z = 1) is NOT shown in this panel -- it is the discrete atom discussed in Part 5.

```python
# Confirm the key numbers from the table above
y2 = swiss_re_curve(2.0)

print(f"Y2 parameters:  g = {y2.g:.4f},  b = {y2.b:.4f}")
print(f"Total loss prob: {y2.total_loss_prob():.4f}  (1/g = {1/y2.g:.4f})")
print(f"Mean destruction rate: {y2.mean():.4f}")
print()
print("Exposure curve values:")
for x in [0.10, 0.25, 0.50, 0.75, 1.00]:
    print(f"  G({x:.2f}) = {float(y2.exposure_curve(x)):.4f}")
```

**What you should see:**

```
Y2 parameters:  g = 7.6906,  b = 9.0196
Total loss prob: 0.1300  (1/g = 0.1300)
Mean destruction rate: 0.2139

Exposure curve values:
  G(0.10) = 0.3551
  G(0.25) = 0.5817
  G(0.50) = 0.7394
  G(0.75) = 0.8681
  G(1.00) = 1.0000
```

The mean destruction rate of 0.214 means: on average, a loss on a Y2 risk destroys 21.4% of the maximum possible loss. Already at x = 0.25 (a loss of 25% of MPL), we have captured 58% of expected loss. This reflects the heavy concentration of partial losses at the low end of the scale.