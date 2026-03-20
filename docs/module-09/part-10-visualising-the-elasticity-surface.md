## Part 10: Visualising the elasticity surface

The elasticity surface shows how price sensitivity varies across two dimensions simultaneously. It is the deliverable that goes to the pricing committee and the commercial director.

```python
%md
## Part 10: Elasticity surface
```

```python
from insurance_causal.elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est_renewal)

# Heatmap: NCD years x age band
fig_surface = surface.plot_surface(df_renewals, dims=["ncd_years", "age_band"])
fig_surface.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")

# Bar chart: elasticity by channel
fig_gate_channel = surface.plot_gate(df_renewals, by="channel")
fig_gate_channel.savefig("/tmp/gate_by_channel.png", dpi=150, bbox_inches="tight")

print("Plots saved to /tmp/")
```

To display the plots inline in the Databricks notebook, add:

```python
import matplotlib.pyplot as plt
plt.show()
```

Note that `plt.show()` in Databricks renders the most recently created figure inline. To display both figures, add a `plt.show()` call after each `savefig`.

### Reading the heatmap

The elasticity surface heatmap shows a grid with NCD years on one axis and age band on the other. Each cell is coloured by the average CATE for customers in that combination. Darker red (more negative) cells are the most elastic segments. The pricing implication is immediate: segments in the dark red cells need either careful price management (to avoid volume loss) or explicit retention discounts.

The top-left corner (low NCD, young age) should be the darkest. This is the 17-24 year old with no NCD: the most elastic customer on the book. A 10% price increase on this customer, at a true elasticity of -3.5, changes their renewal probability by approximately 0.33 percentage points. At a base renewal rate of 70%, that is a relative reduction of 0.5%.

The bottom-right corner (high NCD, older age) should be the lightest. This is the 65+ customer with 5 years NCD: the least elastic, the safest to price towards the ENBP ceiling.