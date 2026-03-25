## Part 10: The elasticity surface

The elasticity surface is the primary deliverable from this analysis — the visualisation that goes to the pricing committee and forms the basis for segment-level pricing strategy. It shows how price sensitivity varies across two dimensions simultaneously.

### Building the segment summary

```python
%md
## Part 10: Elasticity surface
```

```python
surface = ElasticitySurface(est)

# Segment summary table: NCD × channel
summary_ncd_channel = surface.segment_summary(df, by=["ncd_years", "channel"])
print("Elasticity surface — NCD years × channel:")
print(summary_ncd_channel)
```

The `elasticity_at_10pct` column converts the semi-elasticity to a concrete prediction: the expected change in renewal probability for a 10% price increase (log change = 0.0953). This is often the most useful number for the pricing committee — it translates the statistical estimate into a business impact.

### The NCD × age heatmap

```python
# Heatmap: NCD years × age band
fig_surface = surface.plot_surface(df, dims=["ncd_years", "age_band"])
fig_surface.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")
plt.show()
```

The heatmap colours each cell by the average CATE. Red (more negative) is more elastic. The expected pattern:

- **Top-left** (NCD=0, age 17–24): darkest red, most elastic. A 10% price increase on a 22-year-old with no NCD changes renewal probability by around −0.33 pp.
- **Bottom-right** (NCD=5, age 65+): lightest colour, least elastic. Same 10% increase changes renewal by around −0.10 pp.

This 3× difference in sensitivity should drive materially different pricing strategies for the two segments.

### GATE bar charts

```python
# Bar chart with CIs: channel
fig_channel = surface.plot_gate(df, by="channel")
fig_channel.savefig("/tmp/gate_by_channel.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# Bar chart with CIs: NCD
fig_ncd = surface.plot_gate(df, by="ncd_years")
fig_ncd.savefig("/tmp/gate_by_ncd.png", dpi=150, bbox_inches="tight")
plt.show()
```

The bar charts include 95% confidence intervals. If the CI for NCD=0 does not overlap with NCD=5, the difference is statistically significant. On 50,000 observations it should be — the true difference is 2.5 semi-elasticity units (−3.5 vs. −1.0).

### Exporting the surface to Delta

In production, write the segment summary to Unity Catalog as part of the pricing run artefacts:

```python
# Write elasticity surface to Unity Catalog
spark.createDataFrame(
    surface.segment_summary(df, by=["ncd_years", "channel", "age_band"]).to_pandas()
).write.format("delta").mode("overwrite").saveAsTable(
    "pricing.motor.elasticity_surface_v1"
)
print("Elasticity surface written to pricing.motor.elasticity_surface_v1")
```

The table should be versioned (append-mode with a run date column in production) so you can track how the estimated elasticities change over time as you accumulate more renewal experience.
