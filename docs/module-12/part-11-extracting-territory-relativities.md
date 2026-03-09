## Part 11: Extracting territory relativities

Once convergence is confirmed and post-fit diagnostics look clean, we extract the territory relativities. These are the multiplicative factors -- one per area -- that go into the rating engine.

```python
rels = result.territory_relativities(credibility_interval=0.95)

print(rels.head(10))
print()
print(f"Columns: {rels.columns}")
print(f"Relativity range: [{rels['relativity'].min():.4f}, {rels['relativity'].max():.4f}]")
print(f"Areas with relativity > 1.10: {(rels['relativity'] > 1.10).sum()}")
print(f"Areas with relativity < 0.90: {(rels['relativity'] < 0.90).sum()}")
```

The output DataFrame has these columns:

| Column | Meaning |
|--------|---------|
| `area` | Area identifier (matches adj.areas) |
| `b_mean` | Posterior mean of b_i (log scale) |
| `b_sd` | Posterior SD of b_i (log scale) |
| `relativity` | exp(b_i - grand_mean_b); multiplicative factor |
| `lower` | Lower 95% credibility bound on relativity |
| `upper` | Upper 95% credibility bound on relativity |
| `ln_offset` | log(relativity) = b_mean - grand_mean_b; ready to use as GLM offset |

**Normalisation:** By default, relativities are normalised to the geometric mean. That is, exp(mean(log(rel_i))) = 1. No area is the explicit reference -- the factors multiply to 1 across the portfolio. This is the natural normalisation for a territory factor that exists alongside other factors in a multiplicative tariff.

If you prefer a specific reference area (e.g., the area containing your HQ, or the historically used baseline territory):

```python
# Normalise to a specific area
rels_ref = result.territory_relativities(base_area="r5c5")
print(rels_ref.filter(pl.col("area") == "r5c5"))
# Should show relativity = 1.0 for the reference area
```

### Sorting and inspection

```python
# Highest-risk areas
print("Top 10 highest risk:")
print(rels.sort("relativity", descending=True).head(10))

print()
print("Top 10 lowest risk:")
print(rels.sort("relativity").head(10))

# Credibility interval width as a measure of uncertainty
rels = rels.with_columns(
    (pl.col("upper") - pl.col("lower")).alias("ci_width")
)
print()
print("Widest credibility intervals (most uncertain):")
print(rels.sort("ci_width", descending=True).head(10))
```

Notice that the widest credibility intervals correspond to areas with sparse exposure. This is the credibility principle made explicit: areas with more data get tighter estimates.