## Part 8: Understanding normalise_to

The `normalise_to` parameter controls how the relativities are scaled. In a new cell, type this and run it (Shift+Enter):

```python
rels_mean = sr.extract_relativities(normalise_to="mean")

print("Area relativities, normalised to exposure-weighted mean:")
area_mean = rels_mean[rels_mean["feature"] == "area"].sort_values("level")
print(area_mean[["level", "relativity"]].to_string(index=False))

print("\nArea relativities, normalised to base level A:")
area_base = rels[rels["feature"] == "area"].sort_values("level")
print(area_base[["level", "relativity"]].to_string(index=False))
```

You will see two different-looking tables for area, even though the underlying model has not changed.

`normalise_to="base_level"` sets the named base level to exactly 1.000. All other levels are expressed relative to it. This matches GLM convention and is what you use for Radar imports and committee presentations.

`normalise_to="mean"` sets the exposure-weighted portfolio mean to 1.000. Levels above 1.0 are above-average risk; levels below 1.0 are below-average. This is useful for internal portfolio analysis but is harder to compare to a GLM.

**Rule of thumb:** use `base_level` when comparing to a GLM or importing to a rating system. Use `mean` when presenting to an underwriting team who thinks in terms of "above average" and "below average" risk.