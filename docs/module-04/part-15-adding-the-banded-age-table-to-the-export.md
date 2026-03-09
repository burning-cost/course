## Part 15: Adding the banded age table to the export

The banded age relativities also need to go into Radar. But they came from manual aggregation of SHAP values rather than from `extract_relativities()`, so we need to format them manually.

In a new cell, type this and run it (Shift+Enter):

```python
# Format the age band table in Radar format
age_radar = band_rels.select(["age_band", "relativity"]).with_columns([
    pl.lit("driver_age_band").alias("Factor"),
    pl.col("age_band").alias("Level"),
    pl.col("relativity").round(4).alias("Relativity"),
]).select(["Factor", "Level", "Relativity"])

# Write to /dbfs/tmp/
age_radar_pd = age_radar.to_pandas()
age_radar_pd.to_csv("/dbfs/tmp/gbm_age_band_relativities_radar.csv", index=False)

print("Age band Radar export:")
print(age_radar_pd.to_string(index=False))
```

You will see the age band factor table in Radar format. This file, combined with the categorical feature file from Part 14, gives you everything Radar needs for the GBM-derived relativities.