## Part 14: Exporting for Radar

Radar (Willis Towers Watson's pricing software) expects factor tables in a specific CSV format:

```python
Factor,Level,Relativity
area,A,1.0000
area,B,1.1080
...
```

In a new cell, type this and run it (Shift+Enter):

```python
def to_radar_csv(rels_df: pl.DataFrame, output_path: str) -> None:
    """
    Export relativities in Radar factor table import format.

    Expects a Polars DataFrame with columns: feature, level, relativity.
    Continuous features should be excluded or pre-banded.
    """
    radar_df = (
        rels_df.select("feature", "level", "relativity")
        .rename({"feature": "Factor", "level": "Level", "relativity": "Relativity"})
        .with_columns(
            pl.col("Relativity").round(4),
            pl.col("Level").cast(pl.Utf8),
        )
    )
    radar_df.write_csv(output_path)
    print(f"Written {len(radar_df)} rows to {output_path}")


# Export categorical features only
# Continuous features (ncd_years is treated as categorical here; driver_age needs banding first)
cat_rels_for_export = rels.filter(pl.col("feature").is_in(CAT_FEATURES))
to_radar_csv(cat_rels_for_export, "/dbfs/tmp/gbm_relativities_radar.csv")
```

You will see:

```sql
Written 8 rows to /dbfs/tmp/gbm_relativities_radar.csv
```

Verify the output looks correct. In a new cell, type this and run it (Shift+Enter):

```python
radar_check = pl.read_csv("/dbfs/tmp/gbm_relativities_radar.csv")
print(radar_check)
```

You will see the Radar-format CSV contents. Verify that:

1. Every area level (A through F) is present
2. The base level (A) has Relativity = 1.0000
3. Relativities increase monotonically from A to F

**Two practical issues with real Radar imports:**

First, your Radar variable names are almost certainly different from your Python column names. `area` in Python might be `ABI_AREA_BAND` in Radar. Map them explicitly in a lookup dictionary before export and version-control that lookup.

Second, Radar requires every level defined in the model to appear in the import file. If your GBM never saw a level that Radar knows about (e.g. NCD=6 if your insurer does not write those risks), you need to add that row manually with an appropriate relativity - either extrapolate from the nearest observed level or use 1.0 with a documented decision. Never let Radar default silently.