## Part 9: Exporting to Radar

Willis Towers Watson Radar imports external relativity tables as CSVs. The format is:

```python
Factor,Level,Relativity
area,A,1.0000
area,B,1.1041
area,C,1.2185
```

The factor name must match the rating variable name in the Radar model exactly. Radar is case-sensitive. If your Python column is `area` and your Radar variable is `Area`, the import will fail silently - Radar will apply a relativity of 1.0 for all levels. Check the exact names in your Radar project before export.

```python
def to_radar_csv(
    rels: pl.DataFrame,
    output_path: str,
    factor_name_map: dict | None = None,
    decimal_places: int = 4,
) -> None:
    """
    Export relativities in Radar factor table import format.

    factor_name_map: dict mapping Python column names to Radar variable names.
                     e.g. {"area": "PostcodeArea", "ncd_years": "NCDYears"}
    """
    radar_df = rels.select(["feature", "level", "relativity"])

    if factor_name_map:
        radar_df = radar_df.with_columns(
            pl.col("feature").replace(factor_name_map).alias("feature")
        )

    radar_df = (
        radar_df
        .rename({"feature": "Factor", "level": "Level", "relativity": "Relativity"})
        .with_columns(
            pl.col("Relativity").round(decimal_places).alias("Relativity")
        )
    )

    radar_df.write_csv(output_path)
    print(f"Exported {len(radar_df)} factor table rows to {output_path}")


# Export frequency relativities for Radar
# Only export categorical factors - continuous features need banding first
cat_rels = freq_rels.filter(pl.col("level") != "continuous")

# On Databricks, /tmp/ is a local path accessible from the notebook.
# For production, use /dbfs/mnt/pricing/outputs/ or equivalent.
to_radar_csv(
    cat_rels,
    "/tmp/freq_relativities_radar.csv",
    factor_name_map={
        "area": "PostcodeArea",
        "ncd_years": "NCDYears",
        "conviction_flag": "ConvictionFlag",
    },
)
```

One practical issue: Radar requires every level that exists in the policy file to appear in the import table. If your rating variable has NCD=6 (some insurers allow it for advanced drivers) but your Python model merged NCD=5 and NCD=6 into a single level, you need to add a NCD=6 row to the export. Decide the relativity - usually the NCD=5 relativity, or a mild discount beyond it - and add it explicitly. Do not let this happen quietly; document it in your model notes.