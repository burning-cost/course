## Part 8: Validating against Emblem output

### The critical check

The following function compares your Python GLM relativities to an Emblem CSV export. The Emblem CSV format is typically:

```bash
Factor,Level,Relativity,SE,LowerCI,UpperCI
```

```python
def compare_to_emblem(
    python_rels: pl.DataFrame,
    emblem_path: str,
    tolerance: float = 0.001,
) -> pl.DataFrame:
    """
    Compare Python GLM relativities to an Emblem CSV export.
    Returns a comparison DataFrame with a 'match' column.
    """
    emblem_rels = pl.read_csv(
        emblem_path,
        schema_overrides={"Level": pl.Utf8, "Relativity": pl.Float64},
    )

    comparison = (
        python_rels
        .rename({"feature": "Factor", "level": "Level", "relativity": "Python_Rel"})
        .join(
            emblem_rels.rename({"Relativity": "Emblem_Rel"}).select(["Factor", "Level", "Emblem_Rel"]),
            on=["Factor", "Level"],
            how="inner",
        )
        .with_columns([
            ((pl.col("Python_Rel") - pl.col("Emblem_Rel")).abs()).alias("abs_diff"),
            ((pl.col("Python_Rel") / pl.col("Emblem_Rel") - 1).abs()).alias("rel_diff"),
        ])
        .with_columns(
            (pl.col("rel_diff") < tolerance).alias("match")
        )
        .sort(["Factor", "Level"])
    )

    n_matched = comparison["match"].sum()
    n_total = len(comparison)
    print(f"Matched: {n_matched}/{n_total} relativities within {tolerance*100:.1f}% tolerance")

    if n_matched < n_total:
        mismatches = comparison.filter(~pl.col("match"))
        print("\nMismatches:")
        print(mismatches.select(["Factor", "Level", "Python_Rel", "Emblem_Rel", "rel_diff"]))

    return comparison
```

### Tolerances to accept

- **Identical data, identical specification:** relativities should match to 4+ decimal places. If they do not, there is a specification mismatch.
- **Same data, minor specification differences** (e.g. Emblem rounds vehicle group to bands, Python uses continuous): expect differences that reflect the specification difference. Document them.
- **Different data vintage:** some differences are expected from the additional data. Verify the sign and approximate magnitude are consistent, but do not try to match exactly.

### Common reasons for mismatches

**Base level differs.** Emblem defaults to the highest-exposure level as base; Python defaults to alphabetical first. If you have not pinned the base level explicitly, every relativity for that factor will be off by a constant multiplier - all quotients `Python/Emblem` for that factor will be the same constant.

**Continuous vs categorical encoding.** Emblem often auto-detects whether a factor should be treated as continuous or categorical. Python's formula interface requires you to be explicit. If Emblem fit vehicle group as categorical (27 dummies for groups 1-27, 28-50 collapsed) and your Python model treats it as continuous, the relativities will not match.

**Missing level consolidation.** Emblem consolidates sparse levels automatically. Python estimates a separate coefficient for every level unless you merge them manually.

**Emblem manual overrides.** If the validation fails despite identical data and identical specification, the most likely explanation is a manual override in Emblem that was never documented. Someone clicked on a factor level in the Emblem UI and typed in a relativity. It happens constantly on live books.

Check whether any Emblem relativities are suspiciously round numbers - 1.000, 0.850, 1.250. A likelihood-based GLM will almost never produce a relativity of exactly 0.850 to three decimal places. If you see round-number relativities, talk to whoever built the original Emblem model before concluding there is a coding error in your Python model.