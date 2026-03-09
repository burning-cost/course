## Part 5: Handling sparse factor levels

Before we move to the severity model, a practical note on something that bites real projects.

Real claims extracts will have factor levels with very few policies: area codes that appear in the policy file but not in the training data, occupation codes with 2 claims, NCD levels with 12 policies. Emblem consolidates sparse levels automatically. Python estimates a separate coefficient for every level unless you intervene. The result is extremely wide confidence intervals (or outright NaN coefficients) for sparse levels.

Here is the pattern to use in Polars for grouping sparse levels by exposure:

```python
# Identify area levels with fewer than 50 earned years of exposure
area_exposure = (
    df
    .group_by("area")
    .agg(pl.col("exposure").sum().alias("total_exposure"))
)

sparse_areas = area_exposure.filter(pl.col("total_exposure") < 50)["area"].to_list()
print(f"Sparse area levels: {sparse_areas}")

# In our synthetic data, this should be empty - all areas are well-populated.
# On real data, group sparse levels into "Other":
if sparse_areas:
    df = df.with_columns(
        pl.when(pl.col("area").is_in(sparse_areas))
        .then(pl.lit("Other"))
        .otherwise(pl.col("area").cast(pl.Utf8))
        .alias("area")
    )
    print(f"Merged {len(sparse_areas)} sparse levels into 'Other'")
```

The threshold (50 earned years here) is a business judgement. A level with fewer than about 30-50 years of exposure will produce a relativity with such a wide confidence interval that it is essentially noise. Merge it with a generic "Other" bucket and document the consolidation in your model notes.