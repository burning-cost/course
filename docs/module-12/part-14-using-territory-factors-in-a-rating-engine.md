## Part 14: Using territory factors in a rating engine

Territory relativities from BYM2 are multiplicative factors that feed into an existing rating formula. There are two ways to incorporate them.

### Option 1: As a fixed GLM offset

The cleanest integration is to add the territory relativity as a fixed offset in a downstream GLM. The `ln_offset` column from `territory_relativities()` is ready-made for this.

```python
# Suppose you have a policy-level DataFrame df_policies with a sector column
# Join the BYM2 relativities
rels_for_join = rels.select(["area", "ln_offset"]).rename({"ln_offset": "territory_log_offset"})

# In Polars
df_policies_with_territory = df_policies.join(
    rels_for_join,
    left_on="postcode_sector",
    right_on="area",
    how="left",
)

# For sectors not in the BYM2 model (e.g., new sectors),
# fill with 0.0 (no territory adjustment -- grand mean)
df_policies_with_territory = df_policies_with_territory.with_columns(
    pl.col("territory_log_offset").fill_null(0.0)
)
```

In the GLM, you then include `territory_log_offset` as a fixed offset (not a free parameter). This forces the GLM to accept the territory factor exactly as BYM2 estimated it, adjusting only the remaining non-territorial effects.

```python
# Pseudo-code for a GLM with offset -- library-specific
# The key is that territory_log_offset has coefficient fixed at 1.0, not estimated
# In statsmodels:
#   model = sm.GLM(y, X, family=Poisson(), offset=df["territory_log_offset"])
```

### Option 2: As a lookup factor in a multiplicative tariff

For simpler rating engines that use lookup tables rather than a GLM at point-of-sale:

```python
# Create the territory factor table for the rating engine
factor_table = rels.select([
    pl.col("area").alias("postcode_sector"),
    pl.col("relativity").alias("territory_factor"),
    pl.col("lower").alias("territory_factor_lower_95"),
    pl.col("upper").alias("territory_factor_upper_95"),
]).sort("postcode_sector")

print(factor_table.head(10))

# The premium calculation becomes:
# premium = base_rate * age_factor * ncb_factor * vehicle_factor * territory_factor
```

The territory factor replaces the Emblem territory band factor. The difference is that:
- Each sector has its own factor (no banding discretisation error)
- Each factor comes with a credibility interval (uncertainty is explicit)
- The factors are smoothed (nearby sectors cannot differ by 30% due to sampling noise)

### Handling new postcode sectors

New postcode sectors (carved out from existing ones by Royal Mail) will not be in the BYM2 model. Use the relativity of the parent sector, or the exposure-weighted average of adjacent sectors, as the starting point. Document this fallback rule clearly -- it is the kind of edge case that surfaces in regulatory review.