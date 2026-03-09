## Part 13: The two-stage pipeline for production use

In production, you do not use the integrated model (raw claims and exposure direct to BYM2). You use the two-stage approach.

### Why two-stage is better for production

**Decoupling.** The main rating model (CatBoost, GLM) is estimated on individual policy records. The spatial model is estimated on aggregated area-level data. These are different estimation problems with different data sources and update frequencies. Decoupling them means you can update the main model quarterly and the territory factors annually without rebuilding the entire system.

**Auditability.** A regulator reviewing your territory factors does not need to understand the full CatBoost model. They need to understand the spatial smoothing methodology applied to the sector-level O/E ratios. That is a simpler story.

**Stability.** Territory factors derived from a two-stage model change less year-on-year than territory factors derived from an integrated model, because the main model already absorbs non-spatial variation. The spatial model is left with a cleaner signal.

### The two-stage workflow

```python
# --- Stage 1 (assumed done in prior modules) ---
# You have a fitted CatBoost or GLM model.
# You compute expected claims per policy from the base model.
# You aggregate to sector level.

# Simulate this: suppose we have these sector-level actuals and base model expectations
# In practice, these come from your fitted base model applied to holdout data

# For this demonstration, generate a "base model" that gets the non-spatial
# variation right but misses the geographic pattern
base_expected = exposure * portfolio_freq  # intercept-only "base model"
# Sector-level observed and expected
sector_observed = claims.copy()
sector_expected = base_expected.copy()

# The O/E ratio per sector
sector_oe = sector_observed / sector_expected.clip(0.1)  # avoid /0

# --- Stage 2: BYM2 on O/E ---
# Pass observed claims as the observed data, expected claims as the "exposure"
# The model will estimate log(O/E) per sector as the territory effect
model_2s = BYM2Model(
    adjacency=adj,
    draws=1000,
    chains=4,
    target_accept=0.9,
    tune=1000,
)

result_2s = model_2s.fit(
    claims=sector_observed,
    exposure=sector_expected,  # E_i = base model prediction (not policy-years)
    random_seed=42,
)

print("Two-stage model fitted.")
diag_2s = result_2s.diagnostics()
print(f"Max R-hat: {diag_2s.convergence.max_rhat:.4f}")
print(f"rho posterior mean: {diag_2s.rho_summary['mean'][0]:.3f}")
```

**Important:** In the two-stage model, the `exposure` argument is the base model's expected claims per sector -- not policy-years. The model is then estimating the multiplicative adjustment on top of the base model. The output relativities are applied as an additional factor in the rating formula.

### Saving results to Delta

```python
# Save relativities to Delta for downstream use
rels_2s = result_2s.territory_relativities()

# Convert Polars to Spark for writing to Delta
rels_spark = spark.createDataFrame(rels_2s.to_pandas())
(rels_spark
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("pricing.territory.bym2_relativities_v1"))

print("Relativities written to Delta table: pricing.territory.bym2_relativities_v1")
```