## Part 11: Comparing to the GLM

The strongest argument for GBM relativities is that they match the GLM on main effects but reveal additional structure where the GLM's linearity assumptions fail. You need to demonstrate this comparison explicitly.

In a new cell, type this and run it (Shift+Enter):

```python
# Fit a Poisson GLM on the same features
glm_formula = (
    "claim_count ~ C(area) + ncd_years + C(has_convictions) + vehicle_group + driver_age"
)

glm = smf.glm(
    formula=glm_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df_pd["exposure"].clip(1e-6)),
).fit()

print(glm.summary().tables[1])
```

The output is a GLM coefficient table. Look for:

- `C(area)[T.B]` through `C(area)[T.F]` - these are the area log-relativities relative to area A
- `ncd_years` - the coefficient for each NCD year (linear effect)
- `C(has_convictions)[T.1]` - the log-relativity for having at least one conviction

Now build a side-by-side comparison. In a new cell, type this and run it (Shift+Enter):

```python
# GLM area relativities
area_levels = ["A", "B", "C", "D", "E", "F"]
glm_area_rels = {}
glm_area_rels["A"] = 1.0
for lvl in area_levels[1:]:
    coef_name = f"C(area)[T.{lvl}]"
    glm_area_rels[lvl] = np.exp(glm.params.get(coef_name, 0.0))

# GBM area relativities from extract_relativities()
gbm_area = rels[rels["feature"] == "area"].set_index("level")

print(f"{'Area':<6} {'True':>8} {'GLM':>8} {'GBM':>8} {'GBM CI':>20}")
print("-" * 55)
true_rels = {"A": 1.0, "B": np.exp(0.10), "C": np.exp(0.20),
             "D": np.exp(0.35), "E": np.exp(0.50), "F": np.exp(0.70)}
for lvl in area_levels:
    true_r  = true_rels[lvl]
    glm_r   = glm_area_rels[lvl]
    gbm_r   = gbm_area.loc[lvl, "relativity"]
    gbm_lo  = gbm_area.loc[lvl, "lower_ci"]
    gbm_hi  = gbm_area.loc[lvl, "upper_ci"]
    print(f"{lvl:<6} {true_r:>8.3f} {glm_r:>8.3f} {gbm_r:>8.3f}  [{gbm_lo:.3f}, {gbm_hi:.3f}]")
```

You will see something like:

```
Area   True      GLM      GBM           GBM CI
-------------------------------------------------------
A     1.000    1.000    1.000  [1.000, 1.000]
B     1.105    1.089    1.108  [1.063, 1.155]
C     1.221    1.208    1.225  [1.183, 1.269]
D     1.419    1.401    1.431  [1.381, 1.483]
E     1.649    1.631    1.668  [1.607, 1.731]
F     2.014    1.971    1.950  [1.869, 2.034]
```

**What to tell the committee:** GLM and GBM agree closely on area relativities. Any differences are within the GBM's confidence interval. Both models recover the true DGP well for this feature.

Now compare NCD. In a new cell, type this and run it (Shift+Enter):

```python
glm_ncd_coef = glm.params.get("ncd_years", 0.0)
gbm_ncd = rels[rels["feature"] == "ncd_years"].set_index("level")

print(f"NCD comparison (GLM coefficient = {glm_ncd_coef:.4f}, true = -0.15)")
print(f"\n{'NCD':<6} {'True':>8} {'GLM':>8} {'GBM':>8}")
print("-" * 36)
for k in range(6):
    true_r = np.exp(-0.15 * k)
    glm_r  = np.exp(glm_ncd_coef * k)
    if k in gbm_ncd.index:
        gbm_r = gbm_ncd.loc[k, "relativity"]
        print(f"{k:<6} {true_r:>8.3f} {glm_r:>8.3f} {gbm_r:>8.3f}")
```

You will see that the GLM and GBM agree closely on NCD - for this dataset, both are well-specified for the NCD effect. The GBM does not add much value on a feature with a clean linear relationship.

### Where the GBM adds value: driver age

The most important comparison is driver age, where the GLM's linear assumption fails. In a new cell, type this and run it (Shift+Enter):

```python
# GLM: linear age coefficient
glm_age_coef = glm.params.get("driver_age", 0.0)
print(f"GLM driver_age coefficient: {glm_age_coef:.5f}")
print("(GLM forces a single linear effect across all ages)")

# GBM: extract band relativities for driver age
print("\nGBM age band vs GLM linear prediction (base: 30-39):")
band_age_mid = {
    "17-21": 19, "22-24": 23, "25-29": 27,
    "30-39": 34, "40-54": 47, "55-69": 62, "70+": 75,
}
base_age = 34  # midpoint of 30-39

print(f"{'Band':<10} {'GLM (linear)':>14} {'GBM (banded)':>14}")
print("-" * 42)
for band_label, mid_age in band_age_mid.items():
    glm_pred = np.exp(glm_age_coef * (mid_age - base_age))
    gbm_row  = band_rels.filter(pl.col("age_band") == band_label)
    gbm_pred = gbm_row["relativity"][0] if len(gbm_row) > 0 else float("nan")
    print(f"{band_label:<10} {glm_pred:>14.3f} {gbm_pred:>14.3f}")
```

You will see that the GLM produces nearly flat or weakly sloped relativities across all age bands because the linear coefficient is pulled towards zero by the majority of mid-range-age drivers. The GBM shows the U-shape clearly: high relativities for young drivers, flat middle, mild uplift at 70+.

This is the table to put in front of the pricing committee. The GBM reveals a real risk pattern that the GLM cannot see.