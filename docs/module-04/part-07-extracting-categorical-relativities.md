## Part 7: Extracting categorical relativities

In a new cell, type this and run it (Shift+Enter):

```python
TRUE_PARAMS = {
    "area_B": 0.10, "area_C": 0.20, "area_D": 0.35,
    "area_E": 0.50, "area_F": 0.70,
    "ncd_years": -0.15,
    "has_convictions": 0.45,
}

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area":             "A",
        "has_convictions":  0,
    },
)

print("Area relativities:")
print(rels[rels["feature"] == "area"].to_string(index=False))
```

**Note on return type:** `extract_relativities()` returns a **pandas DataFrame**, not a Polars DataFrame. This is because the underlying SHAP library (which shap-relativities wraps) works with pandas and numpy natively. Methods like `.to_string(index=False)`, `.set_index("level")`, and `.copy()` used throughout this module are pandas methods - this is intentional and correct.

You will see:

```sql
Area relativities:
 feature level  relativity  lower_ci  upper_ci  mean_shap  shap_std   n_obs  exposure_weight
    area     A       1.000     1.000     1.000     -0.613     0.033    9985           7951.2
    area     B       1.108     1.063     1.155     -0.522     0.037   18042          14368.5
    area     C       1.225     1.183     1.269     -0.430     0.030   24998          19901.8
    area     D       1.431     1.381     1.483     -0.278     0.031   22015          17527.7
    area     E       1.668     1.607     1.731     -0.092     0.034   15048          11982.3
    area     F       1.950     1.869     2.034      0.110     0.037   10012           7968.0
```

The exact numbers will differ slightly from this. The important check is area F: the true DGP has `area_F = 0.70`, giving `exp(0.70) = 2.014`. The extracted relativity of 1.950 is close - the difference is partly sampling variation, partly the GBM's imperfect separation of area from other features.

Now look at NCD. In a new cell, type this and run it (Shift+Enter):

```python
print("NCD relativities:")
ncd_rels = rels[rels["feature"] == "ncd_years"].sort_values("level")
print(ncd_rels[["level", "relativity", "lower_ci", "upper_ci", "n_obs"]].to_string(index=False))

# True DGP: exp(-0.15 * k) for k = 0..5
print("\nTrue DGP NCD relativities:")
for k in range(6):
    print(f"  NCD={k}: {np.exp(-0.15 * k):.3f}")
```

You will see NCD relativities decreasing from 1.000 at NCD=0 to around 0.47-0.50 at NCD=5. The true DGP gives `exp(-0.15 × 5) = exp(-0.75) ≈ 0.472`. If your NCD=5 relativity is between 0.42 and 0.53, the model is working correctly.

Now look at convictions. In a new cell, type this and run it (Shift+Enter):

```python
print("Conviction relativities:")
conv_rels = rels[rels["feature"] == "has_convictions"]
print(conv_rels[["level", "relativity", "lower_ci", "upper_ci", "n_obs"]].to_string(index=False))
print(f"\nTrue DGP conviction relativity: exp(0.45) = {np.exp(0.45):.3f}")
```

You should see the conviction relativity (level=1) somewhere around 1.45-1.65. The true value is `exp(0.45) ≈ 1.568`. The interval should comfortably include 1.568.

### What each column means

The output includes several columns beyond the relativity itself:

- **mean_shap** - the exposure-weighted mean SHAP value for this level. The relativity is `exp(mean_shap - mean_shap_base)`.
- **shap_std** - exposure-weighted standard deviation of SHAP values within this level. Higher values mean more within-level variation - the GBM's predictions for this level are context-dependent.
- **n_obs** - number of observations at this level.
- **exposure_weight** - total exposure in years at this level.
- **lower_ci / upper_ci** - 95% confidence interval on the relativity.

Do not discard these columns when presenting to the pricing committee. The `shap_std` and `n_obs` are what you need to explain why one level has a wide CI and another has a narrow CI.
