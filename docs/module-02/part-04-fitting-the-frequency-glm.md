## Part 4: Fitting the frequency GLM

### The exposure offset - the most important concept in this module

Before fitting, we need to understand what the exposure offset does and why it matters.

A Poisson GLM without an offset fits raw claim counts. A policy with 0.5 earned years should generate half as many expected claims as an identical policy with 1.0 earned years - not because it is lower risk, but because it was only exposed for half the time. Without the offset, the model cannot distinguish between "low risk" and "short exposure." It will learn the exposure duration as a spurious predictor of claim count, producing biased coefficients for everything else.

The offset fixes this by entering the linear predictor as a term with a fixed coefficient of exactly 1:

```sql
log(E[claims_i]) = log(exposure_i) + intercept + beta_area × area_i + ...
```

Rearranging: `E[claims_i] = exposure_i × exp(intercept + betas)`. The model is fitting the annualised rate, and the exposure scales the expected claims for each policy. A policy with 0.5 years contributes exactly half of a full-year policy's expected claims.

In statsmodels, the offset argument takes the log-exposure vector we computed above.

### The formula string

statsmodels uses a formula syntax (from the `patsy` library) to specify the model. It looks like R:

```sql
"claim_count ~ C(area) + C(ncd_years, Treatment(0)) + C(conviction_flag, Treatment(0)) + vehicle_group"
```

- `claim_count` is the response variable (left of the tilde)
- `C(area)` tells patsy to treat area as a categorical factor, creating dummy variables
- `C(ncd_years, Treatment(0))` treats NCD years as categorical with NCD=0 as the explicit base level
- `vehicle_group` (without `C()`) treats vehicle group as a continuous variable - one slope, not k-1 dummies

**Why not `C(vehicle_group)`?** Because with 50 ABI groups, using each as a separate dummy variable costs 49 degrees of freedom. Treating it as continuous costs 1 degree of freedom and gives a smooth, monotone effect if the relationship is approximately linear in logs. The synthetic data was generated with a linear vehicle group effect (0.018 per group), so this is the correct specification here. On real data, plot the one-way A/E first and decide.

### Fitting the model

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

freq_formula = (
    "claim_count ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_freq = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

print(glm_freq.summary())
```

Run this cell. It will take a few seconds (IRLS is iterative). When it completes, you will see a large summary table.

**What to check in the summary:**

- `Converged: True` - if this says False, the model did not reach a solution. Treat all results with suspicion.
- `Method: IRLS` - confirms we are using the same algorithm as Emblem
- `Deviance` - the total deviance. For Poisson, a well-specified model has deviance roughly equal to the residual degrees of freedom (here, around 99,988). If deviance is materially above df_resid, the model may be overdispersed.
- The coefficient table - columns are: `coef` (the log-relativity), `std err` (standard error), `z` (z-statistic), `P>|z|` (p-value), and 95% confidence interval

### Reading the coefficient table

The coefficient for area B will appear as `C(area)[T.B]`. The `[T.B]` suffix means "treatment contrast: level B relative to the reference level (A)." The coefficient is approximately 0.099, so the area B frequency relativity is `exp(0.099) ≈ 1.104`. The true parameter was 0.10, so `exp(0.10) = 1.105`. Our model has recovered it to within rounding error.

Check whether the model has converged correctly:

```python
print(f"Converged: {glm_freq.converged}")
print(f"Iterations: {glm_freq.nit}")
print(f"Deviance: {glm_freq.deviance:,.1f}")
print(f"Residual df: {glm_freq.df_resid:,.0f}")
print(f"Deviance/df: {glm_freq.deviance / glm_freq.df_resid:.3f}")

# Check for aliased (dropped) parameters
nan_params = glm_freq.params[glm_freq.params.isna()]
if len(nan_params) > 0:
    print(f"\nWARNING: {len(nan_params)} aliased parameters (NaN coefficients):")
    print(nan_params)
else:
    print("\nNo aliased parameters - design matrix is full rank")
```

You should see `Converged: True`, iterations in single digits, and a deviance/df ratio close to 1.0. A ratio above 1.3 indicates overdispersion - more on that in the diagnostics section.

**A new Python concept: dictionary.** `TRUE_PARAMS` earlier in the module is a dictionary - a collection of key-value pairs. You access a value with `TRUE_PARAMS["area_B"]`. Dictionaries are useful for storing named parameters like this, where the name (e.g. "area_B") makes the code self-documenting.

### Extracting relativities

The raw output from statsmodels is in log-space (the coefficient table) and uses patsy's naming convention. For a pricing actuary used to Emblem's factor tables, we want multiplicative relativities in a clean format. This function does the conversion:

```python
def extract_freq_relativities(glm_result, base_levels: dict) -> pl.DataFrame:
    """
    Extract multiplicative relativities from a fitted statsmodels GLM.

    Returns a Polars DataFrame with columns:
        feature, level, log_relativity, relativity, se, lower_ci, upper_ci

    base_levels: dict mapping each factor name to its base level value.
                 These get relativity=1.0, which is the definition of a base level.
    """
    records = []
    params = glm_result.params
    conf_int = glm_result.conf_int()

    for param_name, coef in params.items():
        if param_name == "Intercept":
            continue

        lo = conf_int.loc[param_name, 0]
        hi = conf_int.loc[param_name, 1]
        se = glm_result.bse[param_name]

        # Parse the patsy parameter name: "C(area)[T.B]" -> feature="area", level="B"
        if "[T." in param_name:
            feature_part = param_name.split("[T.")[0]
            level_part = param_name.split("[T.")[1].rstrip("]")
            if feature_part.startswith("C("):
                feature_part = feature_part[2:].split(",")[0].split(")")[0].strip()
        else:
            # Continuous feature - single coefficient, not per-level
            feature_part = param_name
            level_part = "continuous"

        records.append({
            "feature": feature_part,
            "level": level_part,
            "log_relativity": coef,
            "relativity": np.exp(coef),
            "se": se,
            "lower_ci": np.exp(lo),
            "upper_ci": np.exp(hi),
        })

    rels = pl.DataFrame(records)

    # Add a row for each base level (relativity = 1.0 by definition)
    base_rows = []
    for feat, base_level in base_levels.items():
        base_rows.append({
            "feature": feat,
            "level": str(base_level),
            "log_relativity": 0.0,
            "relativity": 1.0,
            "se": 0.0,
            "lower_ci": 1.0,
            "upper_ci": 1.0,
        })

    return pl.concat([pl.DataFrame(base_rows), rels]).sort(["feature", "level"])
    # Note: .sort(["feature", "level"]) sorts lexicographically on the string "level" column.
    # For factors with numeric levels above 9, this produces wrong order: "10" sorts before "2".
    # If your factor has levels 0-9 only, string sort is fine. For NCD years 0-10+, cast level
    # to Int32 before sorting, or sort after converting to the display format you need.


freq_rels = extract_freq_relativities(
    glm_freq,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)

# Look at area relativities
print(freq_rels.filter(pl.col("feature") == "area"))
```

**What you should see:**

```python
shape: (6, 7)
┌─────────┬───────┬─────────────────┬────────────┬──────────┬────────────┬────────────┐
│ feature ┆ level ┆ log_relativity  ┆ relativity ┆ se       ┆ lower_ci   ┆ upper_ci   │
│ str     ┆ str   ┆ f64             ┆ f64        ┆ f64      ┆ f64        ┆ f64        │
╞═════════╪═══════╪═════════════════╪════════════╪══════════╪════════════╪════════════╡
│ area    ┆ A     ┆ 0.0             ┆ 1.0        ┆ 0.0      ┆ 1.0        ┆ 1.0        │
│ area    ┆ B     ┆ 0.099...        ┆ 1.104...   ┆ 0.024... ┆ 1.057...   ┆ 1.152...   │
│ area    ┆ C     ┆ 0.197...        ┆ 1.218...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ D     ┆ 0.348...        ┆ 1.417...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ E     ┆ 0.499...        ┆ 1.647...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ F     ┆ 0.648...        ┆ 1.912...   ┆ ...      ┆ ...        ┆ ...        │
└─────────┴───────┴─────────────────┴────────────┴──────────┴────────────┴────────────┘
```

Area F relativity: approximately 1.912. True value: `exp(0.65) = 1.916`. We are within 0.2%.

Now verify the NCD relativities against what we know the true values should be:

```python
# True NCD=5 vs NCD=0 relativity: exp(-0.13 × 5) = exp(-0.65) ≈ 0.522
# True conviction uplift: exp(0.42) ≈ 1.52

print("NCD relativities:")
print(freq_rels.filter(pl.col("feature") == "ncd_years"))

print("\nConviction relativity:")
print(freq_rels.filter(pl.col("feature") == "conviction_flag"))

true_ncd5 = np.exp(-0.13 * 5)
true_conviction = np.exp(0.42)
print(f"\nTrue NCD=5 relativity: {true_ncd5:.4f}")
print(f"True conviction relativity: {true_conviction:.4f}")
```

If the model is working correctly, the estimated relativities will be within a few percent of the true values. They will not be exact - a sample of 100,000 policies has sampling variation.