## Part 7: Diagnostics

Running a GLM without diagnostics is not modelling. These checks tell you whether the model is well-specified and where it fails.

### Deviance residuals for the frequency model

The deviance residual for each observation measures how far the observed claim count is from the model's prediction, on a scale that accounts for the Poisson distribution.

```python
import matplotlib.pyplot as plt
from scipy import stats

resid_deviance = glm_freq.resid_deviance
resid_std = resid_deviance / np.sqrt(glm_freq.scale)
fitted_vals = glm_freq.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs fitted
axes[0].scatter(np.log(fitted_vals), resid_std, alpha=0.1, s=5, color="steelblue")
axes[0].axhline(0, color="black", linestyle="--", lw=1)
axes[0].axhline(2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].axhline(-2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].set_xlabel("log(fitted frequency)")
axes[0].set_ylabel("Deviance residual")
axes[0].set_title("Residuals vs Fitted - Frequency GLM")

# Normal QQ plot
stats.probplot(resid_std, dist="norm", plot=axes[1])
axes[1].set_title("Normal QQ - Deviance Residuals")

plt.tight_layout()
plt.show()
```

**What to look for:**

- Residuals should show no strong pattern against fitted values. A funnel shape (residuals increasing with fitted values) suggests overdispersion.
- More than 5% of residuals outside the red ±2 lines suggests either genuine overdispersion or a systematic missing feature.
- The QQ plot for Poisson GLM deviance residuals will not be perfectly normal (the Poisson is discrete), but the upper tail should not be dramatically heavier than normal.

### Actual vs Expected by factor level

This is the diagnostic Emblem shows in its factor charts, and it is the single most useful check for a pricing model. For each level of each rating factor, you compute the ratio of observed claims to predicted claims. A well-specified model should have A/E close to 1.0 for all levels.

```python
def ae_by_factor(
    df: pl.DataFrame,
    fitted_values: np.ndarray,
    feature: str,
) -> pl.DataFrame:
    """
    Compute actual vs expected claim counts by factor level.
    """
    return (
        df
        .with_columns(
            pl.Series("expected_claims", fitted_values)
        )
        .group_by(feature)
        .agg([
            pl.col("claim_count").sum().alias("actual_claims"),
            pl.col("expected_claims").sum().alias("expected_claims"),
            pl.col("exposure").sum().alias("exposure"),
        ])
        .with_columns(
            (pl.col("actual_claims") / pl.col("expected_claims")).alias("ae_ratio")
        )
        .sort(feature)
    )


ae_area = ae_by_factor(df, glm_freq.fittedvalues, "area")
print("A/E by area:")
print(ae_area)
```

A/E ratios close to 1.0 for area are expected - area is in the model, so the model is calibrated to it. The more important check is for **factors not in the model**. If you omit driver age and the A/E for young drivers is consistently above 1.0, the model is materially underpricing that group. Add age to the model.

```python
# Check A/E by driver age bands - driver age IS in the model (as young/old flags),
# so we should see good A/E here. Try removing it and see what happens.
df_diag = df.with_columns(
    pl.when(pl.col("driver_age") < 25).then(pl.lit("17-24"))
    .when(pl.col("driver_age") < 35).then(pl.lit("25-34"))
    .when(pl.col("driver_age") < 50).then(pl.lit("35-49"))
    .when(pl.col("driver_age") < 65).then(pl.lit("50-64"))
    .otherwise(pl.lit("65+"))
    .alias("age_band")
)

ae_age = ae_by_factor(df_diag, glm_freq.fittedvalues, "age_band")
print("\nA/E by age band:")
print(ae_age)
```

**A new Python concept: `pl.when().then().otherwise()`.** This is Polars' equivalent of a nested IF statement. `pl.when(condition).then(value).when(condition2).then(value2).otherwise(fallback)` evaluates each condition in order and returns the corresponding value. It runs across all 100,000 rows simultaneously.

### Overdispersion

For real UK motor data, the Poisson model will almost always be overdispersed - the deviance will be materially above the residual degrees of freedom. A deviance/df ratio above 1.3 is common; ratios above 2.0 are not unusual on books with bodily injury cover.

When this happens, you have two options:

**Quasi-Poisson:** same point estimates as Poisson, but standard errors are inflated to account for overdispersion. The relativities themselves do not change - only the confidence intervals widen. Use this when you want conservative confidence intervals but do not want to change the relativities.

```python
glm_freq_quasi = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.quasi.Quasipoisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

print(f"Quasi-Poisson dispersion estimate: {glm_freq_quasi.scale:.3f}")
print("(If > 1.0, the data is overdispersed relative to Poisson)")
```

**Negative Binomial:** models overdispersion explicitly with an additional dispersion parameter. The coefficients will differ slightly from Poisson. More appropriate when you expect genuine extra-Poisson variation in the data-generating process.

For a first migration from Emblem (which uses Poisson), quasi-Poisson is the lower-risk option. Relativities are identical to Poisson; only the standard errors change.