## Part 12: Gotchas when moving from Emblem

We have worked with several teams on this migration. These are the problems that actually bite people.

**Emblem's automatic base level selection.** Emblem picks the most credible level as the base - usually the highest-exposure level. statsmodels picks alphabetically. Forgetting to align these is the number one source of "why don't the numbers match."

**Emblem's missing value handling.** Emblem treats missing values as a separate level, "Unknown," and estimates a relativity for it. statsmodels drops rows with missing values unless you handle them explicitly. If 3% of your policies have missing vehicle group and Emblem is pricing them as "Unknown" while Python is dropping them, your models are fit on different data.

```python
# Check for missing values before fitting
missing_report = df.null_count()
print(missing_report)

# Options:
# 1. Impute with the mean or mode for continuous variables
df = df.with_columns(
    pl.col("vehicle_group").fill_null(strategy="mean").cast(pl.Int32)
)

# 2. Create an "Unknown" level for categorical factors
df = df.with_columns(
    pl.col("area").fill_null("Unknown")
)
```

**Credibility-weighted relativities.** Emblem has a credibility option that shrinks sparse level relativities toward 1.0. By default in statsmodels, you get maximum likelihood estimates with no shrinkage. If Emblem's published relativities show NCD=6 at 1.000 while your Python estimate is 0.78 with a wide CI, Emblem may have applied credibility weighting.

**The deviance statistic and likelihood ratio tests.** Emblem reports "change in scaled deviance" when you add a factor. statsmodels reports total deviance. To get the chi-squared test for adding a factor, compute the deviance difference manually:

```python
# Fit base model and extended model
glm_base = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0))",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

glm_extended = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0)) + vehicle_group",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

# Likelihood ratio test for adding vehicle_group
from scipy import stats as scipy_stats

lr_stat = glm_base.deviance - glm_extended.deviance
df_diff = glm_base.df_resid - glm_extended.df_resid
p_value = scipy_stats.chi2.sf(lr_stat, df_diff)
print(f"LR chi-squared: {lr_stat:.2f}, df: {df_diff}, p-value: {p_value:.4f}")
```

---

## Summary

The Python GLM workflow produces output that is numerically consistent with Emblem when given the same data and the same specification. On synthetic data without manual overrides, the relativities match to four decimal places. On real Emblem models, validate any overrides explicitly before declaring a match.

The difference between Emblem and Python is not in the model. It is in the surrounding infrastructure: version control, reproducibility, auditability, and integration with the rest of the modelling stack.

**The critical steps, in order:**

1. Get the exposure right. Use earned exposure, clip at a small positive number, filter out zeros.
2. Match the base levels to Emblem's explicitly. Do not rely on defaults.
3. Handle missing values deliberately - decide between dropping, imputing, or creating an "Unknown" level.
4. Truncate large losses before the severity GLM. Document the cap.
5. Exclude NCD from the severity formula - it is a frequency signal, not a severity driver.
6. Check for aliased parameters and non-convergence before trusting results.
7. Run A/E diagnostics for factors not in the model as well as those that are.
8. Check deviance/df - if materially above 1, consider quasi-Poisson or negative binomial.
9. Validate against Emblem's published relativities on matched data, accounting for any manual overrides.
10. Log everything to MLflow and Unity Catalog before exporting to Radar.

The model is the easy part.

---

## What's next

**Module 3: Gradient Boosted Models with CatBoost** - replaces the GLM frequency model with a CatBoost model. Covers hyperparameter tuning, cross-validation designed for insurance data, and model comparison against the GLM benchmark from this module.

**Module 4: Validation and Monitoring** - builds the monitoring infrastructure to track model performance month by month, detect drift, and generate the FCA evidence pack automatically.