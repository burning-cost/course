## Part 10: The likelihood-ratio test

The NID gives us a ranked shortlist. The likelihood-ratio test tells us which candidates are statistically significant.

### What the LR test does

For each candidate pair (i, j), we refit the GLM with an interaction term added and compute the change in deviance:

```bash
Δdeviance = deviance(base GLM) - deviance(GLM + interaction)
```

Under the null hypothesis (no interaction effect), `Δdeviance` follows a chi-squared distribution with `n_cells = (L_i - 1)(L_j - 1)` degrees of freedom. A large `Δdeviance` relative to the chi-squared critical value means the interaction significantly improves the fit.

### The n_cells parameter

`n_cells` is the parameter cost of adding the interaction:

- Categorical × categorical: `(L_i - 1)(L_j - 1)` — the number of new parameters in the model
- Categorical × continuous: `L_i - 1` — one slope per non-baseline category level
- Continuous × continuous: 1 — a single product term

A 6-level age band interacting with a 5-level vehicle group band has `5 × 4 = 20` cells. A 6-level age band interacting with continuous annual mileage has 5 cells. These differences matter for the chi-squared test (higher degrees of freedom means a higher bar for significance) and for model complexity.

### Bonferroni correction

We are testing 15 pairs simultaneously. If we use a naive p-value threshold of 0.05, we expect 0.05 × 15 = 0.75 false positives by chance alone. Bonferroni correction divides the threshold by the number of tests:

```bash
threshold = 0.05 / 15 = 0.0033
```

A pair is `recommended = True` only if its p-value falls below 0.0033. This is conservative but correct: we are building a model that will price insurance, and a spurious interaction that gets added to a production GLM costs more than a genuine interaction that gets missed.

```python
# The full interaction table with LR test results
table = detector.interaction_table()
print("Full interaction table (top 15 NID candidates, tested):")
print(
    table.select([
        "feature_1", "feature_2",
        "nid_score_normalised",
        "n_cells",
        "delta_deviance", "delta_deviance_pct",
        "lr_p", "recommended",
    ])
)
```

**What you should see:**

- `age_band × vehicle_group` should show `delta_deviance` of several hundred to a few thousand, `lr_p` < 0.001, `recommended = True`
- `ncd_years × has_convictions` should also be `recommended = True`
- Several other pairs may have low NID scores and fail to reach significance after Bonferroni correction

```python
# Summary: how many recommended?
n_recommended = table.filter(pl.col("recommended") == True).height
print(f"\n{n_recommended} interactions recommended out of {table.height} tested")
print(f"Bonferroni threshold: {0.05 / table.height:.5f}")
```