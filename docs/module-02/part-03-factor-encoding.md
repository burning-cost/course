## Part 3: Factor encoding

### Why encoding matters

This is where most GLM migrations go wrong, and it is worth spending time on before touching statsmodels.

In a GLM, every categorical factor must have a reference category - what Emblem calls the "base level." For a factor with k levels, the model estimates k-1 coefficients. The reference category has no coefficient; its effect is absorbed into the intercept. The relativity for each non-reference level is `exp(beta)` relative to the reference.

Python's statsmodels picks the base level automatically. By default, it uses the first level alphabetically or numerically. So for area, that is A. For NCD years, that is 0. For vehicle group if treated as a factor, that is 1. These happen to match Emblem's typical defaults for this dataset - but if you are not explicit about it, you are relying on a coincidence.

If your Emblem model uses area A as base and your Python model defaults to something different, every single area relativity will be off by a constant multiplier. It will not look like a coding error. The relativities will have the right shape but the wrong scale. This is the number one source of "why don't the numbers match" on Emblem-to-Python migrations.

### Encoding area as an Enum

A Polars `Enum` is a categorical column with an explicit, ordered list of allowed values. By defining the order ourselves, we control which level appears first when the data is passed to statsmodels, and therefore which level statsmodels treats as the base.

```python
# Encode area with explicit ordering - area A is first, so it becomes the base
area_order = ["A", "B", "C", "D", "E", "F"]
df = df.with_columns(
    pl.col("area").cast(pl.Enum(area_order))
)
```

Run this cell. No output is expected - it modifies the DataFrame in place. Check it worked:

```python
print(df.dtypes)
```

You should see `area` listed as `Enum(categories=['A', 'B', 'C', 'D', 'E', 'F'])`.

### Preparing for statsmodels

statsmodels requires pandas DataFrames, not Polars DataFrames. We convert at the point of model fitting and keep everything else in Polars. The reason: Polars is faster and more readable for data manipulation; statsmodels simply does not support Polars yet. The conversion is a one-liner.

We also need to add the log-exposure column here, which is the exposure offset for the frequency GLM. We will explain exactly what that is in Part 4.

```python
import numpy as np

df_pd = df.to_pandas()
df_pd["log_exposure"] = np.log(df_pd["exposure"].clip(lower=1e-6))

print(f"Converted to pandas: {df_pd.shape[0]:,} rows, {df_pd.shape[1]} columns")
```

**What the `.clip(lower=1e-6)` does:** It replaces any exposure values below 0.000001 with 0.000001 before taking the log. `log(0)` is negative infinity, which will corrupt the GLM fitting. This clip prevents that. In our synthetic data, exposures are always at least 0.05, so the clip makes no difference here. On real data it protects you from data errors.

### Checking for missing values

statsmodels drops rows with missing values silently. If your real dataset has missing vehicle groups and Emblem treats them as "Unknown" while Python drops them, you are fitting two different models on different data. Always check before fitting.

```python
print("Missing value counts:")
print(df.null_count())
```

In our synthetic data, there should be no missing values. On real data, any non-zero count here needs a decision: impute, create an "Unknown" level, or drop and document.