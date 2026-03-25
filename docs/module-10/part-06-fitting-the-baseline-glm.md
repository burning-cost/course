## Part 6: Fitting the baseline GLM

The CANN needs the GLM's predicted frequencies as input. We fit the baseline Poisson GLM — main effects only, no interactions — using `glum`.

```python
from glum import GeneralizedLinearRegressor
import pandas as pd

# Convert to pandas for glum
X_pd = X.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_pd[col] = pd.Categorical(X_pd[col].astype(str))

# Fit baseline Poisson GLM
glm_base = GeneralizedLinearRegressor(
    family="poisson",
    alpha=0.0,        # No regularisation for now
    fit_intercept=True,
)
glm_base.fit(X_pd, y, sample_weight=exposure_arr)

# Get the GLM's fitted frequencies (on response scale, not log)
mu_glm = glm_base.predict(X_pd)

print(f"Baseline GLM fitted values — min: {mu_glm.min():.4f}, max: {mu_glm.max():.4f}")
print(f"Sum of fitted values: {mu_glm.sum():.1f} vs observed: {y.sum():.1f}")
```

**What to check:** The sum of fitted values should be very close to the sum of observed claims. This is the Poisson GLM constraint: total predicted claims equal total observed claims on the training data. If they differ by more than 0.1%, the GLM has not converged.

### Compute baseline deviance

```python
def poisson_deviance(y_true, y_pred, weights):
    """Poisson deviance: 2 * Σ w * (y*log(y/μ) - (y - μ))"""
    mu = np.clip(y_pred, 1e-8, None)
    log_term = np.where(y_true > 0, y_true * np.log(y_true / mu), 0.0)
    return 2.0 * np.sum(weights * (log_term - (y_true - mu)))

base_deviance = poisson_deviance(y, mu_glm, exposure_arr)
# glm_base.coef_ excludes the intercept; add 1 for parameter count
base_n_params = len(glm_base.coef_) + 1
base_aic = base_deviance + 2 * base_n_params
print(f"Baseline GLM deviance: {base_deviance:,.1f}")
print(f"Baseline GLM AIC:      {base_aic:,.1f}")
print(f"Number of parameters:  {base_n_params}")
```

Note: `glm_base.coef_` in glum contains one entry per feature level (after encoding categoricals), but does not include the intercept. The intercept is stored separately in `glm_base.intercept_`. Always add 1 when counting total model parameters.

Keep a note of these numbers. At the end of the module we will compare the interaction-enhanced GLM against these values.
