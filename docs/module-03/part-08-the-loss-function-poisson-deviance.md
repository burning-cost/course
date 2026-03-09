## Part 8: The loss function - Poisson deviance

We need a metric for comparing model performance across CV folds. RMSE is wrong for count data.

RMSE treats a miss of 2 claims on a policy with 0.1 expected claims the same as a miss of 2 claims on a policy with 3 expected claims. The first is a catastrophic relative error; the second is 67% error. RMSE weights them equally. This is not what we want for insurance pricing.

The Poisson deviance weights residuals appropriately for count data. Create a new cell and define the function:

```python
def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """
    Scaled Poisson deviance per unit exposure.

    y_true and y_pred are on the count scale (not frequency).
    We convert to frequency internally, then compute deviance per unit exposure.
    Dividing by total exposure gives a per-policy-year figure that is
    comparable across portfolios with different exposure distributions.

    Lower is better.
    """
    freq_pred = np.clip(y_pred / exposure, 1e-10, None)  # avoid log(0)
    freq_true = y_true / exposure
    deviance = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true / freq_pred), 0.0)
        - (freq_true - freq_pred)
    )
    return float(deviance.sum() / exposure.sum())
```

Run this cell. No output - you are defining a function for later use.

The formula penalises large misses more when the prediction is small. On UK motor data, a well-fitted Poisson GLM achieves deviance around 0.18-0.22 on a temporal validation set. The CatBoost model typically achieves 0.16-0.20 after tuning. The absolute values matter less than the comparison on the same test set.