## Part 7: The non-conformity score - why raw residuals fail for insurance

This section explains the most important technical choice in conformal prediction for insurance data. Read it carefully before moving to the calibration step.

### How conformal calibration works

The conformal predictor does not change the base model. The base model still produces point predictions. What conformal calibration does is:

1. Take the calibration set (data the model has never seen)
2. Compute a "non-conformity score" for each observation - a measure of how surprising that observation is given the model's prediction
3. Sort those scores and store the `(1 - alpha)` quantile

When predicting intervals for a new observation, the predictor asks: "what outcome range would have a non-conformity score below the stored quantile?" That range becomes the prediction interval.

The coverage guarantee follows from the sorting step: if calibration and test observations are exchangeable, a new test observation's score is equally likely to be any rank in the combined distribution. So the probability that its score falls below the stored quantile is at least `1 - alpha`.

### Why raw residuals produce wrong intervals for insurance

The simplest non-conformity score is the absolute residual: `|y - ŷ|`. This fails for insurance data.

Insurance losses are right-skewed and heteroscedastic. A risk with predicted pure premium £500 has more absolute variance than a risk with predicted pure premium £50. The natural variance of insurance losses scales with the mean: this is exactly what the Tweedie variance model captures. A miss of £100 on a £100-risk is a 100% error. A miss of £100 on a £1,000-risk is a 10% error. The raw residual treats them identically.

The consequence: the calibration quantile is set primarily by the majority of low-risk policies (there are more of them and they have smaller absolute residuals). When applied to high-risk policies, the same fixed-width interval is too narrow. The high-risk policies have genuine residuals that are larger in absolute terms, but the calibration quantile reflects the scale of the low-risk majority.

On typical UK motor data, raw residual intervals achieve approximately 90% overall coverage but only 72-75% coverage in the top risk decile. The aggregate number looks fine. The top decile - where reserves are concentrated - is seriously under-covered.

### The Pearson-weighted score

The `pearson_weighted` score normalises by the predicted Tweedie variance:

```
score = |y - ŷ| / ŷ^(p/2)
```

For Tweedie `p=1.5`, this divides by `ŷ^0.75`. This is the Pearson residual for a Tweedie model - it removes the mean-variance relationship and produces scores that are approximately homoscedastic across risk levels. A miss of £100 on a £100-risk and a miss of £1,000 on a £1,000-risk produce similar scores, reflecting similar fractional errors.

The result: interval widths scale with the predicted loss level. A £500-predicted risk has a wider absolute interval than a £50-predicted risk, which is correct - the genuine uncertainty is larger. Coverage is approximately flat across deciles because the calibration quantile is now set at the right scale for every risk level.

The `tweedie_power=1.5` parameter you pass to the predictor must match the `variance_power=1.5` you used when training the model. If they differ, the Pearson normalisation uses the wrong exponent and coverage will be wrong in ways that may not be obvious from the marginal coverage alone.