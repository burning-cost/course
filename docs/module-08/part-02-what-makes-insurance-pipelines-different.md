## Part 2: What makes insurance pipelines different

A generic ML pipeline template — the kind used for recommendation systems or fraud scoring — does not map directly onto insurance pricing. There are three structural differences that require explicit handling.

### Time is not a feature, it is a split boundary

In most ML problems, observations are exchangeable: a record from January and a record from December are drawn from the same distribution and either can validly appear in train or test. A random 80/20 split is fine.

Insurance observations are not exchangeable across time. A policy from 2021 and a policy from 2025 existed in different regulatory environments (Consumer Duty, enacted July 2023), different economic conditions (claims inflation peaked at 32% in UK motor in 2022-23), and potentially different book compositions. A random split means some 2025 policies appear in training — the model sees future information. A 2021 validation set measures performance on past conditions. Both directions are wrong.

Every train-test split in an insurance pipeline must be temporal. The most recent accident year is the test set. The model is trained on everything before it. This is not a best-practice recommendation — it is a requirement for valid validation metrics in a temporally structured process.

Walk-forward cross-validation applies the same logic across multiple folds: train on years 1-2, validate on year 3; train on years 1-3, validate on year 4; and so on. Each fold respects the temporal boundary. The last fold's validation set is the most realistic estimate of future performance because it is the most recent out-of-time period.

### Exposure is a structural parameter, not a weight

A policy with 0.5 years of exposure and 1 claim has an observed frequency of 2.0 per policy-year. A policy with 1.0 year of exposure and 1 claim has an observed frequency of 1.0. The raw claim counts are the same. The information content is not.

The Poisson frequency model handles this with a log-exposure offset. The model predicts expected claim count — not expected frequency — and the log of exposure enters as a fixed additive term in the linear predictor:

```
log(E[claims]) = log(exposure) + f(features)
```

In CatBoost, this is the `baseline` parameter in the Pool constructor. Setting `baseline=np.log(exposure)` tells CatBoost that `log(exposure)` is a known term in the model's output, not something to learn. The model then learns `f(features)`, and `exp(log(exposure) + f(features)) = exposure * exp(f(features))` is the predicted claim count.

Using `weight=exposure` instead is wrong. It scales the observation's contribution to the loss without adjusting the model's output scale. The predictions are then claim frequencies, not claim counts — but the Pool target is claim counts. The model fits, but its predictions have the wrong units relative to a correctly specified offset model.

This distinction matters for SHAP relativities. The SHAP values from a correctly offset model give the log-additive contribution of each feature to `log(frequency)`. Exponentiating gives the multiplicative relativity — the same quantity as a GLM's `exp(beta)`. A model fit with `weight=exposure` instead of `baseline` will produce SHAP values on a different scale and the relativities will not be comparable to GLM output.

### The IBNR problem in cross-validation

Claims develop over time. A claim notified in November 2025 may still be receiving payments in 2029. The incurred value as of the training data extract date is not the ultimate value.

The IBNR buffer in walk-forward cross-validation removes the most recent N months from the trailing edge of each training fold. This forces the model to rely on claims that have had more time to develop before being used as training targets.

For UK private motor property damage, most claims are settled within 6 months and 95% within 12 months. A 6-month buffer is acceptable. For UK motor bodily injury — where litigation can run years — the buffer should be at least 12-18 months. For long-tail commercial lines (employers' liability, solicitors' PI), 24 months is the minimum to avoid material IBNR contamination in training.

The synthetic dataset in this module uses annual accident years, so the buffer's effect is limited to trimming the most recent year from training folds. In production with monthly accident periods, the buffer removes specific months from the trailing edge of each fold's training window.

### Severity needs a separate model

The alternative to a separate frequency-severity model is a Tweedie model fit on pure premium (observed claims / exposure). This is simpler and is not wrong for some purposes.

For a capstone pipeline, separate models are better for three reasons:

**SHAP interpretability.** The frequency SHAP shows what drives claim occurrence. The severity SHAP shows what drives claim size. For motor, these tell different stories: age and annual mileage dominate frequency; vehicle group and region dominate severity. A single Tweedie model mixes the two effects and makes it harder to communicate the pricing rationale to the underwriting committee.

**Conformal calibration.** Conformal intervals on the severity model require a calibration set that is held out from severity model training. If we fit one Tweedie model on all claims, the natural calibration set is the held-out accident year — but that year includes zero-claim policies where severity is undefined. The calibration set for a severity-specific conformal predictor is clean: only policies with at least one claim in the calibration year.

**Underwriting referral.** The conformal interval on severity identifies high-uncertainty risks — policies where the model's severity prediction has wide bounds. These are candidates for underwriting referral or minimum premium floors. You cannot produce this signal from a pure premium model without fitting a secondary model on the residuals.

The severit model trains only on policies with at least one claim. The frequency model trains on all policies. For scoring, the severity model predicts expected severity given a claim occurs — this is a valid prediction for any policy, whether or not it had a claim in the training data. The pure premium is then frequency prediction times severity prediction for all policies.
