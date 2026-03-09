## Part 2: What makes insurance pipelines different

Before building the pipeline, you need to understand why a generic ML pipeline template -- the kind used for image classification or fraud detection -- does not work for insurance pricing without modification.

### Time is not just a feature

In most ML problems, the train-test split is random. Observations are exchangeable: a record from day 1 and a record from day 100 are drawn from the same distribution and either can validly appear in train or test. This is false for insurance. A policy written in 2022 and a policy written in 2025 are not exchangeable. Loss cost trends, book mix, regulatory changes, economic conditions all mean the 2025 policy is in a genuinely different environment. A random split means some 2025 policies appear in training and some 2022 policies appear in validation. The model sees future data during training and the validation measures performance on past data. Both are wrong.

Every train-test split in an insurance pipeline must be temporal. The most recent period -- typically one accident year -- is the test set. Everything before it is training. This is not optional.

### Exposure is not a weight

A policy with 0.5 years of exposure and 1 claim has an observed frequency of 2.0. A policy with 1.0 year of exposure and 1 claim has an observed frequency of 1.0. They have the same raw claim count but very different information content. The 1.0-year policy is more reliable.

The correct way to handle this in a Poisson regression is with a log-exposure offset. The model predicts claim count, and the log of exposure enters as a fixed term in the linear predictor:

```sql
log(E[claims]) = log(exposure) + f(features)
```

In CatBoost, this is implemented with `baseline=np.log(exposure)` in the Pool constructor. The `baseline` parameter adds a fixed term to the model's output before the loss is computed -- it is the log-offset.

Using `weight=exposure` instead is wrong. With `weight`, the observation contribution to the loss is scaled by exposure, but the model still predicts `exp(f(features))` with no offset. A policy with 0.5 years of exposure contributes half as much to the likelihood, but the model's prediction for it is not adjusted for the shorter exposure period. The predictions are on the wrong scale.

Module 3 explained this. This module implements it correctly throughout. Every Pool that includes exposure uses `baseline=np.log(exposure)`.

### Severity needs its own model

A pure premium model (frequency times severity) could in principle be fit as a single Tweedie model. For a capstone pipeline, we fit frequency and severity separately because:

1. The Poisson frequency model has a log-exposure offset. The Gamma/Tweedie severity model does not -- severity is conditioned on a claim occurring, so exposure is not the right offset for the severity prediction.

2. SHAP relativities are more interpretable from the separate models. The frequency SHAP shows what drives claim occurrence; the severity SHAP shows what drives claim size. These tell different stories.

3. Conformal intervals are typically calibrated on severity predictions, where the skewed distribution makes uncertainty quantification most valuable.

The severity model trains only on policies with at least one claim. A policy with zero claims has no observed severity, and its inclusion would contaminate the likelihood. When we compute the pure premium, we predict severity for ALL policies (not just those with claims). The severity prediction gives the expected loss given a claim occurs -- it does not require the policy to have had a claim in training.

### The IBNR problem

Insurance claims develop over time. A claim notified in November 2024 may still be receiving payments in 2027. The incurred value as of the training data extract date is not the ultimate value.

If you include the most recent accident years at face value in training data, you are training on partially developed claims. For property damage motor, claims develop quickly -- 95% of ultimate within 12 months. For motor bodily injury, development can run 5-10 years. Training on immature claims produces a model that will systematically under-predict severity for similar future claims, because the future claims will develop to a higher ultimate value.

The IBNR buffer in walk-forward cross-validation addresses this by excluding the most recent N months from each training fold. This forces the model to rely on more developed claims data for training. The buffer length depends on the line of business.