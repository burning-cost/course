## Part 3: The exposure offset - the most important implementation detail

Before we build the model, we need to cover something that is easy to get wrong and hard to detect once you have got it wrong.

### What an exposure offset is

In a Poisson GLM, exposure enters as an offset: `log(mu) = Xb + log(exposure)`. The model predicts claim count `mu = exp(Xb) * exposure`. A policy with 0.5 years of exposure gets half the predicted claims of an otherwise identical policy with a full year - which is correct, because it was only exposed for half the year.

In CatBoost, the equivalent is the `baseline` parameter. The model's linear predictor is `f(X) + baseline`, and with a log link, the prediction is `exp(f(X) + baseline)`. To replicate the GLM offset, you pass `baseline=np.log(exposure)`.

### The trap

The wrong version: `baseline=exposure`. This passes the raw exposure value (typically 0.25 to 1.0 for mid-year policies) directly as the baseline. The prediction becomes `exp(f(X) + exposure)`. For a policy with 0.5 years of exposure, this multiplies the prediction by `exp(0.5) = 1.65` rather than `exp(log(0.5)) = 0.5`. The model still trains and converges - CatBoost has no way of knowing the baseline is wrong - but the predictions are systematically incorrect in a way that depends on exposure level.

Even worse: no baseline at all. This assumes every policy has exposure = 1.0. For a book where most policies have exposure below 1.0, this overestimates predicted counts uniformly and produces a calibration shift that looks like a model problem but is actually an implementation error.

The correct pattern, written once clearly so it is easy to refer back to:

```python
# exposure is in years (e.g. 0.25 to 1.0 for mid-year policies)
# CORRECT: log-transform before passing as baseline
train_pool = Pool(
    X_train,
    y_train,
    baseline=np.log(exposure_train),   # must be log(exposure), not exposure
    cat_features=CAT_FEATURES,
)
```

Exercise 1 in `exercises.md` quantifies this error empirically. We recommend doing Exercise 1 before continuing, as it builds real intuition for why this matters.