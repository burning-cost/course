## Part 16: Limitations for regulatory presentation

When presenting conformal prediction intervals to regulators or a pricing committee, you must be explicit about the assumptions and limitations. Eight documented limitations follow. Document all of them, even the ones that do not apply to your specific use case - regulators and actuaries are more comfortable with a methodology that acknowledges its boundaries than one that claims to be universally applicable.

### Limitation 1: Marginal, not conditional, coverage

The coverage guarantee is marginal: at least 90% of all test observations are covered. It is not conditional: it does not guarantee 90% coverage within every risk segment. Coverage-by-decile validation tests this empirically, but passes or fails at the decile level, not at the level of individual feature combinations.

**Mitigation:** run coverage-by-decile and by key feature values (area, driver age band, vehicle group band) before using intervals for any downstream purpose.

### Limitation 2: Exchangeability assumption

The guarantee requires calibration and test data to be exchangeable (from the same distribution). Temporal drift breaks this. Inflation, regulatory changes (e.g. FCA pricing remedies in January 2022), and changes in the book composition (new distribution channels, portfolio acquisitions) all introduce distribution shift that can invalidate the guarantee.

**Mitigation:** calibrate on the most recent available data. Monitor coverage on recent live business quarterly. Recalibrate when coverage falls below 85%.

### Limitation 3: The base model must be directionally correct

Conformal calibration adjusts the width of intervals to achieve coverage, but it cannot fix a model that is systematically wrong about which risks are more or less dangerous. If the base model's risk ranking has degraded (Gini coefficient fallen substantially), wider intervals will not restore meaningful coverage-by-decile.

**Mitigation:** track the base model's Gini coefficient over time alongside coverage metrics. If coverage-by-decile deteriorates despite recalibration, and Gini has fallen, retrain the base model.

### Limitation 4: Asymmetric intervals and the CLT aggregation

Individual intervals for Tweedie models are asymmetric: the distance from the point estimate to the upper bound is much larger than the distance to the lower bound (which is clipped at zero for many policies). The CLT-based portfolio range aggregation uses a symmetric normal approximation, which overestimates the portfolio SD: the zero-capped lower bounds are treated as symmetric lower bounds, inflating the estimated portfolio standard deviation.

**Mitigation:** present the independence range as an optimistic lower bound on portfolio uncertainty — it is a lower bound on the naive bound, not a true lower bound on portfolio uncertainty. For more accurate portfolio ranges, simulate from the Tweedie distribution at the calibrated quantile scale.

### Limitation 5: Minimum calibration set size

The finite-sample correction to the coverage guarantee is `1/(n+1)`. For n=1,000, this is 0.001 - negligible. But the precision of the coverage estimate itself follows a binomial distribution with standard deviation `sqrt(alpha*(1-alpha)/n)`. For n=500 and alpha=0.10, the 95% confidence interval on coverage is approximately ±2.6pp. For small books, the calibration set may be too small to precisely verify that coverage is achieving the target.

**Mitigation:** use at least 2,000 calibration observations. For books with fewer total policies, consider using cross-conformal prediction (Exercise 2 explores the sample-size tradeoff).

### Limitation 6: Feature distribution shift (covariate shift)

Calibration was run on a specific distribution of features. If the future book has substantially different feature distribution - new markets, different vehicle groups, changed distribution channel demographics - the coverage guarantee is weakened because the calibration scores may not represent the scale of prediction error on the new feature distribution.

**Mitigation:** compare the feature distribution of the calibration set to the current book quarterly. Flag large shifts (e.g. mean vehicle group rising by 5+ points) as a trigger for recalibration.

### Limitation 7: Intervals do not account for parameter uncertainty in the base model

The base CatBoost model was trained on a specific dataset. A different training set (e.g. with different random seed or a different bootstrapped subsample) would produce different predictions. The conformal intervals account for prediction error given the fitted model but not for the uncertainty in the model parameters themselves.

**Mitigation:** for high-stakes applications (e.g. reserve range for a reinsurance negotiation), complement conformal intervals with a model uncertainty estimate from bootstrap resampling of the base model.

### Limitation 8: The pearson_weighted score assumes Tweedie is the correct distributional family

The Pearson normalisation divides by `ŷ^(p/2)`, which is the correct normalisation for a Tweedie model. If the true distribution departs substantially from Tweedie - for example, if severity is bimodal (small attritional claims plus occasional large claims with a different distribution) - the normalisation may not fully homogenise the scores, resulting in residual heteroscedasticity in coverage.

**Mitigation:** validate coverage-by-decile and additionally validate by severity quantile (do intervals covering large-claim policies achieve similar coverage to those covering small-claim policies?). If heteroscedasticity persists, try `nonconformity="deviance"` as an alternative score.