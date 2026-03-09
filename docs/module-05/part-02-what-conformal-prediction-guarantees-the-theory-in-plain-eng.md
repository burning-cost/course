## Part 2: What conformal prediction guarantees - the theory in plain English

You do not need to read a statistics paper to use conformal prediction correctly. But you do need to understand what the coverage guarantee says, because the FCA may ask about its mathematical basis.

### The guarantee

Given:
- A trained model
- A calibration dataset drawn from the same distribution as future test data
- A target coverage level `1 - alpha` (e.g. 90%)

The conformal predictor produces intervals such that, over repeated test observations:

**P(y_new is inside the interval) >= 1 - alpha**

This means: if you produce 90% prediction intervals using a correctly calibrated conformal predictor, at least 90% of future observations will fall inside their intervals. The "at least" is precise: the actual coverage is guaranteed to be at least the target, not exactly the target.

### What it does NOT guarantee

The guarantee is **marginal**: it applies across all test observations combined. It does not say that 90% of young drivers will be covered, or 90% of high-vehicle-group policies will be covered. Without additional care, the intervals could achieve 99% coverage in the bottom risk decile and only 72% in the top decile, and the marginal number is still 90%.

For insurance this is unacceptable. The top decile of risks - the ones contributing most to reserve uncertainty and the most likely to generate large adverse outcomes - is where we most need reliable coverage. A reserve range that achieves 90% coverage across the whole portfolio but only 72% coverage for the largest risks is not useful for the reserving team.

The solution is the **variance-weighted non-conformity score** (specifically the Pearson-weighted score explained in Part 5), which makes intervals scale with the predicted loss level. Combined with the coverage-by-decile diagnostic (Part 7), this ensures the intervals are valid where you need them to be valid.

### The exchangeability requirement

The conformal guarantee requires that the calibration data and test data are **exchangeable**: the joint distribution of any combined sample should be invariant to permutation. In practice this means calibration data and test data must come from the same underlying distribution.

For insurance, temporal trends break this. If claims have been inflating at 8% per year, a calibration observation from 2022 and a test observation from 2024 are not exchangeable - the 2024 observation has been through two additional years of inflation. This is why we calibrate on recent business, why the data split is temporal (not random), and why recalibration is necessary when the book changes.

**The practical rule:** calibrate on the most recent 20% of your data before the test period. Recalibrate at least annually, or quarterly if your book changes quickly.

### The formal finite-sample guarantee

For a calibration set of size `n`, the exact conformal guarantee is:

**P(y_new inside interval) >= 1 - alpha - 1/(n + 1)**

The `1/(n+1)` term is the finite-sample correction. For n=1,000 observations, it is 0.001 - negligible. For n=100, it is 0.01 - still small but worth noting if your calibration set is very small. Exercise 2 explores this empirically.