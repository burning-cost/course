## Part 16: What we have not covered

This module built a working monitoring framework for a deployed CatBoost frequency model. Before moving to the summary, here is what we left out and why.

### Severity monitoring

We focused entirely on frequency. Severity monitoring uses the same tools (A/E ratio, PSI on predicted severity, Gini drift on a ranked severity model) but has three complications we did not address:

**Claims development.** Severity in motor is not finalised at notification. A bodily injury claim notified in month 1 may still be paying in year 5. Monitoring severity on incurred-to-date values understates the true severity for recent accident years. A development factor adjustment is needed before computing A/E on severity. This requires IBNR methodology that is beyond the scope of this module.

**Exposure measurement for severity.** The A/E denominator for severity is the predicted severity given a claim, not the claim count. The "exposure" for each claim is 1 claim, not the policy exposure in years. The `ae_ratio_ci()` function handles this correctly when you pass `exposure=None` (each observation has unit exposure), but the interpretation changes.

**Outlier claims.** Severity has a heavy right tail. A single catastrophic injury claim can move portfolio-level severity A/E by 10 percentage points or more. Before concluding that elevated severity A/E represents concept drift, check whether a small number of large claims are driving the result. Standard practice is to cap claims at the 95th or 99th percentile for monitoring purposes, with the capped result reported alongside the uncapped.

### Multivariate drift

PSI and CSI are univariate — they look at one feature at a time. Multivariate drift, where the combination of features has changed even if each marginal distribution is stable, is harder to detect. A portfolio that has gained policies with both young drivers and high-performance vehicles simultaneously is a different risk than the sum of each feature shift suggests, but univariate CSI would not catch this interaction.

Detecting multivariate drift requires either supervised methods (train a classifier to predict whether a policy is from the reference or current period) or density ratio estimation methods. These are research-level tools that are not yet standard in insurance pricing practice in the UK. We mention them so you know they exist; we did not implement them because the tooling is not mature enough to recommend for production use.

### Monitoring for individual segments separately

We showed segment A/E breakdowns but did not build a monitoring system that tracks PSI and Gini separately for, say, the young driver segment and the commercial fleet segment. A portfolio with very different sub-books may need separate monitoring cadences and thresholds for each sub-book.

This is straightforward to implement by running the monitoring pipeline separately for each segment filter, but it multiplies the number of tables and alerts. At the point where you have more than three or four materially distinct sub-books, consider whether separate pricing models (one per sub-book) are more appropriate than a single model with segment-level monitoring overlays.

### Explainability drift

SHAP values change as the model is retrained or as the feature importance landscape shifts. Monitoring whether SHAP-based relativities have changed is a way of detecting whether the model's interpretation of a feature has drifted — not just whether the distribution of that feature has changed.

This is useful for actuarial sign-off: if the model previously showed a monotonically increasing risk with age up to 25 and that relationship has weakened or reversed, you want to know. `insurance-monitoring` does not currently include a SHAP drift calculator. If you need this, compute SHAP values for the reference and current periods separately (using the `shap` library, which integrates with CatBoost) and compare the feature importance rankings and value distributions manually.

```python
import shap

# Reference period SHAP
explainer = shap.TreeExplainer(model)
shap_ref = explainer.shap_values(X_ref)

# Current period SHAP
shap_cur = explainer.shap_values(X_cur)

# Compare mean absolute SHAP by feature
feature_importance_ref = np.abs(shap_ref).mean(axis=0)
feature_importance_cur = np.abs(shap_cur).mean(axis=0)

for i, feature in enumerate(feature_names):
    change = feature_importance_cur[i] - feature_importance_ref[i]
    pct_change = change / feature_importance_ref[i] * 100
    print(f"{feature:<30} ref={feature_importance_ref[i]:.4f}  "
          f"cur={feature_importance_cur[i]:.4f}  "
          f"change={pct_change:+.1f}%")
```

A feature that was a top-5 driver in the reference period and has dropped out of the top 10 in the current period is a signal worth investigating.

### Real-time monitoring

Everything in this module runs monthly as a batch job. For a real-time pricing engine (quoting in seconds), monthly batch monitoring may not catch problems fast enough. Real-time monitoring would stream incoming quote and claim events and compute running metrics with narrow time windows (daily or weekly rather than monthly).

Databricks Structured Streaming can handle this. The architecture would use Delta Live Tables rather than batch Delta writes, and Databricks SQL Alerts would trigger on the streaming table rather than the monthly batch table. This is a significant architectural step up and requires experience with streaming pipelines. We do not cover it here.

### Champion-challenger monitoring

A common deployment pattern is champion-challenger: the champion model prices all risks; a challenger model prices a random sample (typically 5-10%) to generate live performance data before a full rollout. Monitoring a champion-challenger setup requires comparing the A/E ratios and Ginis of two models on overlapping policy populations — more complex than the single-model setup we built here.

For champion-challenger Gini comparison, `gini_drift_test()` accepts the champion as the reference period and the challenger as the current period. Pass the champion's actuals and predictions as `reference_actual/reference_predicted` and the challenger's as `current_actual/current_predicted`. The test then asks: has the challenger's discrimination improved or worsened relative to the champion, at the 95% or 32% confidence level?

### Threshold calibration

The PSI thresholds (0.10 amber, 0.25 red) and A/E thresholds ([0.95, 1.05] green) in `MonitoringReport` are defaults based on credit scoring conventions and actuarial practice. They are not calibrated to your specific portfolio.

A small portfolio (say, 2,000 policies with 80 expected claims per month) will have wide confidence intervals on the A/E ratio. The A/E might reach 1.15 in a bad month due to pure random noise, with the CI comfortably containing 1.0. The default amber threshold would not fire, correctly. But a large portfolio (50,000 policies, 2,000 expected claims per month) with the same A/E of 1.15 has a very narrow CI that definitively excludes 1.0. The default thresholds will fire — also correctly.

The `AERatioThresholds` uses point estimate bands (not CI) — but as portfolio size grows, the CI narrows and the point estimate A/E converges to the true ratio, so an A/E of 1.15 on a large book is a real signal regardless of the CI. The PSI threshold does not depend on portfolio size in the same way (PSI is a relative measure), so the 0.10/0.25 default is a reasonable starting point regardless of book size. You may want to tighten the A/E thresholds for a large book where even small A/E deviations represent material mis-pricing.

The right way to calibrate thresholds is to look at historical monitoring runs for your specific book and identify the natural noise level in each metric before any real drift occurred. That gives you a baseline against which to set meaningful alert levels. We did not do this in the module because it requires a live book with a monitoring history — you cannot do it on synthetic data.
