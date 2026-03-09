# Module 5: Conformal Prediction Intervals for Insurance Pricing

In Module 4 you extracted SHAP relativities from a CatBoost model and produced rating factor tables comparable to a GLM output. The model tells you what to charge. This module addresses a question the model does not answer on its own: how confident should you be in that number?

A CatBoost Tweedie model predicts a pure premium of £342 for a specific risk. That is the expected loss given the features. But how uncertain is that estimate? For a typical NCD-5 driver in area C with a mid-range vehicle group, the model has seen many similar risks in training and the prediction is reliable. For a 19-year-old with 9 conviction points in vehicle group 47 - a combination that appears a handful of times in the training data - the £342 point estimate is real, but the actual outcome could be anywhere from £0 to £15,000.

Most pricing workflows use the point estimate alone and ignore the uncertainty. This module shows why that is a problem for reserving, minimum premiums, and FCA Consumer Duty, and how conformal prediction provides uncertainty estimates with a mathematical coverage guarantee that you can audit and defend.

By the end of this module you will have:

- Calibrated a conformal predictor on held-out insurance data and generated 90% prediction intervals
- Validated that coverage is consistent across risk deciles, not just overall
- Flagged uncertain risks for underwriting referral using relative interval width
- Built risk-specific minimum premium floors from the conformal upper bound
- Produced a portfolio reserve range estimate with explicit assumptions
- Logged everything to MLflow and written intervals to Unity Catalog for audit

---