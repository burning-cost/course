## Part 18: Limitations to document

Every time you present GBM relativities to a pricing committee or regulator, you need to be explicit about what the intervals do and do not tell you. Write these into your committee paper, not just your notebook.

**Confidence intervals cover data uncertainty only.** The 95% CI on the area F relativity tells you how precisely the data pins down the mean SHAP contribution given the 10,000 area F policies you have. It does not tell you whether a different training run of the GBM would produce the same relativity. Two bootstrap refits of the model will give slightly different SHAP values. The library does not quantify this model uncertainty - be explicit when you say "95% confidence interval."

**SHAP attribution for correlated features is not unique.** If `vehicle_group` and `driver_age` are correlated (older drivers tend to drive lower-group vehicles in the real world), some of each feature's true attribution may be allocated to the other under `tree_path_dependent`. The total effect of the correlated cluster is correct; the individual feature attributions are approximate.

**Log-link only.** `exp(SHAP)` gives multiplicative relativities only for log-link objectives: Poisson, Gamma, Tweedie. Do not apply this methodology to a linear-link model.

**Risk relativities only.** These are pure risk relativities. They do not include expense loadings, profit margins, or reinsurance costs. Every slide showing these tables should say "risk only relativities."

**Interaction effects.** A vehicle group × driver age interaction is partly attributed to each feature's SHAP values. The vehicle group relativity from the GBM is not directly comparable to the GLM's vehicle group coefficient if the GLM has no interaction term - the GBM's estimate carries its share of the interaction.

---

## Summary

The workflow in five steps:

1. `sr = SHAPRelativities(model, X, exposure, categorical_features=..., continuous_features=...)`
2. `sr.fit()` - computes SHAP values via CatBoost's native TreeSHAP (15-60 seconds for 100k rows)
3. `sr.validate()` - run this before trusting any output; stop if reconstruction fails
4. `sr.extract_relativities(normalise_to="base_level", base_levels=...)` - get the factor table
5. `sr.extract_continuous_curve(feature, smooth_method="loess")` + manual aggregation by band - get banded continuous features

The output `rels` DataFrame has columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`. Every number needed to explain, challenge, and defend the relativities to a pricing committee is in that table.

The GBM finds things the GLM cannot - the driver age U-shape, the young driver × high vehicle group interaction, non-linear NCD effects in some portfolios. SHAP makes those findings visible, auditable, and presentable. That is how the GBM gets out of the notebook and into production.