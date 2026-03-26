## Part 14: When to use which method — the decision framework

This framework assumes you have already decided that naive observed rates are not appropriate. Some form of partial pooling is needed. The question is which form.

### Use Bühlmann-Straub when:

- **One grouping variable.** Schemes, vehicle classes, or postcode districts in isolation. B-S is a one-dimensional method; it handles one grouping at a time.
- **Many groups.** At least 5, ideally 20+. With fewer groups, the estimate of between-group variance (a_hat) is unreliable. Five groups gives a very noisy a_hat, and a noisy a_hat makes K unreliable.
- **Speed matters.** B-S runs in milliseconds. PyMC takes minutes. For daily monitoring, real-time pricing, or exploratory analysis, B-S wins.
- **Regulatory transparency.** Bühlmann-Straub has a 55-year track record in actuarial methodology documentation. The FCA Consumer Duty pack can cite Bühlmann & Straub (1970) and explain Z without specialist software. A documented, auditable methodology is substantially stronger than an undocumented one, even if the undocumented approach is technically superior.
- **Downstream of a main model.** If you are applying credibility to residuals from a GLM or GBM, B-S is the natural tool.

### Use full Bayesian hierarchical models when:

- **Multiple crossed grouping variables simultaneously.** Area AND vehicle group AND NCD band, all with partial pooling. B-S handles one dimension at a time; PyMC handles arbitrary combinations.
- **Few groups.** With fewer than 10 affinity schemes, the uncertainty in the structural parameters is material. Bayesian propagates it correctly.
- **Credible intervals are required.** B-S gives point estimates of Z and the credibility estimate. Bayesian gives the full posterior distribution — 90% credible intervals, probability that a rate exceeds a threshold, etc.
- **Two-level geographic hierarchy.** Districts nest within areas. The nested model in Exercise 2 handles this correctly.
- **Proper Poisson or Gamma likelihood.** B-S is derived from a Normal observation model. Full Bayesian uses the correct likelihood for the data type.

### The two-stage approach

For most UK personal lines pricing projects, the correct architecture is:

1. **Stage 1:** CatBoost or GLM on the full dataset for main effects — driver age, vehicle group, NCD, area bands
2. **Stage 2:** Bühlmann-Straub on district-level O/E residuals from Stage 1, with `log_transform=True`

This is not a shortcut. It is the principled decomposition: the main model handles the rating factors the GLM/GBM can identify from large samples; credibility handles the district-level departures that require pooling.

**Important:** the Stage 2 district credibility factor is a multiplicative adjustment applied on top of the Stage 1 model's area factor. The district O/E factor multiplies the GLM area factor — it does not replace it. The final district-level rate is: Stage 1 base rate × GLM area factor × credibility-weighted district O/E factor.

Reserve full Bayesian for cases where Stage 2 is insufficient — multiple crossed groupings, very few groups, or where the regulator asks for confidence intervals on individual segment rates.