# Module 9: Demand Elasticity

Module 7 gave you a rate optimiser. It needed a demand model as an input — a number telling it how renewal probability responds to price changes. The number came from somewhere. In most UK pricing teams, that somewhere is a consultant benchmark, a judgment call, or a naive logistic regression. All three are worse than they look.

This module teaches you to estimate the price coefficient properly. The method is Double Machine Learning (Chernozhukov et al., 2018), a causal inference technique that accounts for the confounding structure of insurance pricing data. The naive regression is systematically biased in a predictable direction. DML corrects it.

By the end you will have:

- Understood exactly why the naive regression is biased, and by how much
- Run the pre-flight diagnostic that tells you whether your data has enough price variation to identify elasticity
- Fitted a causal elasticity estimator on UK motor renewal data using `RenewalElasticityEstimator`
- Estimated heterogeneous elasticity — the per-customer price sensitivity — using CausalForestDML
- Built an elasticity surface showing how price sensitivity varies by NCD band, age, and channel
- Run a per-policy profit-maximising optimisation subject to the PS21/5 ENBP constraint
- Produced a compliance audit trail suitable for an FCA section 166 review

The core library is `insurance_causal.elasticity`. We also use `insurance_causal.elasticity.diagnostics` for the pre-flight check, `insurance_causal.elasticity.surface` for the segment summary, and `insurance_causal.elasticity.optimise` for the per-policy optimisation.

[Download the notebook for this module](notebook.py)

---
