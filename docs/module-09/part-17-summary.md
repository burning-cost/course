## Part 17: Summary

This module covered the full pipeline from confounding problem to FCA-compliant per-policy pricing.

**The methodology:**

The naive regression of renewal probability on quoted price is biased because risk factors drive both price and renewal behaviour. Double Machine Learning (Chernozhukov et al. 2018) removes this bias by residualising both the outcome and the treatment on the observable confounders before estimating the price coefficient. The pre-flight diagnostic — Var(D̃)/Var(D) — confirms there is enough residual price variation to identify the causal effect. Below 0.10, the result is not trustworthy.

**The estimator:**

`RenewalElasticityEstimator` wraps `CausalForestDML` from EconML with CatBoost nuisance models. It produces:
- An average treatment effect (ATE) with 95% CI: the portfolio-average semi-elasticity
- Per-customer CATEs: individual-level price sensitivity
- Group average treatment effects (GATEs) by any categorical variable
- Per-customer confidence intervals for the CATEs

**The heterogeneity:**

Price sensitivity varies substantially across the book. In a typical UK motor portfolio, NCD-0 PCW customers are 3–4× more elastic than NCD-5 direct customers. Using a single average elasticity in pricing treats these customers identically. The GATE table makes the differences explicit and confidence-interval-bounded.

**The optimisation:**

`RenewalPricingOptimiser` takes the per-customer CATEs and finds the profit-maximising renewal price for each policy, with `tech_prem` as the floor and `enbp` as the hard PS21/5 ceiling. For policies where the ENBP constraint binds — the profit-maximising price would be above ENBP — the optimiser prices at the ceiling and records the constraint as binding. The fraction of binding constraints is the quantified commercial cost of PS21/5.

**The compliance:**

`enbp_audit()` produces the per-policy compliance report for FCA ICOBS 6B.2. It is saved to a Delta table with run metadata for the audit trail. The FCA's expectation is individual-level compliance, not average-level. The audit table is the artefact you produce for a section 166 request.

---

## Libraries used in this module

| Library | Version | Purpose |
|---------|---------|---------|
| `insurance-causal` | latest | DML elasticity, diagnostics, surface, optimiser, audit |
| `catboost` | 1.2+ | Nuisance models in CausalForestDML |
| `econml` | 0.15+ | CausalForestDML estimator |
| `statsmodels` | 0.14+ | Naive logistic benchmark |
| `polars` | 0.20+ | Data manipulation throughout |
| `numpy` | 1.26+ | Numerical operations |
| `matplotlib` | 3.7+ | Surface and demand curve plots |
| `mlflow` | 2.0+ | Experiment tracking and audit trail |

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1–C68.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148–1178.
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2020/2025). Generic machine learning inference on heterogeneous treatment effects in randomized experiments. *NBER Working Paper 24678*.
- Chernozhukov, V., Newey, W., & Singh, R. (2022). Automatic debiased machine learning of causal and structural effects. *Econometrica*, 90(3), 967–1027.
- FCA PS21/5 (2021). General Insurance Pricing Practices Policy Statement.
- FCA ICOBS 6B.2 (2022). Renewal pricing rules.
- Guelman, L., & Guillén, M. (2014). A causal inference approach to measure price elasticity in automobile insurance. *Expert Systems with Applications*, 41(2), 387–396.
