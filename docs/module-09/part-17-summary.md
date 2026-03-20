## Part 17: Summary

This module covered:

1. Why the risk model is not sufficient for commercial pricing, and what demand modelling adds
2. The confounding problem in naive regression of conversion on price
3. Building and validating conversion and retention models (logistic and CatBoost backends)
4. The near-deterministic price problem and how to diagnose it before fitting any elasticity model
5. Double Machine Learning for causal price elasticity estimation
6. Heterogeneous CATE estimation via CausalForestDML
7. The elasticity surface: visualising price sensitivity across two dimensions
8. Building a portfolio demand curve
9. Per-policy profit-maximising optimisation subject to the PS21/5 ENBP constraint
10. The ENBP compliance audit at the individual policy level
11. Connecting the demand model to the rate optimiser from Module 7
12. Practical considerations for production deployment

The core methodological lesson is that the naive regression approach to elasticity estimation is systematically biased in insurance data. The bias is predictable in direction (the naive estimate is typically more negative than the true elasticity) and can be corrected using DML. Running the treatment variation diagnostic before every elasticity estimation is not optional - it is the test that confirms the data is good enough to use.

The FCA context means that these tools have regulatory as well as commercial value. A causal elasticity model with an audit trail is a better answer to a section 166 review than a judgment-based coefficient. The ENBP audit at the per-policy level satisfies the FCA's expectation of individual-level compliance checking, not just average compliance.

---

## Libraries used in this module

| Library | Version | Purpose |
|---------|---------|---------|
| `insurance-optimise` (via `insurance_optimise.demand`) | latest | Conversion, retention, and global elasticity models |
| `insurance-causal` (via `insurance_causal.elasticity`) | latest | Heterogeneous elasticity with CausalForestDML, optimiser, audit |
| `catboost` | 1.2+ | Nuisance models in DML, conversion and retention classifiers |
| `econml` | 0.15+ | CausalForestDML estimator |
| `polars` | 0.20+ | Data manipulation |
| `numpy` | 1.26+ | Numerical operations |
| `matplotlib` | 3.7+ | Elasticity surface and demand curve plots |

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.
- Athey, S. & Wager, S. (2019). Estimating treatment effects with causal forests. *Annals of Statistics*, 47(2), 1148-1178.
- Guven, M. & McPhail, M. (2013). Beyond the cost model: demand modelling for P&C pricing. *CAS Forum*.
- Guelman, L. & Guillén, M. (2014). A causal inference approach to measure price elasticity in automobile insurance. *Expert Systems with Applications*, 41(2), 387-396.
- FCA PS21/5 (2021). General Insurance Pricing Practices Policy Statement.
- FCA EP25/2 (July 2025). Evaluation of GIPP Remedies.