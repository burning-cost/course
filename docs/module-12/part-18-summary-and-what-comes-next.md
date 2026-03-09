## Part 18: Summary and what comes next

You have completed the course. In twelve modules, you have gone from reading data from Delta tables to deploying production-ready pricing models with auditable spatial territory factors.

To recap what this module covered:

- Emblem-style postcode banding creates artificial discontinuities and does not borrow strength. BYM2 is the principled alternative.
- Spatial autocorrelation in insurance residuals is almost always present and significant at postcode sector level. Test with Moran's I before modelling.
- The BYM2 model combines an ICAR component (spatially smooth, captures regional patterns) with an IID component (captures area-specific outliers). The rho parameter tells you how much of the geographic variation is spatially structured.
- Fitting via PyMC 5 requires MCMC diagnostics: R-hat < 1.01, ESS > 400, zero divergences. Do not use output that fails these checks.
- The two-stage pipeline (base model first, then BYM2 on O/E residuals) is recommended for production. It decouples the spatial model and makes both components auditable independently.
- Territory relativities slot into a rating engine as log-offsets in a downstream GLM or as lookup factors in a multiplicative tariff.
- BYM2 reduces estimation error substantially compared to k-means banding, especially for sparse areas.

### Further reading

- Riebler, A., Sørbye, S.H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145--1165. The original BYM2 paper. Read the motivation and the rho interpretation in sections 2 and 3.
- Gschlössl, S., Schelldorfer, J., & Schnaus, M. (2019). Spatial statistical modelling of insurance risk. *Scandinavian Actuarial Journal*. Direct application to non-life insurance.
- Brockman, M.J., & Wright, T.S. (1992). Statistical motor rating: making effective use of your data. *Journal of the Institute of Actuaries*, 119, 457--543. The classic reference for UK motor territory rating. Sections 4 and 5 describe the data problems that BYM2 addresses.
- Vehtari, A., et al. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*. The source for the R-hat < 1.01 threshold we use throughout.
- insurance-spatial library documentation: [burning-cost.github.io/insurance-spatial](https://burning-cost.github.io/insurance-spatial)
- Blangiardo, M., & Cameletti, M. (2015). *Spatial and Spatio-temporal Bayesian Models with R-INLA*. Wiley. Chapter 5 covers BYM models in detail with accessible worked examples; the R-INLA examples translate directly to PyMC.