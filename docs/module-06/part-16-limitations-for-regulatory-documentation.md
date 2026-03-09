## Part 16: Limitations — for regulatory documentation

Regulatory and internal governance presentations require honest documentation of methodology limitations. Here are the limitations for this module's methods, in the precision expected for an FCA Consumer Duty or IFoA actuarial standards filing.

**1. Bühlmann-Straub assumes Normal errors in the working scale.** The BLUP property holds without distributional assumptions, but the quality of the EPV and VHM estimates depends on the data not being severely non-Normal. Very skewed loss rates — common when large bodily injury claims drive thin-cell volatility — inflate the EPV estimate and produce Z values that are too low (over-shrinkage). Log-transforming before applying B-S mitigates this but does not eliminate it for extreme distributions.

**2. MCMC is slow and requires diagnostic skill.** A pricing team accustomed to GLMs that complete in under a minute will find Bayesian MCMC unfamiliar. R-hat diagnostics, divergence checks, and ESS requirements are new concepts. Budget time for this learning curve and compute for exploratory runs before committing to Bayesian methods in a production pipeline.

**3. The Poisson likelihood assumes observed variance equals expected variance.** If claim count data are overdispersed (variance greater than mean) — which is common in insurance due to unobserved heterogeneity (driving behaviour, actual vehicle condition, claims propensity) — the Poisson model understates uncertainty for individual districts. The Negative Binomial likelihood is more appropriate when overdispersion is detected. In PyMC: replace `pm.Poisson` with `pm.NegativeBinomial` and add a dispersion parameter `alpha_disp = pm.HalfNormal("alpha_disp", sigma=1.0)`.

**4. Hierarchical models with few groups are poorly identified at the top level.** If you have 6 affinity schemes, the estimate of sigma — the between-group standard deviation — is itself highly uncertain. The Bayesian posterior for sigma will be wide, propagating uncertainty into all scheme-level credibility factors. This is correct behaviour, but it can be uncomfortable for stakeholders expecting precise answers. Report the posterior for sigma explicitly, not just the point estimate.

**5. Bühlmann-Straub groups must be approximately exchangeable.** The method assumes group hypothetical means are independently drawn from the same prior distribution. For UK postcodes, this assumption is violated: districts in the same urban area are correlated. KT1, KT2, and KT3 share road networks, parking conditions, crime rates, and flood risk. They are not independent draws from a common prior. Flat B-S underestimates the between-group variance in this case and over-shrinks correlated groups. The two-level hierarchical model (Exercise 2) is the correct fix for structured geographic data.

**6. Structural parameters are estimated from the same data used for credibility weighting.** This is empirical Bayes: v and a are estimated from the data and then used in the credibility formula. This introduces a subtle upward bias in Z values — the data that inform the structural parameters are the same data used to evaluate credibility, violating the independence assumption in the theoretical derivation. For portfolios with 50+ groups across 5+ years, this bias is negligible. For fewer groups or shorter histories, it may be material.

**7. The log-Normal random effect distribution may not match the true distribution of district risks.** The Bayesian hierarchical model places a Normal distribution on the log-rates, corresponding to a log-Normal distribution on the rates themselves. If the true distribution has heavier tails — for example, a handful of genuinely extreme-risk micro-areas — the log-Normal will over-shrink the extreme districts. The mitigation is to check the shrinkage plot for evidence of excessive shrinkage on extreme districts and to run a prior sensitivity check with a heavier-tailed Student-t prior: replace `pm.Normal("u_district_raw", ...)` with `pm.StudentT("u_district_raw", nu=4, ...)`.

**8. Credibility estimates are an input to the pricing decision, not the decision itself.** In production, credibility-weighted district relativities are typically capped and floored before entering the rating structure — for example, no district moves more than 50% above or below the portfolio mean in a single review cycle. This prevents any single district driving a loss-making or uncompetitive premium. If you implement capping and flooring, document it explicitly: FCA Consumer Duty requires that the methodology choices — including constraints on outputs — are recorded and explainable.

**9. Posterior predictive validation is mandatory, not optional.** A hierarchical model that passes MCMC convergence diagnostics (R-hat, ESS, no divergences) can still be misspecified. Run posterior predictive checks — simulate datasets from the posterior and compare to observed data — before presenting results. Exercise 3 covers this in detail. A model that fails posterior predictive checks is misspecified regardless of convergence, and misspecified models produce systematically biased credibility factors.

---

## Summary: what just happened

This module covered the two principled methods for handling thin cells in UK insurance pricing.

**Bühlmann-Straub credibility:**
- Derives the optimal blend of group experience and portfolio mean from first principles
- Three parameters: grand mean (mu), within-group variance (EPV, v), between-group variance (VHM, a)
- Credibility factor Z = w / (w + K), K = v/a
- Works in log-rate space for Poisson/multiplicative frameworks (`log_transform=True`)
- Fast, auditable, regulatory-friendly
- Correct for one grouping variable with many groups

**Bayesian hierarchical models:**
- Probabilistic programming with PyMC 5
- Non-centered parameterisation eliminates funnel geometry
- Three mandatory convergence checks: R-hat < 1.01, ESS > 400, zero divergences
- Outputs posterior distributions, not just point estimates — 90% credible intervals on every district rate
- Correct for multiple groupings, few groups, or when uncertainty quantification is required
- Slower, requires diagnostic skill, but more honest about uncertainty

**The two-stage workflow most UK teams need:**
1. CatBoost or GLM for main rating factors
2. Bühlmann-Straub on district-level O/E residuals

The four exercises extend this: Exercise 1 applies B-S to severity data, Exercise 2 builds a two-level geographic hierarchy in PyMC, Exercise 3 validates the Bayesian model with posterior predictive checks, and Exercise 4 formats results for a pricing committee presentation.

---

*This module can also be worked via the burning-cost `credibility` library (github.com/burning-cost/credibility) and `bayesian-pricing` library (github.com/burning-cost/bayesian-pricing) for a higher-level API. The implementations in this tutorial are the reference — they are what those libraries do internally, made explicit so you can understand and adapt them.*