## Part 3: The confounding problem

### Why price and risk are correlated

In insurance pricing, the quoted price is set by the rating system. The rating system takes the customer's risk characteristics — age, NCD years, vehicle group, postcode — and produces a technical premium. The quoted price is a function of the technical premium. Therefore the quoted price is, approximately, a function of the risk characteristics.

This creates a problem for any regression of renewal outcome on quoted price. High-risk customers receive high prices. High-risk customers may also have different price sensitivity from low-risk customers, for reasons that have nothing to do with price level. A 22-year-old driver who is quoted £1,800 for annual motor insurance has fewer market alternatives than a 45-year-old quoted £600 — some segments genuinely have thin alternative supply. Their renewal rate is lower not only because of the high price, but because of their risk profile and the structure of the market they are in.

The naive regression of renewal probability on quoted price sees: "higher prices are associated with lower renewal rates." It estimates a negative coefficient. But that coefficient is too large in magnitude because it conflates two effects:

1. The genuine causal price effect: a higher price causes a lower renewal probability
2. The risk composition effect: high-risk customers face high prices *and* have lower renewal rates for structural reasons unrelated to the price itself

The confounder is the risk class. It causes both the price (through the rating system) and the outcome (through market structure and product availability). Conditioning on price without conditioning on risk gives a biased estimate of the price effect.

### A concrete example

Two risk segments in a UK motor book:

| Segment | Quoted price | Renewal rate |
|---------|-------------|--------------|
| Low risk (NCD 5, age 45) | £420 | 82% |
| High risk (NCD 0, age 22) | £1,100 | 61% |

A naive regression estimates the price coefficient as roughly:

```
(61% - 82%) / (£1,100 - £420) = -0.031 percentage points per pound
```

But within the low-risk segment, varying the price between £380 and £460 might move renewal rate from 84% to 80% — a semi-elasticity of about -1.5 on the log scale. Within the high-risk segment, varying price between £1,000 and £1,200 might move renewal rate from 63% to 59% — a different semi-elasticity of about -1.0. (High-risk customers tend to be less elastic because their alternatives are more constrained.)

The naive regression gives you a blend of the within-segment price response and the between-segment risk composition effect. The two are inseparable without conditioning on the risk factors.

### How Double Machine Learning fixes it

Double Machine Learning (DML, Chernozhukov et al. 2018, *Econometrics Journal* 21(1): C1–C68) separates the two effects by residualising both the outcome and the treatment on the confounders.

**Step 1.** Fit a model predicting the outcome (renewal indicator Y) from the confounders X — the risk factors. Call the residuals Ỹ = Y − E[Y|X].

**Step 2.** Fit a model predicting the treatment (log price change D) from the same confounders. Call the residuals D̃ = D − E[D|X].

**Step 3.** Regress Ỹ on D̃. The coefficient θ is the causal price semi-elasticity.

The key insight: after removing from D the part explained by risk factors, what remains in D̃ is the price variation driven by commercial decisions — rate review loadings, underwriting overrides, A/B pricing tests. This residual variation is approximately exogenous with respect to the confounders, because it was not determined by the individual customer's risk profile. The regression in Step 3 therefore picks up only the genuine price-to-demand relationship.

Cross-fitting (running nuisance models on held-out folds) ensures that overfitting in Steps 1 and 2 does not transmit bias into Step 3. We use 5-fold cross-fitting throughout.

The crucial practical question is whether there is enough residual variation in D̃ to identify θ. If the re-rating system nearly perfectly predicts price changes from risk factors, D̃ has near-zero variance and the coefficient is numerically unreliable — the insurance analogue of a weak instrument. We diagnose this before fitting anything.
