## Part 3: The confounding problem

Before writing any model code, we need to understand why the naive approach is wrong. This is the most important concept in the module. Analysts who skip this and jump to the regression often end up with an elasticity estimate that is biased by a factor of 2 or more.

### Why price and risk are correlated

In insurance, the quoted price is set by the underwriting system. The underwriting system takes the customer's risk features - age, vehicle group, NCD years, postcode, etc. - and produces a technical premium. The commercial loading (the ratio of quoted price to technical premium) may be relatively flat, but the absolute quoted price varies enormously across risk classes.

This means: in a quote dataset, high-risk customers receive high prices. And high-risk customers may also have different price sensitivity. Young drivers on a PCW who face a quote of £1,800 have fewer alternatives than middle-aged drivers facing £600 - some car insurance is genuinely hard to get for a 19-year-old with a sports car. Their conversion rate is lower not just because of the higher price, but because they have nowhere else to go at any comparable price.

The naive regression of conversion on quoted price sees: "higher prices are associated with lower conversion rates." It estimates a negative coefficient. But the coefficient is too large in absolute value (or sometimes in the wrong direction for subsets) because it is conflating two effects:

1. The genuine causal price effect: higher price causes lower conversion
2. The risk composition effect: high-risk customers face both higher prices and lower conversion rates for structural reasons

This is the confounding problem. The confounder is the risk class. It affects both the price (through the rating system) and the outcome (through market alternatives and risk preferences).

### A concrete example of the bias

Let us make this precise. Suppose we have two risk segments:

- Segment A (low risk): quoted price £400, 20% conversion rate
- Segment B (high risk): quoted price £900, 8% conversion rate

A naive regression would estimate that the price coefficient is roughly (8% - 20%) / (£900 - £400) = -0.024 percentage points per pound. But this mixes the risk composition effect with the price effect.

Within Segment A only, if you vary the price between £360 and £440, the conversion rate might go from 22% to 18% - a genuine price elasticity of about -2.0 on the log scale. Within Segment B only, if you vary the price between £850 and £950, the conversion rate might only go from 9% to 7% - a smaller elasticity of about -1.2. High-risk customers are less elastic, not more.

The naive regression gives you a blend of both effects. The within-segment price response and the between-segment risk composition are inseparable without the right method.

### How Double Machine Learning fixes it

Double Machine Learning (DML, Chernozhukov et al. 2018) separates the two effects by residualising. The procedure is:

**Step 1.** Fit a model predicting the outcome (conversion or renewal) from the confounders only - the risk features. Call the residuals `Y_tilde`.

**Step 2.** Fit a model predicting the treatment (log price ratio) from the same confounders. Call the residuals `D_tilde`.

**Step 3.** Regress `Y_tilde` on `D_tilde`. The coefficient is the causal price elasticity.

The key insight: after removing the part of price variation explained by risk features (Step 2), what remains in `D_tilde` is the variation driven by commercial decisions - rate reviews, loadings, A/B tests. This residual variation is approximately exogenous with respect to the confounders, because it is not a function of the individual customer's risk. The regression in Step 3 therefore picks up only the genuine price-demand relationship.

Cross-fitting (running the nuisance models on held-out data) ensures that overfitting in Steps 1 and 2 does not bias Step 3. We use 5-fold cross-fitting throughout this module.

We will see this in practice in Parts 6-8. First we need to set up the data.