## Part 16: UK context -- the London market and commercial lines

### How exposure curves are used in Lloyd's syndicates

Every Lloyd's syndicate that writes property catastrophe or per-risk XL reinsurance uses exposure curves. The standard Swiss Re curves (Y1-Y4 and Lloyd's) are embedded in Lloyd's own guidance and in the pricing tools used across the market. When a syndicate quotes a per-risk XL layer, the exposure rating calculation using one of these curves is the baseline. Burning cost is used as a cross-check when experience data is sufficient.

The Lloyd's curve (c = 5.0) is specifically calibrated for the large industrial and petrochemical risks that dominate the Lloyd's property market. It was not derived from UK high street commercial property data -- it reflects the engineering characteristics of the risks that Lloyd's was built around.

Lloyd's syndicates are subject to managing agent oversight of curve selection for cat and per-risk XL submissions. Document the chosen curve and the reasoning in the underwriting file; the managing agent's actuarial team will review it.

For UK commercial lines insurers (as opposed to reinsurers), exposure curves enter the picture in two scenarios:

1. **Limit adequacy assessment**: when a commercial lines insurer wants to check whether its limits are adequate for the risk profile, it uses the exposure curve to estimate what proportion of losses fall above the current policy limits. If 5% of expected losses fall above the limit on a Y2 risk, and the limit was chosen by judgment, the insurer may be systematically under-covered.

2. **Layered commercial property**: large UK commercial risks are often placed in layers. A building with a sum insured of £50 million might be placed as £5m ground-up, £20m excess of £5m, £25m excess of £25m. Each layer requires its own pricing, and the exposure curve provides the layer allocations.

### The R `mbbefd` package and the Python gap

Until `insurance-ilf`, UK actuaries who wanted to fit MBBEFD curves had two options: R's `mbbefd` package (Spedicato, Dutang et al., v0.8.13 on CRAN) or a custom Excel/NumPy implementation. The R package is mature and well-tested. The Excel approach is common but produces results that are hard to audit and impossible to integrate into a production pricing pipeline.

We are not aware of a maintained, openly available Python implementation of MBBEFD fitting and per-risk XL pricing prior to `insurance-ilf`. For UK pricing teams moving from Excel to Python and Databricks -- which is the journey this course supports -- building from scratch was the only option. `insurance-ilf` fills that gap.

### Connecting to your GLM and GBM work

The modules on GLMs, GBMs, and SHAP focused on the structure of claims frequency and severity across risk characteristics. The exposure curve sits orthogonally to that structure: it describes how, within a risk of a given profile, losses are distributed by size.

A natural integration: your GLM or GBM provides the expected frequency and severity for each risk cell. The exposure curve then tells you what proportion of that expected severity falls in any given layer. Combining the two gives you a full excess-of-loss price at the risk level, which can be aggregated across a portfolio to produce treaty rates that reflect both the risk mix and the size distribution within each risk.