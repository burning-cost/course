# Module 9: Demand Modelling and Price Elasticity

In Module 7 you built a constrained rate optimiser that could find the profit-maximising factor adjustments while satisfying your LR target, volume floor, and ENBP ceiling. It worked. But it required a demand model as an input - and you were given one: a logistic curve with a manually specified price coefficient. Where did that coefficient come from?

The honest answer in most UK pricing teams is: judgment, benchmarks from consultants, or a naive regression on observed lapse data. All three are worse than they look. The judgment is unauditable. The benchmarks are from other books, other market conditions, and other years. The naive regression is biased in a way that we will demonstrate precisely.

This module shows how to estimate the price coefficient properly, using causal inference methods that account for the confounding structure of insurance pricing data. By the end you will be able to take a quote or renewal dataset, estimate a causal price elasticity, audit the result for the near-deterministic price problem, build a demand curve, and run an FCA-compliant renewal pricing optimisation - all in Python, all reproducible, all auditable.

We use two libraries throughout: `insurance-demand` for conversion and retention modelling, and `insurance-elasticity` for the full causal estimation pipeline on renewal data. They have overlapping concerns and you may end up using both in production. We explain where each one fits.
[Download the notebook for this module](notebook.py)

---
