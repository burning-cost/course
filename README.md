# Modern Insurance Pricing with Python and Databricks

A free, hands-on course for UK pricing actuaries and analysts. Twelve modules taking you from Databricks basics to a production-grade rate review pipeline — GLMs, GBMs, SHAP relativities, conformal prediction, Bayesian credibility, constrained rate optimisation, causal demand modelling, and spatial territory rating.

The course is published at [burning-cost.github.io/course](https://burning-cost.github.io/course/).

---

## Who this is for

Pricing actuaries and analysts at UK personal lines insurers who want to move their rate review work into Python and Databricks. If you are coming from Emblem and Radar — Willis Towers Watson's GLM fitting and rating engine tools — this course shows you how to replicate and extend what you do there.

You do not need to be a software engineer. If you can read Python, filter a DataFrame, and call a function, you have enough Python for this course. GLM familiarity is assumed from Module 2 onwards — you should know what a link function is and what a factor table looks like.

---

## What it covers

| # | Module | Topics | Time |
|---|--------|---------|------|
| 01 | Databricks for Pricing Teams | Unity Catalog schema design, Delta tables with time travel, scheduled Workflows, FCA Consumer Duty infrastructure | 3–4 h |
| 02 | GLMs in Python: The Bridge from Emblem | Poisson frequency and gamma severity GLMs, exposure offsets, actual-versus-expected diagnostics, Emblem parity validation, Radar export | 5–6 h |
| 03 | GBMs for Insurance Pricing: CatBoost | Walk-forward cross-validation with IBNR buffer, Optuna hyperparameter tuning, frequency-severity model, MLflow champion-challenger governance | 4–5 h |
| 04 | SHAP Relativities: From GBM to Rating Factor Tables | Multiplicative rating factors from CatBoost SHAP values, smoothed curves, confidence intervals, GLM comparison, Radar/Akur8 export | 4–5 h |
| 05 | Conformal Prediction Intervals | Distribution-free coverage guarantees on Tweedie models, underwriting referral flags, minimum premium floors, reserve range estimates | 4 h |
| 06 | Credibility and Bayesian Pricing | Bühlmann-Straub credibility (EPV/VHM structural parameters), PyMC hierarchical models, shrinkage plots, thin-cell segments | 5–6 h |
| 07 | Constrained Rate Optimisation | SLSQP optimisation, loss ratio target, volume floor, per-factor movement caps, ENBP constraint, efficient frontier and shadow prices | 4–5 h |
| 08 | End-to-End Pricing Pipeline | Capstone: 200,000-policy synthetic motor portfolio, shared transform layer, CV, conformal intervals, rate optimisation, Consumer Duty audit record | 6–8 h |
| 09 | Demand Modelling and Price Elasticity | Conversion and retention models, Double Machine Learning (EconML CausalForestDML), heterogeneous CATE, ENBP-constrained renewal optimisation | 5–6 h |
| 10 | Interaction Detection | Combined Actuarial Neural Networks (CANN), Neural Interaction Detection (NID), Bonferroni correction, likelihood-ratio tests, GLM rebuild | 4–5 h |
| 11 | Model Monitoring and Drift Detection | PSI, CSI, actual-versus-expected ratios, Gini drift z-test, traffic-light alerts, Delta Lake logging, Databricks job scheduling | 4–5 h |
| 12 | Spatial Territory Rating | BYM2 Bayesian spatial model via PyMC, adjacency matrices, Moran's I, territory relativity extraction, comparison to Emblem postcode groups | 5–6 h |

Total: approximately 54–70 hours of study and practical work.

Modules 1–8 form the core path from Databricks setup to a complete rate review. Modules 9–12 extend the pipeline with specialist techniques.

---

## This course is completely free

No registration, no paywall, no certificate to buy. The full course content, including all notebooks, tutorials, and exercise solutions, is published openly.

All code runs on Databricks. Modules 1–5 and the classical credibility section of Module 6 run on the free Databricks tier. The Bayesian sections (Modules 6 and 12), the rate optimisation modules (7 and 8), and the demand modelling module (9) require a paid workspace. These requirements are noted clearly in each module.

---

## Open-source libraries

Several modules use open-source insurance pricing libraries built by Burning Cost. Install them all at once in a Databricks notebook cell:

```python
%pip install insurance-datasets insurance-cv shap-relativities insurance-conformal \
    credibility bayesian-pricing rate-optimiser insurance-optimise \
    insurance-causal insurance-interactions insurance-monitoring insurance-spatial --quiet
dbutils.library.restartPython()
```

Each notebook also includes an install cell at the top that installs only what that module needs. Source code for all libraries is at [github.com/burning-cost](https://github.com/burning-cost).

---

## Getting started

1. Read the [Getting Started guide](https://burning-cost.github.io/course/getting-started/) for account setup and notebook import instructions.
2. Sign up for a Databricks Free Edition account at [databricks.com/try-databricks](https://www.databricks.com/try-databricks).
3. Start with Module 1, which sets up the Unity Catalog schema and Delta table structure that every subsequent module depends on.

---

## Repository structure

```
docs/
  getting-started.md
  module-01/          # Tutorial parts, exercises, Databricks notebook
  module-02/
  ...
  module-12/
mkdocs.yml            # MkDocs configuration for the published site
```

Each module directory contains:
- `overview.md` — scope, prerequisites, estimated time
- `part-NN-*.md` — tutorial sections
- `exercises.md` — exercises with worked solutions
- `notebook.py` — Databricks source-format notebook (import directly into your workspace)

---

## Contact

Report errors or ask questions at pricing.frontier@gmail.com. Include the module number and section. Response within 2 business days.

More content and tools at [burning-cost.github.io](https://burning-cost.github.io).
