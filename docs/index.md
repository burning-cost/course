# Modern Insurance Pricing with Python and Databricks

A hands-on course for UK pricing actuaries and analysts. Twelve modules taking you from Databricks basics to production-grade pricing pipelines — GLMs, GBMs, SHAP relativities, conformal prediction, Bayesian credibility, constrained rate optimisation, demand elasticity, and spatial territory models.

---

## What this course covers

The course is structured around the problems pricing teams actually face. Each module tackles one problem in depth, with a full tutorial, working code, and exercises with solutions.

If you are coming from Emblem (Willis Towers Watson's GLM fitting tool) or Radar (their rating engine), this course shows you how to do the same work — and more — in Python on Databricks. No Emblem or Radar experience is required.

All code runs on Databricks. Most modules run on the free tier. Where a paid workspace is needed, it is noted clearly.

---

## Modules

### Module 1: Databricks for Pricing Teams

Set up a production-ready Databricks environment for a pricing team. Unity Catalog schema design, Delta tables with time travel, scheduled Workflows, and the FCA Consumer Duty infrastructure requirements.

**Duration:** 3–4 hours &nbsp;&nbsp; **Requires:** Free Edition (Workflows require paid workspace)

[Go to Module 1 &rarr;](module-01/overview.md){.md-button}

---

### Module 2: GLMs in Python: The Bridge from Emblem

Poisson frequency and gamma severity GLMs in Python, built to match Emblem output. Factor encoding, exposure offsets, actual-versus-expected diagnostics, Emblem parity validation, and Radar export.

**Duration:** 5–6 hours &nbsp;&nbsp; **Requires:** Free Edition

[Go to Module 2 &rarr;](module-02/overview.md){.md-button}

---

### Module 3: GBMs for Insurance Pricing: CatBoost

CatBoost frequency-severity model on a synthetic motor portfolio. Walk-forward cross-validation with IBNR buffer, Optuna hyperparameter tuning, Gini coefficient diagnostics, and MLflow champion-challenger governance.

**Duration:** 4–5 hours &nbsp;&nbsp; **Requires:** Free Edition

[Go to Module 3 &rarr;](module-03/overview.md){.md-button}

---

### Module 4: SHAP Relativities: From GBM to Rating Factor Tables

Extract multiplicative rating factor tables from a CatBoost model using SHAP values. Smoothed curves for continuous features, confidence intervals, GLM benchmark comparison, and export to Radar/Akur8/Emblem.

**Duration:** 4–5 hours &nbsp;&nbsp; **Requires:** Free Edition

[Go to Module 4 &rarr;](module-04/overview.md){.md-button}

---

### Module 5: Conformal Prediction Intervals

Distribution-free prediction intervals with guaranteed coverage — no distributional assumptions. Calibrate and validate a conformal predictor on a CatBoost Tweedie model. Applications for underwriting referral flags and reserve range estimates.

**Duration:** 4 hours &nbsp;&nbsp; **Requires:** Free Edition

[Go to Module 5 &rarr;](module-05/overview.md){.md-button}

---

### Module 6: Credibility and Bayesian Pricing: The Thin-Cell Problem

Bühlmann-Straub credibility (EPV/VHM/K structural parameters), connection to empirical Bayes, and hierarchical Bayesian frequency modelling with PyMC. Shrinkage plots, posterior credibility factors, and Unity Catalog storage.

**Duration:** 5–6 hours &nbsp;&nbsp; **Requires:** Free Edition (Bayesian section requires 16 GB RAM)

[Go to Module 6 &rarr;](module-06/overview.md){.md-button}

---

### Module 7: Constrained Rate Optimisation

Replace ad-hoc Excel rate exercises with formally stated constrained optimisation. Loss ratio target, volume floor, per-factor movement caps, ENBP constraint, efficient frontier analysis and shadow pricing.

**Duration:** 4–5 hours &nbsp;&nbsp; **Requires:** Paid Databricks workspace

[Go to Module 7 &rarr;](module-07/overview.md){.md-button}

---

### Module 8: End-to-End Pricing Pipeline

Capstone module. Full UK personal lines rate review pipeline — 200,000 synthetic motor policies, shared transform layer, walk-forward CV, CatBoost models, conformal intervals, constrained rate optimisation, and Consumer Duty compliance audit record.

**Duration:** 6–8 hours &nbsp;&nbsp; **Requires:** Paid Databricks workspace

[Go to Module 8 &rarr;](module-08/overview.md){.md-button}

---

### Module 9: Demand Modelling and Price Elasticity

Conversion and retention modelling with CatBoost. Causal price elasticity via Double Machine Learning (EconML CausalForestDML), heterogeneous CATE estimates, profit-maximising price identification, and ENBP-constrained renewal pricing optimisation.

**Duration:** 5–6 hours &nbsp;&nbsp; **Requires:** Paid Databricks workspace

[Go to Module 9 &rarr;](module-09/overview.md){.md-button}

---

### Module 10: Interaction Detection

Automated GLM interaction detection using Combined Actuarial Neural Networks (CANN) and Neural Interaction Detection (NID). Bonferroni correction, likelihood-ratio tests, GLM rebuild with discovered interactions, and SHAP interaction validation.

**Duration:** 4–5 hours &nbsp;&nbsp; **Requires:** Free Edition

[Go to Module 10 &rarr;](module-10/overview.md){.md-button}

---

### Module 11: Model Monitoring and Drift Detection

Detect when a deployed pricing model degrades. Population stability index, characteristic stability index, actual-versus-expected ratios with confidence intervals, Gini drift z-test, automated traffic-light triggers, Delta Lake logging, and Databricks job scheduling for continuous monitoring.

**Duration:** 4–5 hours &nbsp;&nbsp; **Requires:** Free Edition (scheduling exercises require paid workspace)

[Go to Module 11 &rarr;](module-11/overview.md){.md-button}

---

### Module 12: Spatial Territory Rating

Replace Emblem-style postcode group rating with Bayesian spatial models. Adjacency matrix construction, Moran's I spatial autocorrelation test, BYM2 model fitted via PyMC, territory relativity extraction, and integration into a downstream GLM rating engine.

**Duration:** 5–6 hours &nbsp;&nbsp; **Requires:** Paid Databricks workspace

[Go to Module 12 &rarr;](module-12/overview.md){.md-button}

---

## How to use this course

Work through the modules in order — each one builds on the previous. If you are already comfortable with Databricks, you can start at Module 2.

Each module follows the same pattern:

1. Read the tutorial, working through the code in your Databricks notebook
2. Complete the exercises — they are not optional, they are where the learning happens
3. Check your solutions against the provided answers

The code in every module runs against synthetic data that is generated within the notebook itself. You do not need access to your insurer's data to complete the course.

---

## Setup

Before starting Module 1, read the [Getting Started](getting-started.md) guide. It covers Databricks account setup, library installation, and how the notebooks are organised.
