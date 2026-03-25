# Module 8: End-to-End Pricing Pipeline

In Module 7 you solved the constrained optimisation problem: given a technical premium, a demand model, and four simultaneous constraints, find the factor adjustments that minimise customer disruption while hitting the loss ratio target. That is the final analytical step in a pricing review.

This module is about everything that connects those analytical steps together — and why the connections are where reviews fail.

The NCB encoding incident in Part 1 is representative. The modelling team produced a technically sound CatBoost frequency model. The scoring team applied the same model with a different feature encoding. No exception was raised. The model ran for three months generating systematically wrong predictions. The monetary cost was one quarter of mispriced renewals. The regulatory cost was a Consumer Duty breach. The root cause was not a modelling error: it was a missing discipline around the connection between training and scoring.

End-to-end pipelines exist to enforce those connections. When every stage reads from a named Delta table, writes its output to another named Delta table, and logs its identity to MLflow, the disconnection that enabled the NCB incident cannot happen silently. The feature engineering runs once, in one function, at both training and scoring time. The calibration year is the same data version the model was trained on. The rate optimiser consumes the actual model predictions, not a separately generated proxy. The audit record names every table version and model run ID, so the FCA's section 166 request has a complete answer.

This is the capstone module. By the end you will have run a complete UK motor rate review in one Databricks notebook:

- Raw data ingested and written to a versioned Delta table
- Feature engineering defined once in a shared transform layer, applied at training and scoring
- Walk-forward cross-validation with an IBNR buffer that prevents partially developed claims from contaminating training
- CatBoost Poisson frequency model and Tweedie severity model, both tuned with Optuna on the last temporal fold
- SHAP relativities extracted using `shap_relativities` into the same format as GLM exp(beta) relativities
- Conformal prediction intervals calibrated on a temporally held-out calibration set, with coverage validated by decile
- Calibration testing with the Murphy decomposition: balance property, auto-calibration, and the RECALIBRATE vs REFIT verdict before handing predictions to the rate optimiser
- Rate optimisation using `insurance_optimise` with ENBP, volume, loss ratio, and factor-bounds constraints
- A pipeline audit record that ties every output to the data version and model run ID that produced it

Modules 9-12 build on this pipeline at specific insertion points. Module 9 adds a causal demand model for retention-aware pricing. Module 10 automates interaction detection. Module 11 adds production monitoring using `insurance_monitoring`. Module 12 adds postcode-level spatial territory factors. Each slots into the pipeline at a named stage. The pipeline architecture you build here is the skeleton they all attach to.

[Download the notebook for this module](notebook.py)

---
