# Module 8: End-to-End Pricing Pipeline

You have spent seven modules building the individual components of a pricing model. Module 3 taught you how to fit a Poisson GBM with the correct exposure offset. Module 4 showed you how to extract SHAP relativities. Module 5 gave you conformal prediction intervals with a provable coverage guarantee. Module 6 handled thin cells with Bayesian credibility. Module 7 optimised rate changes subject to loss ratio and volume constraints.

This module connects everything into a single pipeline that runs from raw data to a pricing committee pack. The pipeline is not a technical showcase. It is an organisational discipline. It exists because the individual components, however well-built, fail when they are connected incorrectly.

By the end of this module you will have run a complete UK motor rate review in one Databricks notebook: data ingestion, feature engineering, walk-forward validation, hyperparameter tuning, frequency and severity modelling, SHAP relativities, conformal prediction intervals, rate optimisation, and an audit record that satisfies the FCA's Consumer Duty reproducibility requirements.

---