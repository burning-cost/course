# Module 3: GBMs for Insurance Pricing - CatBoost

In Module 2 you fitted a Poisson GLM for claim frequency and a Gamma GLM for claim severity on a synthetic UK motor portfolio. You extracted relativities, ran diagnostics, and logged the model to MLflow. That GLM is your benchmark. This module asks whether a gradient boosted machine can beat it - and teaches you how to answer that question honestly.

The honest answer requires a proper evaluation framework. By the end of this module you will have fitted a CatBoost frequency model using walk-forward cross-validation designed for insurance data, tuned its hyperparameters with Optuna, fitted a Gamma-equivalent severity model, and produced the two diagnostics - Gini coefficient and double lift chart - that you would present to a pricing committee. Everything is logged to MLflow so the comparison is auditable.

We use the same synthetic motor portfolio as Module 2. The GLM you fitted there is the benchmark throughout this module.
[Download the notebook for this module](notebook.py)

---