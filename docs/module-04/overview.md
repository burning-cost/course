# Module 4: SHAP Relativities - From GBM to Rating Factor Tables

In Module 3 you trained a CatBoost frequency model on a synthetic UK motor portfolio, validated it with walk-forward cross-validation, tuned its hyperparameters with Optuna, and logged everything to MLflow. The model beats the GLM on Poisson deviance. But it has not gone to production yet.

The reason it has not gone to production is the same reason GBMs sit unused in notebooks across every major UK insurer: the pricing committee wants factor tables. The Chief Actuary wants to know what the model does to NCD. Radar needs an import file. The FCA's Consumer Duty requires you to explain what the model does to different groups of customers.

A CatBoost model is 300+ trees. You cannot read 300 trees. But you can decompose those trees mathematically, extract the contribution of each rating factor, and produce a table that is directly comparable to the GLM output from Module 2. That is what this module teaches.

The tool is SHAP: SHapley Additive exPlanations. For tree models with a log link, SHAP values translate directly into multiplicative relativities - the same format as `exp(beta)` from a GLM. By the end of this module you will have extracted a full set of rating factor tables from the CatBoost model, validated them against the GLM, formatted them for Radar import, and logged everything to MLflow.
[Download the notebook for this module](notebook.py)

---