# Module 10: Automated Interaction Detection

In Module 2 you built a Poisson frequency GLM for UK motor insurance. In Module 3 you trained a CatBoost GBM on the same data. The CatBoost model had lower deviance. In Module 4 you extracted SHAP relativities and compared them to the GLM factors.

Here is a question you might have asked during Module 4: if the SHAP relativities for `driver_age` and `vehicle_group` both match the GLM fairly well but the CatBoost model still beats the GLM on deviance, where is the extra predictive power coming from?

The answer is interactions.

A GLM with additive main effects assumes that the effect of being a young driver is the same regardless of what vehicle you drive. A 22-year-old in vehicle group 5 and a 22-year-old in vehicle group 45 are both young, but the latter is a substantially worse risk — not because vehicle group 45 is worse on average, and not because being 22 is worse on average, but because the combination of high age risk and high vehicle group risk is supermultiplicative. The GLM, which multiplies two independent factors together, cannot capture this. The GBM can.

This module shows you how to find those missing interactions systematically, test them statistically, and add the significant ones back to your GLM.
[Download the notebook for this module](notebook.py)

---
