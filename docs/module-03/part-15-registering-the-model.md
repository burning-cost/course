## Part 15: Registering the model

The last step is registering the frequency model in the MLflow Model Registry. We register it as "challenger" - not "production." The challenger alias means it is available for review, but nothing in the production pipeline loads a model by the `challenger` alias. Only a human decision - explicitly changing the alias to `production` - moves it into production. That is the governance gate.

Create a new cell:

```python
client = MlflowClient()
MODEL_NAME = "motor_freq_catboost_m03"

freq_uri   = f"runs:/{freq_run_id}/freq_model"
registered = mlflow.register_model(model_uri=freq_uri, name=MODEL_NAME)

# Use aliases (not stage transitions, which are deprecated in MLflow 2.9+)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="challenger",
    version=registered.version,
)

# Tag with metadata for the governance review
client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="cv_strategy",
    value="walk_forward_ibnr1",
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="gini_lift",
    value=str(round(gini_gbm - gini_glm, 4)),
)

print(f"Registered: {MODEL_NAME} version {registered.version} as 'challenger'")
```

**Why `mlflow.register_model` not `log_model`?** `mlflow.catboost.log_model()` logs the model as an artefact attached to the run. `mlflow.register_model()` takes that logged artefact and registers it in the Model Registry - a separate catalogue that tracks versions, aliases, and tags across runs. A model can be logged to many runs but only registered once per meaningful version.

**The `@challenger` loading syntax:** Once registered, you can load the model anywhere in the organisation with:

```python
model = mlflow.catboost.load_model("models:/motor_freq_catboost_m03@challenger")
```

Module 4 retrains its own model with a slightly different feature set (using `has_convictions` instead of `conviction_points`, and adding `annual_mileage`), so it does not load this registered model directly. However, this same loading pattern is used in Module 8's production pipeline, where the registered production model is loaded by alias for scoring and monitoring.

**On stage transitions vs aliases:** Do not use `client.transition_model_version_stage()`. It is deprecated in MLflow 2.9+ and generates warnings on Databricks Runtime 14.x. Use `set_registered_model_alias()` instead, as shown above.

---

## What the pricing committee needs to see

When you present this to the pricing committee, they need two things.

**A clear statement of the evaluation framework.** Something like: "We trained on accident years 2019-2023 and tested on 2024, using the same training data version for both GLM and GBM. Walk-forward CV with an IBNR buffer of one year was used for hyperparameter tuning. The test year was held out throughout and not used in any tuning decisions."

**The double lift chart with interpretation.** Something like: "The GBM outperforms the GLM by X Gini points. The double lift chart shows the additional discrimination is concentrated in the top three deciles of the GBM/GLM ratio - policies where the GBM is predicting materially more than the GLM. Those policies have actual frequencies two to three times the portfolio average, confirming genuine risk identification. The bottom seven deciles show essentially flat actual frequencies, consistent with both models agreeing on the bulk of the book."

That is a complete and honest statement. If the Gini lift is 0.03+, you have a reasonable case for moving to Module 4 (SHAP extraction). If it is 0.01 or less, the GBM is not adding enough to justify the governance overhead.

---

## What comes next

Module 4 - SHAP Relativities. The GBM is currently a black box to the pricing committee. The challenger model is registered, the Gini lift is logged, and the double lift chart is saved to MLflow. None of that is enough for the committee to approve deployment.

Module 4 uses `shap-relativities` to extract multiplicative factor tables from the GBM. These are tables the committee can read, debate, and sign off on - the same format as the GLM relativities from Module 2. If the SHAP relativities look actuarially sensible (young drivers are expensive, high vehicle groups are expensive, high NCD is cheap), the committee can approve. If any factor table looks wrong, the committee can reject without the GBM touching the production system.

That governance gate is the point. Building an accurate GBM is straightforward. Getting it through a rigorous governance process and into production pricing is what takes expertise.