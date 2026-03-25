## Part 14: Connecting elasticity to the rate optimiser

Module 7's rate optimiser accepts a demand model as an input. The Module 7 notebook used a manually specified logistic curve with a price coefficient taken from judgment or a benchmark. You now have a causally estimated coefficient. This part shows how to connect the two.

### The interface

The rate optimiser expects a callable that takes a price ratio and a feature DataFrame and returns a renewal probability for each row. The fitted `RenewalElasticityEstimator` does not have this exact interface — it estimates CATEs, not levels — but you can wrap it.

The practical connection for Module 7 is simpler: replace the manual `price_coef` in `make_logistic_demand()` with the ATE from the DML model.

```python
%md
## Part 14: Connecting to Module 7 rate optimiser
```

```python
# In Module 7 you called:
# demand_model = make_logistic_demand(intercept=2.0, price_coef=-2.2, tenure_coef=0.04)

# Replace with the DML-estimated ATE:
dml_price_coef = ate  # from est.ate() — causally estimated
print(f"Replacing manual coefficient (-2.2) with DML estimate ({dml_price_coef:.3f})")

# The Module 7 call becomes:
# demand_model = make_logistic_demand(
#     intercept=2.0,
#     price_coef=dml_price_coef,   # <-- DML-estimated
#     tenure_coef=0.04,
# )
```

### What changes in the optimiser output

If the DML ATE is less negative than the manual coefficient (e.g., −1.9 instead of −2.2), the optimiser believes the book is less elastic than assumed. The practical effect:

- The efficient frontier shifts: a given LR target is achievable at lower volume loss than the manual model predicted
- The optimal factor adjustments will be slightly more aggressive (higher rates)
- The shadow price on the volume constraint decreases — the constraint is worth less because volume responds less to price

If the DML ATE is more negative (e.g., −2.5 instead of −2.2), the optimiser will constrain the rate action more than the manual model did.

Neither direction is "better" — the DML estimate is simply more likely to be correct than the manual guess.

### The heterogeneous extension

For a fully segmented optimisation — different price strategies for NCD-0 PCW versus NCD-5 direct — the per-customer CATEs from `est.cate(df)` feed directly into the `RenewalPricingOptimiser` as shown in Part 12. This is a per-policy approach rather than a factor-table approach, but the two can coexist: use the factor-table optimiser from Module 7 for the initial rate structure, and the per-policy optimiser for retention discounts on elastic customers.

### Storing the elasticity estimate in MLflow

```python
import mlflow

mlflow.set_experiment("/Users/your.email@company.com/insurance-pricing/elasticity")

with mlflow.start_run(run_name="renewal_elasticity_causal_forest"):
    mlflow.log_metric("ate", ate)
    mlflow.log_metric("ate_ci_lower", lb)
    mlflow.log_metric("ate_ci_upper", ub)
    mlflow.log_param("n_training_records", len(df))
    mlflow.log_param("cate_model", "causal_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("catboost_iterations", 500)
    mlflow.log_param("confounders", str(confounders))
    mlflow.log_metric("treatment_variation_fraction", report.variation_fraction)
    mlflow.log_metric("enbp_breach_count", n_breaches)

    run_id = mlflow.active_run().info.run_id
    print(f"Logged to MLflow. Run ID: {run_id}")
```

The run ID links the ATE estimate used in the rate review to the model that produced it. For a section 166 audit, this is the starting point of your methodological audit trail.
