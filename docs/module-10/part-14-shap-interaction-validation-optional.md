## Part 14: SHAP interaction validation (optional)

NID tells you what interactions the CANN learned from the GLM's residuals. As a second opinion, you can compute SHAP interaction values from the CatBoost model you trained in Module 3. When both methods flag the same pair, the evidence is stronger.

This section requires `catboost` and `shapiq`. If you did not install them, skip to Part 15.

```python
# Check if CatBoost and shapiq are available
try:
    from catboost import CatBoostRegressor, Pool
    import shapiq
    SHAP_AVAILABLE = True
    print("CatBoost and shapiq available.")
except ImportError:
    SHAP_AVAILABLE = False
    print("CatBoost or shapiq not available. Skipping SHAP validation.")
    print("Install with: %pip install catboost shapiq")
```

### Train a CatBoost model as the SHAP oracle

```python
if SHAP_AVAILABLE:
    cat_features = ["area", "vehicle_group", "age_band", "annual_mileage"]

    X_pd_cb = X.to_pandas()
    for col in cat_features:
        X_pd_cb[col] = X_pd_cb[col].astype(str)

    pool = Pool(
        X_pd_cb,
        label=y,
        baseline=np.log(exposure_arr),  # log-exposure offset for Poisson model; do not use weight=
        cat_features=cat_features,
    )

    cb_model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Poisson",
        random_seed=42,
        verbose=False,
    )
    cb_model.fit(pool)
    print("CatBoost training complete.")
```

This is the same CatBoost Poisson model from Module 3. If you have the Module 3 notebook still open, use that fitted model directly rather than retraining.

### Re-run the detector with SHAP validation

```python
if SHAP_AVAILABLE:
    detector_shap = InteractionDetector(family="poisson", config=cfg)
    detector_shap.fit(
        X=X,
        y=y,
        glm_predictions=mu_glm,
        exposure=exposure_arr,
        shap_model=cb_model,   # Pass the CatBoost model for SHAP interaction validation
    )

    # The full table now includes shap_score, shap_rank, and consensus_score
    table_shap = detector_shap.interaction_table()
    print(
        table_shap.select([
            "feature_1", "feature_2",
            "nid_score_normalised",
            "shap_score_normalised",
            "nid_rank", "shap_rank",
            "consensus_score",
            "recommended",
        ]).head(10)
    )
```

**Interpreting the consensus:** A pair that ranks first by NID and also ranks first by SHAP is very strong evidence of a genuine interaction. A pair that ranks third by NID but fifteenth by SHAP is worth scrutinising: the CANN found something the GBM did not, which could mean the interaction is an artefact of CANN training, or that the SHAP calculation is obscuring a real interaction via the independence assumption (SHAP interaction values assume features can be independently perturbed, which is violated by the structural correlation between age and NCD in UK motor).

### Why NID and SHAP can disagree

SHAP interaction values (technically: Shapley interaction indices) answer a different question from NID. SHAP interaction values measure how much the combined perturbation of two features changes the prediction, over and above the sum of individual perturbations. NID measures how much co-participation in the CANN's hidden units two features show.

The two methods can disagree because:

1. SHAP interaction values depend on the reference distribution (the conditional or marginal expectation). For correlated features like age and NCD, the reference distribution matters a lot.
2. NID depends on the CANN having converged. A noisy training run can inflate scores for spurious pairs.

Neither method is ground truth. The LR test is the ground truth for statistical significance. NID and SHAP are ranking tools for prioritising which pairs to test.