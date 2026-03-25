## Part 14: SHAP interaction validation (optional)

NID tells you what interactions the CANN learned from the GLM's residuals. As a second opinion, you can compute SHAP interaction values from a CatBoost model. When both methods flag the same pair, the evidence is stronger.

This section requires the `shap` extra for `insurance-interactions`. If you did not install it, skip to Part 15.

```python
# Check if the shap extra dependencies are available
try:
    import catboost
    import shapiq
    SHAP_AVAILABLE = True
    print("catboost and shapiq available.")
except ImportError:
    SHAP_AVAILABLE = False
    print("catboost or shapiq not available. Skipping SHAP validation.")
    print("Install with: %pip install \"insurance-interactions[shap]\"")
```

### Train a CatBoost model as the SHAP oracle

The `insurance-interactions` library provides `fit_catboost()` to train the GBM oracle. This handles the Polars-to-pandas conversion, categorical feature encoding, and the correct loss function.

```python
if SHAP_AVAILABLE:
    from insurance_interactions.shap_interactions import fit_catboost

    cb_model = fit_catboost(
        X=X,
        y=y,
        exposure=exposure_arr,
        family="poisson",
        iterations=500,
        depth=6,
        learning_rate=0.05,
        seed=42,
        verbose=False,
    )
    print("CatBoost training complete.")
```

`fit_catboost` infers categorical columns from the Polars dtype and uses `sample_weight=exposure` in the Pool, which is the correct treatment for Poisson frequency (CatBoost does not support a log-offset for Poisson, so exposure enters as a weight).

### Re-run the detector with SHAP validation

```python
if SHAP_AVAILABLE:
    detector_shap = InteractionDetector(family="poisson", config=cfg)
    detector_shap.fit(
        X=X,
        y=y,
        glm_predictions=mu_glm,
        exposure=exposure_arr,
        shap_model=cb_model,   # Pass the fitted CatBoost model
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

When `shap_model` is passed to `detector.fit()`, the library calls `compute_shap_interactions` internally and merges the scores into `interaction_table()`. If the shapiq computation fails (e.g., the model type is unsupported), a `UserWarning` is emitted and the pipeline continues with NID scores only — it does not raise.

**Interpreting the consensus:** A pair that ranks first by NID and also ranks first by SHAP is very strong evidence of a genuine interaction. A pair that ranks third by NID but fifteenth by SHAP is worth scrutinising: the CANN found something the GBM did not, which could mean the interaction is a CANN training artefact, or that the SHAP calculation is obscuring a real interaction via the independence assumption.

### Why NID and SHAP can disagree

SHAP interaction values (technically: Shapley interaction indices) answer a different question from NID. SHAP interaction values measure how much the combined perturbation of two features changes the prediction, over and above the sum of individual perturbations. NID measures how much co-participation in the CANN's hidden units two features show.

The two methods can disagree because:

1. SHAP interaction values depend on the reference distribution (the conditional or marginal expectation). For correlated features like age and NCD, the reference distribution matters a lot.
2. NID depends on the CANN having converged. A noisy training run can inflate scores for spurious pairs.

Neither method is ground truth. The LR test is the ground truth for statistical significance. NID and SHAP are ranking tools for prioritising which pairs to test.

Note: `compute_shap_interactions` subsamples to `max_rows=5000` by default before calling shapiq, since full pairwise SHAP on 100,000 rows is expensive. 5,000 rows is typically sufficient for stable rankings.
