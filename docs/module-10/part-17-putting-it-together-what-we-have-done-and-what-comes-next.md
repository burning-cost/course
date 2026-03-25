## Part 17: Putting it together — what we have done and what comes next

### Summary

In this module we:

1. Understood why a correctly-specified GLM with main effects still misses interaction terms, and why manual 2D A/E plots are an incomplete search strategy.

2. Trained a CANN on the residuals of the baseline Poisson GLM. The skip-connection architecture forced the network to learn only what the GLM could not express.

3. Applied NID to the trained CANN weight matrices to get a ranked list of candidate interaction pairs in milliseconds.

4. Tested the top 15 candidates using likelihood-ratio tests with Bonferroni correction, confirming the two planted interactions (age band × vehicle group and NCD × has_convictions) as statistically significant.

5. Rebuilt the GLM jointly with the approved interaction terms, achieving a deviance improvement of around 1.5-2% with 20-25 new parameters.

6. Logged everything to MLflow and (optionally) validated with SHAP interaction values from CatBoost.

### What comes next

The interaction-enhanced GLM from this module is the most complete GLM you have built in this course. The next logical step is the `bayesian-pricing` library (covered in a future module), which handles the credibility problem you will now face: some interaction cells — a specific age band × vehicle group combination, say — may contain only 30 policies. The GLM estimate for that cell's interaction coefficient will have a large standard error. Bayesian partial pooling shrinks sparse cells towards the mean, which is exactly the right treatment.

For the current enhanced GLM, you should also run it through the walk-forward cross-validation from Module 3 to verify that the interaction terms improve out-of-sample performance, not just in-sample deviance.

### The audit trail question

Under PRA SS1/23 model risk governance, you must be able to justify every modelling decision. The interaction detection pipeline produces a clear, reproducible audit trail:

- The NID table explains which pairs the CANN found and in what order
- The LR test table explains which pairs are statistically significant at the Bonferroni-corrected threshold
- The `n_cells` column explains the parameter cost considered
- The MLflow artifact captures the full table at the point of the modelling decision

A peer reviewer can re-run the notebook, verify the results, and follow the reasoning from raw data to final model. That is what the FCA expects under Consumer Duty, and what the PRA expects under SS1/23. Building this audit trail into your workflow from the start costs very little additional effort.