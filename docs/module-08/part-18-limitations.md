## Part 18: Limitations

**The pipeline tunes on a single temporal fold.** Hyperparameters that are optimal for accident year 2023 may not be optimal for 2025. In production, tune on the penultimate fold and validate on the most recent fold. Preserve the most recent year as a clean test year that is not touched until final evaluation.

**The conformal calibration assumes the calibration year's distribution matches the test year.** For a rapidly changing book (new channel mix, new product launch, significant rate action in the intervening period), the exchangeability assumption underlying conformal prediction may not hold. Recalibrate when the book changes materially.

**The rate optimiser uses a logistic demand model.** Actual renewal elasticity is more complex: tenure effects, channel effects, competitive market dynamics, and the interaction between price and risk profile all affect whether a specific policy renews. The logistic model is a reasonable approximation for optimisation but is not a demand forecast.

**The pipeline does not handle mid-term adjustments.** Policies that were adjusted (endorsements, mid-term cancellations) after inception appear in the data as modified versions of the original. The pipeline treats each row as representing the full policy period. For a book with high endorsement rates, this misrepresents the actual exposure and risk profile of individual policies.

**SHAP values are computed on the test set only.** The SHAP relativities in Stage 7 reflect the feature distributions of the 2024 test year. If the 2024 distribution differs from the prospective renewal portfolio (e.g., because new business has been written in different segments), the relativities may not accurately represent the prospective book's factor structure.

**The severity model trains on claims-only data.** This is correct for severity estimation, but it means the severity model's training set is a non-random sample of the full portfolio -- it is biased toward higher-risk policies. Any covariate that predicts claim occurrence (and most do) will be confounded with severity in the claims-only training set. This is a known issue in joint frequency-severity modelling.

**The pipeline audit uses `mode("append")`.** Multiple pipeline runs on the same day produce multiple audit rows with the same `run_date`. Add a UUID or `pipeline_version` field as the primary key if you need to distinguish same-day runs.

**The efficient frontier is not shown.** Stage 9 solves a single-point optimisation for a specific LR target. In practice, you would solve the optimiser across a range of LR targets (e.g., 0.68 to 0.78 in 1pp increments) and plot the resulting volume-LR frontier. The pricing committee makes a decision from the frontier, not from a single optimisation result. This is covered in Module 7 but not implemented in this capstone.