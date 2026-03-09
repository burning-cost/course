# Module 10 Exercises: Interaction Detection and GBM-to-GLM Distillation

Ten exercises. Work through them in order -- each builds on the dataset and models from the previous exercise. Solutions are inside collapsed sections at the end of each exercise.

Before starting: read Parts 1-17 of the tutorial. Every concept used here is explained there.

**Notebook setup.** These exercises continue in the same `module-10-interactions` notebook from the tutorial. If you are starting fresh, re-run Parts 4-6 of the tutorial to regenerate the synthetic portfolio, fit the baseline GLM, and confirm the imports before tackling Exercise 1.

---

## Exercise 1: Understanding what the GLM is missing

**Reference:** Tutorial Parts 1, 5, 7

**What you will do:** Diagnose the baseline GLM's residuals systematically before training the CANN. Build the intuition for where interaction signal lives.

**Context.** You have the 100,000-policy synthetic motor portfolio from the tutorial. The baseline Poisson GLM has been fitted on main effects only. Two interactions were planted in the data: `age_band × vehicle_group` and `ncd_years × conviction_points`. Your job is to find them using only the GLM's residuals -- no CANN, no NID -- to understand what the automated pipeline is doing when it outperforms this manual approach.

### Setup

Add a markdown cell `%md ## Exercise 1: GLM residual diagnostics`, then confirm you have the objects from the tutorial:

```python
# Confirm objects are in scope
assert "glm_base" in dir(), "Fit the baseline GLM from Part 6 of the tutorial first"
assert "mu_glm"   in dir(), "Need mu_glm from Part 6"
assert "X"        in dir(), "Need X (Polars DataFrame) from Part 5"
assert "y"        in dir(), "Need y (claim counts) from Part 5"
assert "exposure_arr" in dir(), "Need exposure_arr from Part 5"

print(f"GLM deviance: {poisson_deviance(y, mu_glm, exposure_arr):,.1f}")
print(f"Policies:     {len(X):,}")
print(f"Features:     {X.columns}")
```

### Tasks

**Task 1.** Build an actual-versus-expected (A/E) ratio table for every pair of features. For each pair `(feature_i, feature_j)`, group by the two features jointly, sum observed claims and GLM-predicted claims, and compute `ae_ratio = observed / predicted`. You want to find the pairs with the largest spread in A/E ratios -- the spread measures how much the GLM is wrong across cells of that pair.

Specifically, for each pair compute:
- `max_ae`: the maximum A/E ratio across all cells of that pair
- `min_ae`: the minimum
- `ae_spread`: `max_ae - min_ae`
- `n_cells_exposed`: the number of (i, j) cells with at least 10 claims

Sort the result by `ae_spread` descending. Which pair has the widest spread?

```python
import polars as pl
import itertools

df_diag = pl.DataFrame({
    "area":              X["area"].to_list(),
    "vehicle_group":     X["vehicle_group"].to_list(),
    "ncd_years":         X["ncd_years"].to_list(),
    "age_band":          X["age_band"].to_list(),
    "conviction_points": X["conviction_points"].to_list(),
    "annual_mileage":    X["annual_mileage"].to_list(),
    "y":                 y,
    "mu_glm":            mu_glm,
    "exposure":          exposure_arr,
})

features = ["age_band", "vehicle_group", "ncd_years", "conviction_points", "area", "annual_mileage"]

# Your code here -- loop over itertools.combinations(features, 2)
```

**Task 2.** The pair with the widest A/E spread should be `age_band × vehicle_group` or very close to it. Now produce the full 2D A/E table for that pair: one row per (age_band, vehicle_group) cell. Print the five cells with the highest A/E ratios. Do the cells that are worst for the GLM correspond to what you would expect from the planted interaction (young driver, high vehicle group)?

**Task 3.** Now do the same for `ncd_years × conviction_points`. Print the 2D A/E table. The interaction planted in the data adds a 0.20 log-unit penalty when `ncd_years == 0` AND `conviction_points > 0`. In relativities, `exp(0.20) ≈ 1.22`. Does the A/E table for the cell (ncd_years=0, conviction_points=9) show a ratio near 1.22? Why might it differ?

**Task 4.** Now check the pair `annual_mileage × conviction_points`. The data has no planted interaction between these two factors. Compute its 2D A/E table and report: what is the `ae_spread`? Compare it to the spreads for the two planted pairs. This is the false positive risk of manual 2D analysis: random variation can produce non-trivial spread even when no interaction exists.

**Task 5.** With 6 features, how many pairs did you check in Task 1? For a real motor pricing model with 12 rating factors, how many pairs would you check? Write a one-paragraph note (as a markdown cell in your notebook) on why manual 2D A/E analysis is incomplete as an interaction search strategy.

<details>
<summary>Hint for Task 1</summary>

`itertools.combinations(features, 2)` generates all unique pairs. For each pair `(f1, f2)`, use Polars `.group_by([f1, f2]).agg(...)` to get cell-level observed and predicted totals.

The spread calculation is straightforward after grouping. Use `.filter(pl.col("observed") >= 10)` to restrict to cells with credible data before computing max/min.

</details>

<details>
<summary>Solution -- Exercise 1</summary>

```python
import itertools
import polars as pl

features = ["age_band", "vehicle_group", "ncd_years", "conviction_points", "area", "annual_mileage"]

spread_records = []
for f1, f2 in itertools.combinations(features, 2):
    cell_df = (
        df_diag
        .group_by([f1, f2])
        .agg([
            pl.sum("y").alias("observed"),
            pl.sum("mu_glm").alias("predicted"),
        ])
        .with_columns(
            (pl.col("observed") / pl.col("predicted")).alias("ae_ratio")
        )
    )
    # Restrict to credible cells
    credible = cell_df.filter(pl.col("observed") >= 10)
    if credible.height < 2:
        continue
    max_ae  = credible["ae_ratio"].max()
    min_ae  = credible["ae_ratio"].min()
    spread_records.append({
        "pair":              f"{f1} x {f2}",
        "ae_spread":         round(max_ae - min_ae, 4),
        "max_ae":            round(max_ae, 4),
        "min_ae":            round(min_ae, 4),
        "n_cells_exposed":   credible.height,
    })

spread_df = pl.DataFrame(spread_records).sort("ae_spread", descending=True)
print("Task 1: A/E spread by feature pair (credible cells only):")
print(spread_df)

# Task 2: Full 2D table for age_band x vehicle_group
ae_age_vg = (
    df_diag
    .group_by(["age_band", "vehicle_group"])
    .agg([
        pl.sum("y").alias("observed"),
        pl.sum("mu_glm").alias("predicted"),
        pl.sum("exposure").alias("exposure"),
    ])
    .with_columns(
        (pl.col("observed") / pl.col("predicted")).alias("ae_ratio")
    )
    .sort("ae_ratio", descending=True)
)
print("\nTask 2: Worst cells for age_band x vehicle_group:")
print(ae_age_vg.head(5))

# Task 3: ncd_years x conviction_points
ae_ncd_cv = (
    df_diag
    .group_by(["ncd_years", "conviction_points"])
    .agg([
        pl.sum("y").alias("observed"),
        pl.sum("mu_glm").alias("predicted"),
    ])
    .with_columns(
        (pl.col("observed") / pl.col("predicted")).alias("ae_ratio")
    )
    .sort(["ncd_years", "conviction_points"])
)
print("\nTask 3: ncd_years x conviction_points A/E table:")
print(ae_ncd_cv)
cell_0_9 = ae_ncd_cv.filter(
    (pl.col("ncd_years") == 0) & (pl.col("conviction_points") == 9)
)
print(f"\nPlanted cell (ncd=0, cv=9): A/E = {cell_0_9['ae_ratio'][0]:.3f}  (expected ~1.22)")

# Task 4: annual_mileage x conviction_points -- no planted interaction
ae_mil_cv = (
    df_diag
    .group_by(["annual_mileage", "conviction_points"])
    .agg([
        pl.sum("y").alias("observed"),
        pl.sum("mu_glm").alias("predicted"),
    ])
    .filter(pl.col("observed") >= 10)
    .with_columns(
        (pl.col("observed") / pl.col("predicted")).alias("ae_ratio")
    )
)
mil_spread = ae_mil_cv["ae_ratio"].max() - ae_mil_cv["ae_ratio"].min()
print(f"\nTask 4: annual_mileage x conviction_points spread: {mil_spread:.4f}")

# Task 5: pair counts
n_features = len(features)
n_pairs = n_features * (n_features - 1) // 2
print(f"\nTask 5: pairs checked = {n_pairs}")
print(f"For 12 features: {12*11//2} pairs")
```

**What you should see:**

- `age_band × vehicle_group` should have the highest `ae_spread`, with the cell (17-21, 41-50) showing A/E around 1.25-1.35 (the 0.30 log-unit interaction maps to `exp(0.30) ≈ 1.35`).
- `ncd_years × conviction_points` should have the second-highest spread, with the cell (0, 9) near 1.22.
- `annual_mileage × conviction_points` will have a non-trivial spread -- somewhere in the range 0.15-0.30 -- despite having no planted interaction. This is pure noise, and it illustrates why the 2D A/E approach requires human judgement to separate signal from noise.

**Task 5 answer:** With 6 features, you checked 15 pairs. With 12 features, you would check 66 pairs. The manual process selectively checks the pairs an actuary expects to be interesting, which means the unexpected interactions -- the ones that are genuinely surprising and therefore most likely to be missed in a real portfolio -- are never checked.

</details>

---

## Exercise 2: Training the CANN with different configurations

**Reference:** Tutorial Parts 8, 3

**What you will do:** Train two CANN configurations, inspect their training histories, and understand the sensitivity of NID scores to training hyperparameters.

**Context.** The tutorial trained a single default CANN. Here you will train two variants and compare their NID rankings. The goal is to understand what "stable NID scores" means in practice, and when you should be worried that the scores are unreliable.

### Setup

```python
from insurance_interactions import InteractionDetector, DetectorConfig

%md ## Exercise 2: CANN configuration comparison
```

### Tasks

**Task 1.** Train a "quick" CANN with `cann_n_epochs=50, cann_patience=10, cann_n_ensemble=1`. Name it `detector_quick`. After training, call `detector_quick.nid_table()` and print the top 10 pairs by NID score. How long did training take?

```python
cfg_quick = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=50,
    cann_patience=10,
    cann_n_ensemble=1,
    top_k_nid=15,
    top_k_final=5,
)
detector_quick = InteractionDetector(family="poisson", config=cfg_quick)
# Your fit() call here
```

**Task 2.** Now train a "careful" CANN with `cann_n_epochs=300, cann_patience=30, cann_n_ensemble=3`. Name it `detector_careful`. After training, print the top 10 pairs by NID score. Compare the rankings side-by-side with the quick CANN.

**Task 3.** Inspect the validation deviance histories for both detectors using `detector.cann.val_deviance_history`. For the quick CANN (single ensemble run), does the curve plateau before epoch 50, or was it still improving when training stopped? What does this tell you about the reliability of the NID scores?

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Quick CANN
axes[0].plot(detector_quick.cann.val_deviance_history[0])
axes[0].set_title("Quick CANN (1 run, 50 epochs max)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation deviance")

# Careful CANN (3 ensemble runs)
for i, hist in enumerate(detector_careful.cann.val_deviance_history):
    axes[1].plot(hist, label=f"Run {i+1}")
axes[1].legend()
axes[1].set_title("Careful CANN (3 runs, 300 epochs max)")
axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.show()
```

**Task 4.** Look at the top-5 NID pairs from each CANN. Answer the following:

- Do both CANNs agree on the top-2 pairs? If not, which one do you trust more and why?
- The planted interactions should appear in both top-5 lists. If one is missing from the quick CANN's list, what is the most likely explanation?
- Would you use the quick CANN's rankings to present interaction candidates to a pricing committee? Why or why not?

**Task 5.** In the `detector_careful` NID table, find the pair with the highest NID score and the pair ranked 10th. Compute the ratio of their NID scores. If the top pair has score 0.85 and the 10th pair has score 0.12, what does this ratio tell you about the credibility of the 10th-ranked interaction?

<details>
<summary>Hint for Task 3</summary>

If the validation deviance curve is still falling at the final epoch, the model stopped training before converging. The NID scores come from the weight matrices at early stopping; if the model had not converged, those weights encode a model that has not finished learning the residual structure. The NID scores from an unconverged CANN are less reliable.

Specifically: if `np.argmin(history)` equals `len(history) - 1`, the model was still improving when patience was exhausted.

</details>

<details>
<summary>Solution -- Exercise 2</summary>

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from insurance_interactions import InteractionDetector, DetectorConfig

# Task 1: Quick CANN
cfg_quick = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=50,
    cann_patience=10,
    cann_n_ensemble=1,
    top_k_nid=15,
    top_k_final=5,
)
detector_quick = InteractionDetector(family="poisson", config=cfg_quick)

t0 = time.time()
detector_quick.fit(X=X, y=y, glm_predictions=mu_glm, exposure=exposure_arr)
t_quick = time.time() - t0
print(f"Quick CANN training time: {t_quick:.1f}s")
print("\nQuick CANN -- top 10 NID pairs:")
print(detector_quick.nid_table().head(10))

# Task 2: Careful CANN
cfg_careful = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_patience=30,
    cann_n_ensemble=3,
    top_k_nid=15,
    top_k_final=5,
)
detector_careful = InteractionDetector(family="poisson", config=cfg_careful)

t0 = time.time()
detector_careful.fit(X=X, y=y, glm_predictions=mu_glm, exposure=exposure_arr)
t_careful = time.time() - t0
print(f"\nCareful CANN training time: {t_careful:.1f}s")
print("\nCareful CANN -- top 10 NID pairs:")
print(detector_careful.nid_table().head(10))

# Task 3: Training histories
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

hist_quick = detector_quick.cann.val_deviance_history[0]
best_quick = int(np.argmin(hist_quick))
axes[0].plot(hist_quick)
axes[0].axvline(x=best_quick, color="red", linestyle="--",
                label=f"Best: epoch {best_quick}")
axes[0].set_title("Quick CANN (1 run, 50 epochs max)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation deviance")
axes[0].legend()

still_improving_quick = (best_quick == len(hist_quick) - 1)
print(f"\nQuick CANN: still improving at final epoch? {still_improving_quick}")
print(f"  Best epoch: {best_quick} out of {len(hist_quick)}")

for i, hist in enumerate(detector_careful.cann.val_deviance_history):
    best_ep = int(np.argmin(hist))
    axes[1].plot(hist, label=f"Run {i+1} (best: {best_ep})")
    print(f"Careful CANN run {i+1}: best epoch {best_ep} / {len(hist)}")
axes[1].legend()
axes[1].set_title("Careful CANN (3 runs, 300 epochs max)")
axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.show()

# Task 4: Side-by-side comparison
quick_top5  = [
    f"{r['feature_1']} x {r['feature_2']}"
    for r in detector_quick.nid_table().head(5).iter_rows(named=True)
]
careful_top5 = [
    f"{r['feature_1']} x {r['feature_2']}"
    for r in detector_careful.nid_table().head(5).iter_rows(named=True)
]

print("\nTask 4: Top-5 comparison")
print(f"{'Rank':<6} {'Quick CANN':<40} {'Careful CANN':<40}")
for i, (q, c) in enumerate(zip(quick_top5, careful_top5), 1):
    match = "AGREE" if q == c else "DIFFER"
    print(f"{i:<6} {q:<40} {c:<40}  {match}")

planted = {"age_band x vehicle_group", "vehicle_group x age_band",
           "ncd_years x conviction_points", "conviction_points x ncd_years"}
print(f"\nPlanted interactions in quick top-5:   "
      f"{sum(1 for p in quick_top5 if p in planted)}/2")
print(f"Planted interactions in careful top-5: "
      f"{sum(1 for p in careful_top5 if p in planted)}/2")

# Task 5: Score ratio
careful_nid = detector_careful.nid_table()
score_1  = careful_nid["nid_score_normalised"][0]
score_10 = careful_nid["nid_score_normalised"][9]
print(f"\nTask 5: NID score rank 1:  {score_1:.4f}")
print(f"         NID score rank 10: {score_10:.4f}")
print(f"         Ratio (10th / 1st): {score_10 / score_1:.4f}")
```

**What to look for:**

- The quick CANN will often miss one of the planted interactions in its top-5, or rank it lower, because a single underconverged training run produces noisier weight matrices.
- The careful CANN's ensemble averaging produces more stable scores. The two planted interactions should reliably appear in positions 1-3.
- If `score_10 / score_1 < 0.15`, the 10th pair has very weak NID support. You should be sceptical of any interaction ranked below a cliff in the NID scores.
- The "still improving" flag for the quick CANN is the key diagnostic: if the validation curve was still falling at epoch 50, the weights were not at their optimal point, and the NID scores extracted from them are unreliable.

**On presenting to a pricing committee:** The quick CANN's output should not be used for a committee presentation without the LR test results as a backstop. The NID ranking from a single underconverged run is too noisy to present as evidence. The careful CANN's ensemble-averaged scores, combined with the LR test results, provide the audit trail needed under PRA SS1/23.

</details>

---

## Exercise 3: The MLP-M variant and correlated features

**Reference:** Tutorial Parts 2, 3; README section "MLP-M variant"

**What you will do:** Train the MLP-M variant of the CANN, compare its NID scores to the standard variant, and understand why MLP-M matters when features are correlated.

**Context.** In UK motor insurance, `ncd_years` and `conviction_points` are negatively correlated: drivers with more NCD years have typically had fewer accidents and convictions. In the synthetic dataset, this correlation is mild (it was generated independently). In a real portfolio, `driver_age` and `ncd_years` are strongly correlated: 20-year-olds cannot have more than 3 years of NCD. This correlation can cause the standard CANN to spread interaction signal across spurious pairs. MLP-M mitigates this.

### Setup

```python
%md ## Exercise 3: MLP-M and feature correlation
```

### Tasks

**Task 1.** Compute the Pearson correlation between `ncd_years` and `conviction_points` in the synthetic dataset. Then compute the Spearman rank correlation between `driver_age` (the original continuous variable) and `ncd_years`. Which pair is more correlated?

```python
import numpy as np

# The original continuous variables are available from the data generation in Part 5
# If you no longer have them in scope, re-run the data generation cell
# We use the banded versions for the GLM but need the continuous versions here
print("ncd_years unique values:", sorted(X["ncd_years"].unique().to_list()))
print("conviction_points unique values:", sorted(X["conviction_points"].unique().to_list()))

# Pearson correlation
ncd_arr = X["ncd_years"].to_numpy().astype(float)
cv_arr  = X["conviction_points"].to_numpy().astype(float)
r_ncd_cv = np.corrcoef(ncd_arr, cv_arr)[0, 1]
print(f"\nPearson r(ncd_years, conviction_points): {r_ncd_cv:.4f}")
```

**Task 2.** Train a standard CANN (already done in Exercise 2 as `detector_careful`) and an MLP-M CANN on the same data. Use `cann_n_epochs=300, cann_patience=30, cann_n_ensemble=3, mlp_m=True` for the MLP-M variant.

```python
cfg_mlpm = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_patience=30,
    cann_n_ensemble=3,
    top_k_nid=15,
    top_k_final=5,
    mlp_m=True,
)
detector_mlpm = InteractionDetector(family="poisson", config=cfg_mlpm)
detector_mlpm.fit(X=X, y=y, glm_predictions=mu_glm, exposure=exposure_arr)
```

**Task 3.** Compare the NID tables from the standard and MLP-M detectors side-by-side. For each pair that appears in the top 10 of the standard CANN, what rank does it get in the MLP-M CANN? Are there pairs that the standard CANN ranks highly but the MLP-M CANN ranks much lower?

**Task 4.** Look at the LR test results for both detectors using `.glm_test_table()`. For pairs that the standard CANN ranks highly but MLP-M ranks low, does the LR test agree with MLP-M (i.e., those pairs are not statistically significant)?

**Task 5.** Write a markdown cell in your notebook that states your recommendation: for this specific synthetic dataset, should you use the standard CANN or MLP-M, and why? Consider: the feature correlation level, the dataset size, and whether the LR tests provide a sufficient safety net even if the NID ranking is noisy.

<details>
<summary>Hint for Task 3</summary>

Build a lookup from the MLP-M NID table: `{(feat1, feat2): rank}` and then look up each standard CANN pair.

For pairs that are only in the standard CANN's top 10 and not in MLP-M's top 10, they may have been "inflated" by the standard CANN absorbing main effect residuals via interaction structure. MLP-M's univariate sub-networks absorb those main effect residuals directly, leaving only genuine cross-feature signal.

</details>

<details>
<summary>Solution -- Exercise 3</summary>

```python
import numpy as np
import polars as pl

# Task 1: Correlations
ncd_arr = X["ncd_years"].to_numpy().astype(float)
cv_arr  = X["conviction_points"].to_numpy().astype(float)
r_ncd_cv = np.corrcoef(ncd_arr, cv_arr)[0, 1]
print(f"Pearson r(ncd_years, conviction_points): {r_ncd_cv:.4f}")

# Spearman rank correlation requires scipy
from scipy.stats import spearmanr
rho, pval = spearmanr(ncd_arr, cv_arr)
print(f"Spearman rho(ncd_years, conviction_points): {rho:.4f} (p={pval:.4f})")

# Task 2: Already trained detector_careful above; now train MLP-M
cfg_mlpm = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_patience=30,
    cann_n_ensemble=3,
    top_k_nid=15,
    top_k_final=5,
    mlp_m=True,
)
detector_mlpm = InteractionDetector(family="poisson", config=cfg_mlpm)
print("Training MLP-M CANN (3 runs x 300 epochs)...")
detector_mlpm.fit(X=X, y=y, glm_predictions=mu_glm, exposure=exposure_arr)
print("MLP-M training complete.")

# Task 3: Side-by-side NID comparison
std_nid  = detector_careful.nid_table().with_row_index("std_rank",  offset=1)
mlpm_nid = detector_mlpm.nid_table().with_row_index("mlpm_rank", offset=1)

# Lookup dict for MLP-M ranks
mlpm_lookup = {
    (r["feature_1"], r["feature_2"]): r["mlpm_rank"]
    for r in mlpm_nid.iter_rows(named=True)
}

print("\nTask 3: Standard CANN top-10 with MLP-M ranks")
print(f"{'Std rank':<10} {'Pair':<40} {'Std score':<12} {'MLP-M rank':<12}")
for r in std_nid.head(10).iter_rows(named=True):
    pair  = (r["feature_1"], r["feature_2"])
    mprank = mlpm_lookup.get(pair, mlpm_lookup.get((pair[1], pair[0]), "N/A"))
    print(f"{r['std_rank']:<10} {r['feature_1']} x {r['feature_2']:<28} "
          f"{r['nid_score_normalised']:.4f}      {str(mprank):<12}")

# Task 4: LR test table comparison
print("\nTask 4: GLM test results (standard CANN)")
print(
    detector_careful.glm_test_table().select([
        "feature_1", "feature_2", "delta_deviance", "lr_p", "recommended"
    ])
)
print("\nGLM test results (MLP-M CANN)")
print(
    detector_mlpm.glm_test_table().select([
        "feature_1", "feature_2", "delta_deviance", "lr_p", "recommended"
    ])
)

# Task 5: Which is recommended?
std_rec  = detector_careful.glm_test_table().filter(pl.col("recommended") == True).height
mlpm_rec = detector_mlpm.glm_test_table().filter(pl.col("recommended") == True).height
print(f"\nStandard CANN: {std_rec} recommended interactions")
print(f"MLP-M CANN:    {mlpm_rec} recommended interactions")
```

**What you should see:**

- `r(ncd_years, conviction_points)` should be mildly negative (around -0.10 to -0.20) because the data was generated independently with minor implicit correlation from the base rate.
- The two planted interactions should appear in both CANNs' top-5 NID rankings, but the order of pairs ranked 3-10 may differ.
- The LR test is the safety net: pairs that the standard CANN inflates due to main effect residual leakage will typically fail the LR test. For this dataset, the results should be largely consistent between the two approaches because the correlation is mild.
- **Recommendation for this dataset:** Standard CANN is adequate. MLP-M becomes important when features are structurally correlated (real UK motor, where age and NCD are nearly deterministically linked for young drivers). The LR test catches the false positives either way, so the practical impact on the final model is small -- but MLP-M produces a cleaner NID ranking, which matters when presenting the methodology to a technical reviewer.

</details>

---

## Exercise 4: Manual LR testing without the pipeline

**Reference:** Tutorial Part 10

**What you will do:** Run the likelihood-ratio test manually for one candidate pair, then verify your result matches what `test_interactions()` produces. This exercise ensures you understand the test, not just how to call the library.

**Context.** The `test_interactions()` function runs 15 LR tests automatically. Before trusting it, you should be able to reproduce one result by hand using `glum` directly.

### Setup

```python
from glum import GeneralizedLinearRegressor
import pandas as pd
import scipy.stats
import numpy as np

%md ## Exercise 4: Manual likelihood-ratio test
```

### Tasks

**Task 1.** Fit the baseline Poisson GLM using `glum.GeneralizedLinearRegressor` with all six main effects. Compute the total Poisson deviance manually using the formula:

```sql
deviance = 2 * sum(exposure * (y * log(y / mu) - (y - mu)))
```

where the sum over rows with `y == 0` contributes `2 * exposure * mu` (since `0 * log(0) = 0`). Call this `dev_base`.

```python
import pandas as pd
from glum import GeneralizedLinearRegressor

X_pd = X.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_pd[col] = pd.Categorical(X_pd[col].astype(str))

glm_manual = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True)
glm_manual.fit(X_pd, y, sample_weight=exposure_arr)
mu_manual = glm_manual.predict(X_pd)

# Your deviance calculation here
# Handle y == 0 carefully
```

**Task 2.** Now add a categorical-by-categorical interaction for `age_band × vehicle_group`. Create the combined categorical column by concatenating the string values of both features:

```python
X_int = X_pd.copy()
X_int["_ix_age_band_vehicle_group"] = pd.Categorical(
    X_pd["age_band"].astype(str) + "_X_" + X_pd["vehicle_group"].astype(str)
)
```

Fit a second GLM on `X_int` and compute `dev_interaction`.

**Task 3.** Compute the LR test statistic and p-value:

```bash
chi2 = dev_base - dev_interaction
df   = n_cells  # (L_age - 1) * (L_vg - 1)
p    = 1 - chi2_cdf(chi2, df=df)
```

How many levels does `age_band` have? How many does `vehicle_group` (banded) have? What is `n_cells`?

**Task 4.** Call `detector_careful.glm_test_table()` and find the row for `age_band × vehicle_group`. Compare your manual `chi2`, `df`, and `p` to the library's values. They should match to within floating-point rounding.

**Task 5.** Now compute `delta_aic` and `delta_bic` for this interaction:

```python
aic = deviance + 2 * n_params
bic = deviance + log(n) * n_params
```

Is `delta_aic` negative? Is `delta_bic` negative? If BIC is positive (the interaction worsens BIC) but AIC is negative (it improves AIC), what does this tell you about the trade-off between fit and complexity at this sample size?

<details>
<summary>Hint for Task 1</summary>

```python
mu = np.clip(mu_manual, 1e-8, None)
log_term = np.where(y > 0, y * np.log(y / mu), 0.0)
dev_base = 2.0 * np.sum(exposure_arr * (log_term - (y - mu)))
```

The `np.where` avoids `0 * log(0)` which is mathematically 0 but numerically produces `nan`.

</details>

<details>
<summary>Solution -- Exercise 4</summary>

```python
import pandas as pd
import numpy as np
import scipy.stats
from glum import GeneralizedLinearRegressor

# Task 1: Baseline GLM and deviance
X_pd = X.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_pd[col] = pd.Categorical(X_pd[col].astype(str))
for col in ["ncd_years", "conviction_points"]:
    X_pd[col] = X_pd[col].astype(float)

glm_manual = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True)
glm_manual.fit(X_pd, y, sample_weight=exposure_arr)
mu_base = np.clip(glm_manual.predict(X_pd), 1e-8, None)

log_term = np.where(y > 0, y * np.log(y / mu_base), 0.0)
dev_base = 2.0 * np.sum(exposure_arr * (log_term - (y - mu_base)))
n_params_base = len(glm_manual.coef_) + 1
print(f"Task 1: Baseline deviance: {dev_base:,.2f}")
print(f"        Parameters:        {n_params_base}")

# Task 2: Add age_band x vehicle_group interaction
X_int = X_pd.copy()
X_int["_ix_age_band_vehicle_group"] = pd.Categorical(
    X_pd["age_band"].astype(str) + "_X_" + X_pd["vehicle_group"].astype(str)
)

glm_int = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True)
glm_int.fit(X_int, y, sample_weight=exposure_arr)
mu_int = np.clip(glm_int.predict(X_int), 1e-8, None)

log_term_int = np.where(y > 0, y * np.log(y / mu_int), 0.0)
dev_int = 2.0 * np.sum(exposure_arr * (log_term_int - (y - mu_int)))
n_params_int = len(glm_int.coef_) + 1
print(f"\nTask 2: Interaction GLM deviance: {dev_int:,.2f}")
print(f"        Parameters:               {n_params_int}")

# Task 3: LR test
n_age  = X["age_band"].n_unique()
n_vg   = X["vehicle_group"].n_unique()
n_cells = (n_age - 1) * (n_vg - 1)
chi2   = dev_base - dev_int
p_val  = float(scipy.stats.chi2.sf(chi2, df=n_cells))

print(f"\nTask 3: LR test")
print(f"  age_band levels:     {n_age}")
print(f"  vehicle_group levels: {n_vg}")
print(f"  n_cells (df):        {n_cells}")
print(f"  chi2:                {chi2:.4f}")
print(f"  p-value:             {p_val:.6e}")

# Task 4: Compare to library
lib_table = detector_careful.glm_test_table()
lib_row = lib_table.filter(
    (pl.col("feature_1") == "age_band") & (pl.col("feature_2") == "vehicle_group")
)
if lib_row.is_empty():
    lib_row = lib_table.filter(
        (pl.col("feature_1") == "vehicle_group") & (pl.col("feature_2") == "age_band")
    )
print(f"\nTask 4: Library values for age_band x vehicle_group:")
print(lib_row.select(["feature_1", "feature_2", "n_cells", "lr_chi2", "lr_p"]))
print(f"Manual chi2: {chi2:.4f}  (vs library: {lib_row['lr_chi2'][0]:.4f})")
print(f"Manual p:    {p_val:.6e}  (vs library: {lib_row['lr_p'][0]:.6e})")

# Task 5: AIC and BIC
n = len(X)
aic_base = dev_base + 2 * n_params_base
aic_int  = dev_int  + 2 * n_params_int
bic_base = dev_base + np.log(n) * n_params_base
bic_int  = dev_int  + np.log(n) * n_params_int

delta_aic = aic_int - aic_base
delta_bic = bic_int - bic_base

print(f"\nTask 5: AIC/BIC analysis")
print(f"  delta_AIC: {delta_aic:+,.1f}  ({'better' if delta_aic < 0 else 'worse'})")
print(f"  delta_BIC: {delta_bic:+,.1f}  ({'better' if delta_bic < 0 else 'worse'})")
print(f"  n_new_params: {n_params_int - n_params_base}")
print(f"  AIC penalty per param: 2.0")
print(f"  BIC penalty per param: {np.log(n):.2f}")
```

**Expected results:**

- `dev_base` and `dev_int` should match `mu_glm`'s baseline deviance to within 0.1% (the library uses the same GLM specification).
- `chi2` should be several hundred to a few thousand; `p_val` should be extremely small (typically < 1e-50 for the primary planted interaction).
- `delta_aic` should be strongly negative (the interaction is highly worth its parameter cost). `delta_bic` should also be negative, but less so: with n=100,000, `log(n) ≈ 11.5`, so BIC penalises each parameter about 5.75 times more heavily than AIC. For the large planted interaction, both should improve.

**If BIC worsens but AIC improves:** This would indicate the interaction is real but expensive. With 100,000 policies this is unlikely for the primary planted interaction, but it can happen for the secondary one (ncd × conviction_points) if the n_cells count is high relative to the deviance gain.

</details>

---

## Exercise 5: Building the enhanced GLM and inspecting its coefficients

**Reference:** Tutorial Parts 12, 13

**What you will do:** Use `build_glm_with_interactions()` to refit the GLM with the detected interactions, then carefully interpret the interaction coefficient values.

**Context.** The `detector_careful` has been fitted. You have the suggested interactions. Now you will build the final enhanced GLM, verify that the planted interaction coefficients are recovered accurately, and understand what the output means for pricing.

### Setup

```python
from insurance_interactions import build_glm_with_interactions
import numpy as np

%md ## Exercise 5: Enhanced GLM coefficients
```

### Tasks

**Task 1.** Get the suggested interactions from `detector_careful`:

```python
suggested = detector_careful.suggest_interactions(top_k=5)
print("Suggested interactions:", suggested)
```

Then call `build_glm_with_interactions()` with `suggested`. Print the comparison table.

**Task 2.** The comparison table has `delta_deviance` for the joint model. Compare this to the sum of individual `delta_deviance` values from `detector_careful.glm_test_table()` for the same pairs. The joint improvement should be less than the sum of individual improvements. Compute the "overlap fraction":

```bash
overlap_fraction = 1 - (joint_delta_deviance / sum_of_individual_delta_deviances)
```

What does an overlap fraction of 0.30 mean in plain language?

**Task 3.** Inspect the interaction coefficients in the enhanced GLM. The library creates interaction columns named `_ix_{feat1}_{feat2}`. List all such columns and their coefficients:

```python
coef_names = list(enhanced_glm.feature_names_in_)
ix_cols    = [c for c in coef_names if c.startswith("_ix_")]
ix_coefs   = [enhanced_glm.coef_[coef_names.index(c)] for c in ix_cols]

print(f"{'Interaction term':<55} {'Coef':>8}  {'Relativity':>10}")
for name, coef in sorted(zip(ix_cols, ix_coefs), key=lambda x: abs(x[1]), reverse=True)[:15]:
    print(f"{name:<55} {coef:+.4f}  {np.exp(coef):>10.3f}")
```

**Task 4.** The planted interaction for `age_band × vehicle_group` was a log-additive 0.30 penalty for policies where `age_band ∈ {17-21, 22-25}` AND `vehicle_group ∈ {41-50}`. In the GLM, this interaction is represented as coefficients on the combined categorical `_ix_age_band_vehicle_group`. Find the coefficients for the cells corresponding to (17-21, 41-50) and (22-25, 41-50). Are they close to +0.30? Why might they differ?

**Task 5.** The second planted interaction (`ncd_years == 0` AND `conviction_points > 0`) was a log-additive 0.20 penalty. The library encodes this as a `_ix_ncd_years_conviction_points` column. Look at the coefficients for cells with `ncd_years=0` and `conviction_points ∈ {3, 6, 9}`. Are they close to +0.20? Do they vary across conviction_points levels (i.e., does the GLM give a different coefficient for 3 points vs 9 points)?

The planted interaction was a flat 0.20 regardless of the number of conviction points. Does the GLM recover this flatness, or does it over-fit to the specific levels?

<details>
<summary>Hint for Task 4</summary>

The combined categorical interaction column contains strings of the form `"17-21_X_41-50"`. To find the coefficient for a specific cell, match the column name:

```python
target = "_ix_age_band_vehicle_group"
# Find the level "17-21_X_41-50" in the GLM's design matrix
# The GLM encodes categoricals with one level as baseline
# Look at coef_names for entries like "_ix_age_band_vehicle_group[17-21_X_41-50]"
```

glum encodes categorical levels using bracket notation: `feature_name[level_value]`.

</details>

<details>
<summary>Solution -- Exercise 5</summary>

```python
from insurance_interactions import build_glm_with_interactions
import numpy as np
import polars as pl

# Task 1: Build enhanced GLM
suggested = detector_careful.suggest_interactions(top_k=5)
print("Suggested interactions:", suggested)

enhanced_glm, comparison = build_glm_with_interactions(
    X=X,
    y=y,
    exposure=exposure_arr,
    interaction_pairs=suggested,
    family="poisson",
)
print("\nModel comparison:")
print(comparison)

# Task 2: Overlap fraction
joint_delta = comparison.filter(
    pl.col("model") == "glm_with_interactions"
)["delta_deviance"][0]

test_table = detector_careful.glm_test_table()
# Sum individual deviance gains for the suggested pairs
sum_individual = 0.0
for f1, f2 in suggested:
    row = test_table.filter(
        (pl.col("feature_1") == f1) & (pl.col("feature_2") == f2)
    )
    if row.is_empty():
        row = test_table.filter(
            (pl.col("feature_1") == f2) & (pl.col("feature_2") == f1)
        )
    if not row.is_empty():
        sum_individual += row["delta_deviance"][0]

overlap_fraction = 1.0 - (joint_delta / sum_individual) if sum_individual > 0 else 0.0
print(f"\nTask 2: Deviance overlap analysis")
print(f"  Joint delta_deviance:       {joint_delta:,.1f}")
print(f"  Sum of individual deltas:   {sum_individual:,.1f}")
print(f"  Overlap fraction:           {overlap_fraction:.3f}")
print(f"  ({overlap_fraction*100:.1f}% of individual gains are shared across the approved interactions)")

# Task 3: Interaction coefficients
coef_names = list(enhanced_glm.feature_names_in_)
ix_cols  = [c for c in coef_names if "_ix_" in c]
ix_coefs = [enhanced_glm.coef_[coef_names.index(c)] for c in ix_cols]

print(f"\nTask 3: All {len(ix_cols)} interaction coefficients")
print(f"{'Term':<60} {'Coef':>8}  {'Relativity':>10}")
for name, coef in sorted(zip(ix_cols, ix_coefs), key=lambda x: abs(x[1]), reverse=True)[:20]:
    print(f"{name:<60} {coef:+.4f}  {np.exp(coef):>10.3f}")

# Task 4: Recover the planted age x vg interaction
print("\nTask 4: Planted interaction recovery (age_band x vehicle_group)")
print("  Planted: +0.30 log-units for (17-21 or 22-25) x (41-50)")
print("  Expected relativity: exp(0.30) = 1.350")
print()
target_cells = ["17-21_X_41-50", "22-25_X_41-50"]
for cell in target_cells:
    col = f"_ix_age_band_vehicle_group[{cell}]"
    if col in coef_names:
        coef = enhanced_glm.coef_[coef_names.index(col)]
        print(f"  {cell}: coef = {coef:+.4f}  (relativity = {np.exp(coef):.3f})")
    else:
        # Try alternative naming
        matching = [c for c in ix_cols if cell in c]
        for c in matching:
            coef = enhanced_glm.coef_[coef_names.index(c)]
            print(f"  {c}: coef = {coef:+.4f}  (relativity = {np.exp(coef):.3f})")

# Task 5: Recover the planted ncd x conviction interaction
print("\nTask 5: Planted interaction recovery (ncd_years x conviction_points)")
print("  Planted: +0.20 log-units for ncd_years=0, conviction_points in {3,6,9}")
print("  Expected relativity: exp(0.20) = 1.221")
ncd_ix_cols = [(c, enhanced_glm.coef_[coef_names.index(c)])
               for c in ix_cols if "ncd" in c and "conviction" in c]
for name, coef in sorted(ncd_ix_cols, key=lambda x: x[0]):
    if "0_X" in name or "_X_0" in name or "ncd_0" in name or "0_" in name:
        if any(str(cv) in name for cv in [3, 6, 9]):
            print(f"  {name}: coef = {coef:+.4f}  (relativity = {np.exp(coef):.3f})")
```

**What you should see:**

- **Joint vs individual overlap:** An overlap fraction of 0.25-0.40 is typical when two interactions share a feature (`age_band` appears in both `age_band × vehicle_group` and potentially `age_band × annual_mileage`). The joint gain is less than the sum because adding the first interaction absorbs some of the residual that the second interaction would have improved.

- **Task 4 recovery:** The interaction coefficients for (17-21, 41-50) and (22-25, 41-50) should be in the range +0.20 to +0.35. They will not be exactly +0.30 because: (1) the GLM adds an interaction parameter for every (age, vg) cell combination, not just the planted ones, so the estimation is noisy; (2) the intercept and main effects absorb some of the interaction signal; and (3) finite sample size means the MLE has sampling error.

- **Task 5 recovery:** For the flat 0.20 interaction, you may see coefficients ranging from +0.10 to +0.30 across the three conviction_points levels. The GLM gives each cell a separate estimate rather than knowing the true structure is flat. This is expected: the GLM does not know the data generation process. If the GLM's estimates vary substantially across conviction_points levels despite a flat planted interaction, this is evidence of noise -- which is exactly why you want credibility smoothing (the subject of the Bayesian pricing module).

</details>

---

## Exercise 6: Train/test split and out-of-sample validation

**Reference:** Tutorial Part 17

**What you will do:** Split the data into train and test sets, fit the full pipeline on training data only, and evaluate the interaction-enhanced GLM on held-out data.

**Context.** All the work so far has been in-sample. Adding 20-25 parameters to a GLM and measuring in-sample deviance improvement is not a fair test: it will always improve. The honest assessment is whether the interactions improve out-of-sample performance.

### Setup

```python
%md ## Exercise 6: Out-of-sample validation
```

### Tasks

**Task 1.** Split the data 80/20 into training and test sets. Use the first 80,000 rows for training and the last 20,000 for testing (not random -- keeping the split deterministic means your results are reproducible).

```python
n_train = 80_000

X_train = X[:n_train]
X_test  = X[n_train:]
y_train = y[:n_train]
y_test  = y[n_train:]
exp_train = exposure_arr[:n_train]
exp_test  = exposure_arr[n_train:]

print(f"Train: {len(X_train):,} policies, {y_train.sum():.0f} claims")
print(f"Test:  {len(X_test):,} policies, {y_test.sum():.0f} claims")
```

**Task 2.** Fit the baseline Poisson GLM on training data only. Get the training-set GLM predictions `mu_glm_train`. Then fit the `InteractionDetector` on training data only. Get the suggested interactions.

**Task 3.** Build the enhanced GLM on training data only. Compute test-set deviance for both the baseline GLM and the enhanced GLM. To score the test set with the enhanced GLM, you need to construct the interaction columns in `X_test`:

```python
import pandas as pd

X_test_pd = X_test.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_test_pd[col] = pd.Categorical(X_test_pd[col].astype(str))

# Add interaction columns for each suggested pair
for f1, f2 in suggested_train:
    X_test_pd[f"_ix_{f1}_{f2}"] = pd.Categorical(
        X_test_pd[f1].astype(str) + "_X_" + X_test_pd[f2].astype(str)
    )

mu_enhanced_test = enhanced_glm_train.predict(X_test_pd)
```

**Task 4.** Compute the out-of-sample deviance improvement:

```sql
oos_improvement_pct = (dev_base_test - dev_enhanced_test) / dev_base_test * 100
```

Compare this to the in-sample improvement from Exercise 5. Is the out-of-sample improvement smaller than the in-sample improvement? By how much? What does the difference tell you about over-fitting?

**Task 5.** Compute a lift chart: divide the test set into 10 deciles by the enhanced GLM's predicted frequency, and within each decile plot the average predicted frequency vs average observed frequency. Do the same for the baseline GLM. Which model has better calibration in the top decile (the highest-risk segment)? This is where the interaction terms are doing most of their work.

<details>
<summary>Hint for Task 3</summary>

When scoring the test set, you must construct the interaction categorical column with the same levels that appeared in training. If a (age_band, vehicle_group) combination appears in the test set but not in training, the GLM will assign a coefficient of 0 for that level (because it has no estimate). This is correct behaviour: unknown level = baseline.

Be careful: `pd.Categorical` on the test set interaction column will only have the levels seen in the test set. The GLM may expect the training-set categorical levels. If you get a `ValueError` from glum about unseen levels, convert the interaction column to a string rather than `pd.Categorical`.

</details>

<details>
<summary>Solution -- Exercise 6</summary>

```python
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from glum import GeneralizedLinearRegressor
from insurance_interactions import InteractionDetector, DetectorConfig, build_glm_with_interactions

# Task 1: Train/test split
n_train = 80_000
X_train, X_test   = X[:n_train],       X[n_train:]
y_train, y_test   = y[:n_train],       y[n_train:]
exp_train, exp_test = exposure_arr[:n_train], exposure_arr[n_train:]

print(f"Train: {len(X_train):,} policies, {y_train.sum():.0f} claims")
print(f"Test:  {len(X_test):,} policies, {y_test.sum():.0f} claims")

# Task 2: Fit GLM and detector on train only
X_train_pd = X_train.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_train_pd[col] = pd.Categorical(X_train_pd[col].astype(str))

glm_train = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True)
glm_train.fit(X_train_pd, y_train, sample_weight=exp_train)
mu_glm_train = glm_train.predict(X_train_pd)

cfg_oos = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_patience=30,
    cann_n_ensemble=3,
    top_k_nid=15,
    top_k_final=5,
)
detector_train = InteractionDetector(family="poisson", config=cfg_oos)
print("Training detector on train split...")
detector_train.fit(X=X_train, y=y_train, glm_predictions=mu_glm_train, exposure=exp_train)
suggested_train = detector_train.suggest_interactions(top_k=3)
print(f"Suggested interactions from train split: {suggested_train}")

# Task 3: Build enhanced GLM on train, score test
enhanced_glm_train, comp_train = build_glm_with_interactions(
    X=X_train, y=y_train, exposure=exp_train,
    interaction_pairs=suggested_train, family="poisson",
)
print("\nIn-sample comparison (train set):")
print(comp_train)

# Score test set
X_test_pd = X_test.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_test_pd[col] = pd.Categorical(X_test_pd[col].astype(str))

X_test_int = X_test_pd.copy()
for f1, f2 in suggested_train:
    # Use string (not Categorical) to handle unseen levels gracefully
    X_test_int[f"_ix_{f1}_{f2}"] = (
        X_test_pd[f1].astype(str) + "_X_" + X_test_pd[f2].astype(str)
    )

mu_base_test     = np.clip(glm_train.predict(X_test_pd), 1e-8, None)
mu_enhanced_test = np.clip(enhanced_glm_train.predict(X_test_int), 1e-8, None)

# Task 4: OOS deviance improvement
def poisson_deviance(y_true, y_pred, weights):
    mu = np.clip(y_pred, 1e-8, None)
    log_term = np.where(y_true > 0, y_true * np.log(y_true / mu), 0.0)
    return 2.0 * float(np.sum(weights * (log_term - (y_true - mu))))

dev_base_test     = poisson_deviance(y_test, mu_base_test,     exp_test)
dev_enhanced_test = poisson_deviance(y_test, mu_enhanced_test, exp_test)
oos_improvement   = (dev_base_test - dev_enhanced_test) / dev_base_test * 100

# In-sample
dev_base_train     = poisson_deviance(y_train, mu_glm_train, exp_train)
mu_enhanced_train  = np.clip(enhanced_glm_train.predict(
    pd.concat([X_train_pd] + [
        pd.DataFrame({f"_ix_{f1}_{f2}": (X_train_pd[f1].astype(str) + "_X_" + X_train_pd[f2].astype(str))
                      for f1, f2 in suggested_train})
    ], axis=1)
), 1e-8, None)
dev_enhanced_train = poisson_deviance(y_train, enhanced_glm_train.predict(
    pd.DataFrame(X_train_pd).assign(**{
        f"_ix_{f1}_{f2}": X_train_pd[f1].astype(str) + "_X_" + X_train_pd[f2].astype(str)
        for f1, f2 in suggested_train
    })
), exp_train)
is_improvement = (dev_base_train - dev_enhanced_train) / dev_base_train * 100

print(f"\nTask 4: Deviance improvement")
print(f"  In-sample:      {is_improvement:.4f}%")
print(f"  Out-of-sample:  {oos_improvement:.4f}%")
print(f"  Over-fit ratio: {is_improvement / oos_improvement:.3f}  (1.0 = no over-fit)")

# Task 5: Lift chart
test_df = pd.DataFrame({
    "y":         y_test,
    "exposure":  exp_test,
    "mu_base":   mu_base_test,
    "mu_enhanced": mu_enhanced_test,
})
test_df["decile"] = pd.qcut(test_df["mu_enhanced"], 10, labels=False)

lift = test_df.groupby("decile").agg(
    obs_freq=("y", "sum"),
    base_freq=("mu_base", "sum"),
    enh_freq=("mu_enhanced", "sum"),
    exposure=("exposure", "sum"),
).reset_index()
lift["obs_rate"]  = lift["obs_freq"]  / lift["exposure"]
lift["base_rate"] = lift["base_freq"] / lift["exposure"]
lift["enh_rate"]  = lift["enh_freq"]  / lift["exposure"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lift["decile"], lift["obs_rate"],  "o-", label="Observed",  color="black")
ax.plot(lift["decile"], lift["base_rate"], "s--", label="Base GLM", color="#e07b39")
ax.plot(lift["decile"], lift["enh_rate"],  "^-",  label="Enhanced GLM", color="#2271b3")
ax.set_xlabel("Predicted frequency decile (0=lowest, 9=highest)")
ax.set_ylabel("Mean claim frequency")
ax.set_title("Lift chart: baseline vs interaction-enhanced GLM")
ax.legend()
plt.tight_layout()
plt.show()

# Zoom in on top decile
top = lift[lift["decile"] == 9]
print(f"\nTask 5: Top decile (decile 9)")
print(f"  Observed frequency:        {top['obs_rate'].values[0]:.5f}")
print(f"  Base GLM predicted:        {top['base_rate'].values[0]:.5f}")
print(f"  Enhanced GLM predicted:    {top['enh_rate'].values[0]:.5f}")
```

**What you should see:**

- Out-of-sample improvement should be in the range 1.0-1.8%, compared to an in-sample improvement of 1.5-2.0%. An over-fit ratio close to 1.0 means the interactions generalise well. Ratios above 1.3 would suggest over-fitting.
- In the lift chart, the enhanced GLM should track observed frequencies more closely in the top decile. This is where the young driver × high vehicle group interaction lives: the baseline GLM underpredicts this segment, and the enhanced GLM corrects it.

</details>

---

## Exercise 7: SHAP interaction values as a second opinion

**Reference:** Tutorial Part 14

**What you will do:** Train a CatBoost model, compute SHAP interaction values, and compare the SHAP ranking to the NID ranking. Understand where and why they disagree.

**This exercise requires:** `catboost` and `shapiq`. If these are not installed, run:

```python
%pip install catboost shapiq --quiet
dbutils.library.restartPython()
```

### Setup

```python
try:
    from catboost import CatBoostRegressor, Pool
    import shapiq
    SHAP_AVAILABLE = True
    print("CatBoost and shapiq available.")
except ImportError:
    SHAP_AVAILABLE = False
    print("Skipping Exercise 7 -- catboost or shapiq not installed.")

%md ## Exercise 7: SHAP interaction validation
```

### Tasks

**Task 1.** Train a CatBoost Poisson model on the full training data (use `X`, `y`, `exposure_arr` -- the full 100,000 policies, not the 80/20 split). Use `iterations=300, depth=6, learning_rate=0.05, loss_function="Poisson", random_seed=42`.

After training, print the CatBoost model's in-sample Poisson deviance. Is it lower than the enhanced GLM's deviance? (It should be -- CatBoost can capture interactions automatically.)

```python
if SHAP_AVAILABLE:
    cat_features = ["area", "vehicle_group", "age_band", "annual_mileage"]
    X_pd_cb = X.to_pandas()
    for col in cat_features:
        X_pd_cb[col] = X_pd_cb[col].astype(str)

    pool_full = Pool(
        X_pd_cb,
        label=y,
        weight=exposure_arr,
        cat_features=cat_features,
    )
    cb_model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Poisson",
        random_seed=42,
        verbose=False,
    )
    cb_model.fit(pool_full)
    mu_cb = np.clip(cb_model.predict(pool_full), 1e-8, None)
    dev_cb = poisson_deviance(y, mu_cb, exposure_arr)
    print(f"CatBoost deviance: {dev_cb:,.1f}")
    print(f"Enhanced GLM deviance: {comparison['deviance'][1]:,.1f}")
```

**Task 2.** Re-run the `InteractionDetector` with the CatBoost model passed as `shap_model`. This triggers SHAP interaction computation inside the pipeline.

```python
if SHAP_AVAILABLE:
    cfg_shap = DetectorConfig(
        cann_hidden_dims=[32, 16],
        cann_n_epochs=300,
        cann_patience=30,
        cann_n_ensemble=3,
        top_k_nid=15,
        top_k_final=5,
    )
    detector_shap = InteractionDetector(family="poisson", config=cfg_shap)
    detector_shap.fit(
        X=X, y=y,
        glm_predictions=mu_glm,
        exposure=exposure_arr,
        shap_model=cb_model,
    )
```

**Task 3.** Print the interaction table from `detector_shap.interaction_table()`. This table now includes `nid_score_normalised`, `shap_score_normalised`, `nid_rank`, `shap_rank`, and `consensus_score`. Find the rows where:

- NID rank is high (top 5) but SHAP rank is low (below 8)
- SHAP rank is high (top 5) but NID rank is low (below 8)

For each disagreement, write one sentence explaining a plausible reason why the two methods might disagree for that pair.

**Task 4.** The two planted interactions should appear in both the NID top-5 and the SHAP top-5. Do they? If not, which method missed which planted interaction?

**Task 5.** Does the consensus ranking (which combines NID and SHAP ranks) produce a better shortlist than either method alone? To answer this, check whether the planted interactions both appear in the top-3 consensus-ranked pairs.

<details>
<summary>Hint for Task 3</summary>

SHAP interaction values measure the combined perturbation effect of two features simultaneously, relative to their individual effects. For features that are correlated (like `ncd_years` and `driver_age` in a real portfolio), the SHAP interaction values are sensitive to the choice of reference distribution. In our synthetic dataset, features are mostly independent, so SHAP and NID should agree closely.

If you see a disagreement, the most common cause in this dataset is that one CANN ensemble run produced an unstable weight matrix for a pair where the interaction signal is weak. The ensemble averaging reduces but does not eliminate this noise.

</details>

<details>
<summary>Solution -- Exercise 7</summary>

```python
if SHAP_AVAILABLE:
    import polars as pl

    # Task 1: CatBoost model (already fitted above)
    dev_cb = poisson_deviance(y, mu_cb, exposure_arr)
    dev_enhanced = comparison.filter(
        pl.col("model") == "glm_with_interactions"
    )["deviance"][0]

    print(f"CatBoost deviance:    {dev_cb:,.1f}")
    print(f"Enhanced GLM deviance: {dev_enhanced:,.1f}")
    print(f"Remaining gap:         {dev_cb - dev_enhanced:,.1f}  "
          f"(CatBoost {'better' if dev_cb < dev_enhanced else 'worse'})")

    # Task 2: Already fitted detector_shap above

    # Task 3: Compare NID and SHAP ranks
    tbl = detector_shap.interaction_table()
    print("\nFull interaction table with NID vs SHAP ranks:")
    print(
        tbl.select([
            "feature_1", "feature_2",
            "nid_score_normalised", "shap_score_normalised",
            "nid_rank", "shap_rank",
            "consensus_score", "recommended",
        ]).head(15)
    )

    # Disagreements
    print("\nDisagreements: high NID, low SHAP")
    disagree_nid = tbl.filter(
        (pl.col("nid_rank") <= 5) & (pl.col("shap_rank") > 8)
    )
    print(disagree_nid.select(["feature_1", "feature_2", "nid_rank", "shap_rank"]))

    print("\nDisagreements: high SHAP, low NID")
    disagree_shap = tbl.filter(
        (pl.col("shap_rank") <= 5) & (pl.col("nid_rank") > 8)
    )
    print(disagree_shap.select(["feature_1", "feature_2", "nid_rank", "shap_rank"]))

    # Task 4: Planted interactions in rankings
    planted = [("age_band", "vehicle_group"), ("vehicle_group", "age_band"),
               ("ncd_years", "conviction_points"), ("conviction_points", "ncd_years")]

    for f1, f2 in [("age_band", "vehicle_group"), ("ncd_years", "conviction_points")]:
        row = tbl.filter(
            ((pl.col("feature_1") == f1) & (pl.col("feature_2") == f2)) |
            ((pl.col("feature_1") == f2) & (pl.col("feature_2") == f1))
        )
        if not row.is_empty():
            r = row.row(0, named=True)
            print(f"\n{f1} x {f2}:")
            print(f"  NID rank:  {r['nid_rank']}")
            print(f"  SHAP rank: {r['shap_rank']}")
            print(f"  Consensus: {r['consensus_score']:.4f}")
            print(f"  Recommended: {r['recommended']}")

    # Task 5: Top-3 consensus
    top3_consensus = [
        f"{r['feature_1']} x {r['feature_2']}"
        for r in tbl.head(3).iter_rows(named=True)
    ]
    print(f"\nTop-3 by consensus: {top3_consensus}")
    planted_in_top3 = sum(1 for p in top3_consensus
                          if any(plant in p for plant in
                                 ["age_band", "vehicle_group", "ncd_years", "conviction_points"]))
    print(f"Planted interactions in top-3 consensus: {planted_in_top3}/2")
```

**What you should see:**

- CatBoost should have lower deviance than the enhanced GLM, typically by 500-1,500 units. CatBoost captures all interactions automatically; the enhanced GLM captures only the two we added. This remaining gap is the theoretical limit of what further interaction detection could recover.
- Both planted interactions should appear in both the NID top-5 and the SHAP top-5.
- The consensus score should reliably put both planted interactions in the top-3. If either appears only at rank 4-5, it suggests one of the methods had a weak signal for that interaction -- which should be verified with the LR test result (`recommended == True`).

</details>

---

## Exercise 8: Severity interactions

**Reference:** Tutorial Part 1 ("Frequency vs severity" section); README section "Frequency vs severity"

**What you will do:** Apply the same pipeline to a claim severity (Gamma GLM) problem. Understand why severity interactions are noisier and how to interpret the results.

**Context.** You have claim frequencies. Now generate a synthetic claim severity dataset with one planted interaction and apply the Gamma family detector.

### Setup

```python
%md ## Exercise 8: Severity interactions
import numpy as np
import polars as pl

rng_sev = np.random.default_rng(seed=99)
```

### Tasks

**Task 1.** Generate a synthetic severity dataset. Use only the policies that had at least one claim from the original dataset (where `y >= 1`).

```python
# Claim-level dataset: one row per claim
claim_mask = y >= 1

X_claims     = X.filter(pl.Series(claim_mask))
n_claims_per = y[claim_mask].astype(int)  # may be > 1 for some policies

# Expand: if a policy has 3 claims, it appears 3 times
rows = np.repeat(np.arange(len(X_claims)), n_claims_per)
X_sev = X_claims[rows]
n_sev = len(X_sev)
```

Now generate claim amounts with a planted severity interaction: young drivers (17-21) in high vehicle groups (41-50) have mean severity 1.30x higher than multiplicative main effects alone would predict:

```python
# Base log-severity from main effects
area_sev_effect  = {"A": 0.0, "B": 0.05, "C": 0.10, "D": 0.15, "E": 0.20, "F": 0.30}
vg_sev_effect    = {"1-10": -0.10, "11-20": 0.0, "21-30": 0.10, "31-40": 0.20, "41-50": 0.30}
age_sev_effect   = {"17-21": 0.0, "22-25": -0.05, "26-34": -0.10, "35-49": -0.10, "50-69": -0.05, "70+": 0.05}

log_sev = (
    7.50  # log of ~£1,800 base
    + np.array([area_sev_effect[a] for a in X_sev["area"].to_list()])
    + np.array([vg_sev_effect[v] for v in X_sev["vehicle_group"].to_list()])
    + np.array([age_sev_effect[a] for a in X_sev["age_band"].to_list()])
    + np.where(
        (np.array(X_sev["age_band"].to_list()) == "17-21") &
        (np.array(X_sev["vehicle_group"].to_list()) == "41-50"),
        0.26,   # exp(0.26) ≈ 1.30
        0.0,
    )
)

# Gamma severity: mean = exp(log_sev), shape = 4 (coefficient of variation = 0.5)
shape  = 4.0
sev_mean = np.exp(log_sev)
claim_amounts = rng_sev.gamma(shape=shape, scale=sev_mean / shape)

print(f"Severity claims: {n_sev:,}")
print(f"Mean severity:   £{claim_amounts.mean():,.0f}")
print(f"P90 severity:    £{np.percentile(claim_amounts, 90):,.0f}")
```

**Task 2.** Fit a baseline Gamma GLM on the severity data using `glum`:

```python
from glum import GeneralizedLinearRegressor
import pandas as pd

X_sev_pd = X_sev.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_sev_pd[col] = pd.Categorical(X_sev_pd[col].astype(str))

glm_sev = GeneralizedLinearRegressor(
    family="gamma",
    alpha=0.0,
    fit_intercept=True,
)
# For Gamma GLM, sample_weight = claim counts (here all 1s since we expanded)
glm_sev.fit(X_sev_pd, claim_amounts)
mu_sev_glm = glm_sev.predict(X_sev_pd)
```

**Task 3.** Fit the `InteractionDetector` with `family="gamma"`. Use `cann_n_ensemble=3`. Note that severity datasets have much higher variance than frequency datasets (claim amounts are noisy). How does this affect the NID scores?

**Task 4.** Does the detector recover the planted severity interaction (`age_band × vehicle_group`)? What is its NID rank? What is its LR test result?

**Task 5.** Compare the NID table from the Gamma severity detector to the NID table from the Poisson frequency detector (from Exercise 2). Which interactions appear in both? Which are unique to severity? Does it make actuarial sense that the frequency and severity interaction structures would differ?

<details>
<summary>Hint for Task 3</summary>

Severity datasets are noisier than frequency datasets. Claim amounts have high variance (coefficient of variation ~0.5 in our synthetic data). The CANN may need more epochs and/or a larger ensemble to produce stable NID scores on severity.

If the planted interaction does not appear in the top 5, try `cann_n_epochs=500, cann_n_ensemble=5`. With a smaller effective dataset (only policies that had claims), the training signal is weaker.

</details>

<details>
<summary>Solution -- Exercise 8</summary>

```python
import numpy as np
import polars as pl
import pandas as pd
from glum import GeneralizedLinearRegressor
from insurance_interactions import InteractionDetector, DetectorConfig

# Task 1: Generate severity data (already done above)
print(f"Severity claims: {n_sev:,}")
print(f"Mean:  £{claim_amounts.mean():,.0f}")
print(f"P90:   £{np.percentile(claim_amounts, 90):,.0f}")
print(f"CV:    {claim_amounts.std() / claim_amounts.mean():.3f}")

# Task 2: Baseline Gamma GLM
X_sev_pd = X_sev.to_pandas()
for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
    X_sev_pd[col] = pd.Categorical(X_sev_pd[col].astype(str))

glm_sev = GeneralizedLinearRegressor(family="gamma", alpha=0.0, fit_intercept=True)
glm_sev.fit(X_sev_pd, claim_amounts)
mu_sev_glm = glm_sev.predict(X_sev_pd)

def gamma_deviance(y_true, y_pred, weights=None):
    """Gamma deviance: 2 * sum(w * (-log(y/mu) + (y - mu)/mu))"""
    if weights is None:
        weights = np.ones(len(y_true))
    y_safe = np.clip(y_true, 1e-8, None)
    mu     = np.clip(y_pred,  1e-8, None)
    return 2.0 * float(np.sum(weights * (-np.log(y_safe / mu) + (y_safe - mu) / mu)))

dev_sev_base = gamma_deviance(claim_amounts, mu_sev_glm)
print(f"\nGamma GLM baseline deviance: {dev_sev_base:,.2f}")
print(f"n_params: {len(glm_sev.coef_) + 1}")

# Task 3: Severity detector
cfg_sev = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=400,
    cann_patience=40,
    cann_n_ensemble=3,
    top_k_nid=15,
    top_k_final=5,
)
detector_sev = InteractionDetector(family="gamma", config=cfg_sev)
print("\nTraining severity interaction detector...")
detector_sev.fit(
    X=X_sev,
    y=claim_amounts,
    glm_predictions=mu_sev_glm,
    # Gamma family: no exposure offset (each row is one claim)
)
print("Severity detector training complete.")

print("\nSeverity NID table:")
print(detector_sev.nid_table().head(10))

# Task 4: Recovery of planted severity interaction
sev_table = detector_sev.interaction_table()
age_vg_sev = sev_table.filter(
    ((pl.col("feature_1") == "age_band") & (pl.col("feature_2") == "vehicle_group")) |
    ((pl.col("feature_1") == "vehicle_group") & (pl.col("feature_2") == "age_band"))
)
if not age_vg_sev.is_empty():
    r = age_vg_sev.row(0, named=True)
    print(f"\nTask 4: Planted severity interaction (age_band x vehicle_group)")
    print(f"  NID rank:     {r.get('nid_rank', 'n/a')}")
    print(f"  NID score:    {r['nid_score_normalised']:.4f}")
    if "recommended" in r:
        print(f"  Recommended:  {r['recommended']}")
    if "lr_p" in r and r["lr_p"] is not None:
        print(f"  LR p-value:   {r['lr_p']:.4e}")

# Task 5: Compare frequency vs severity NID top-10
freq_pairs = {
    f"{r['feature_1']}_{r['feature_2']}"
    for r in detector_careful.nid_table().head(10).iter_rows(named=True)
}
sev_pairs = {
    f"{r['feature_1']}_{r['feature_2']}"
    for r in detector_sev.nid_table().head(10).iter_rows(named=True)
}
in_both = freq_pairs & sev_pairs
only_freq = freq_pairs - sev_pairs
only_sev  = sev_pairs - freq_pairs

print(f"\nTask 5: Frequency vs severity NID comparison (top 10 each)")
print(f"  In both top-10:   {sorted(in_both)}")
print(f"  Frequency only:   {sorted(only_freq)}")
print(f"  Severity only:    {sorted(only_sev)}")
```

**What you should see:**

- The severity dataset is smaller than the frequency dataset (only claim-level rows), which reduces the training signal for the CANN.
- The planted severity interaction (`age_band × vehicle_group`) should appear in the NID top-5, but its NID score normalised will typically be lower than the corresponding frequency interaction because claim amounts are noisy.
- Frequency and severity interactions can legitimately differ. Young drivers × high vehicle groups have supermultiplicative frequency (more accidents) AND supermultiplicative severity (more serious accidents). Area × vehicle group may affect frequency (urban accident rates) without affecting severity. `ncd_years × conviction_points` is planted in frequency but not severity -- so it should appear in the frequency NID top-5 but not the severity one.

</details>

---

## Exercise 9: End-to-end pipeline with MLflow logging

**Reference:** Tutorial Parts 15, 16, 17

**What you will do:** Build a complete, production-style interaction detection pipeline that logs everything to MLflow and writes results to Delta. This is what the process looks like in practice.

### Setup

```python
import mlflow
import mlflow.sklearn
import polars as pl
import numpy as np

EXPERIMENT_NAME = "module_10_ex9_pipeline"
mlflow.set_experiment(EXPERIMENT_NAME)

%md ## Exercise 9: End-to-end production pipeline
```

### Tasks

**Task 1.** Build a function `run_interaction_pipeline()` that takes the following arguments and runs the complete pipeline:

- `X`: Polars DataFrame of rating factors
- `y`: claim counts
- `exposure`: exposure array
- `top_k_nid`: int (how many NID candidates to GLM-test)
- `top_k_suggest`: int (how many to suggest from significant results)
- `experiment_name`: MLflow experiment name
- `run_name`: MLflow run name

The function should:
1. Fit the baseline Poisson GLM
2. Train the `InteractionDetector` with `cann_n_ensemble=3`
3. Build the enhanced GLM with suggested interactions
4. Log all metrics, parameters, and the interaction table to MLflow
5. Return `(enhanced_glm, suggested_pairs, comparison_df, interaction_table_df)`

**Task 2.** Run your pipeline function twice:
- Run A: `top_k_nid=10, top_k_suggest=2, run_name="conservative"`
- Run B: `top_k_nid=20, top_k_suggest=5, run_name="liberal"`

**Task 3.** In the MLflow UI (Databricks left sidebar, Experiments), compare the two runs. Which has better AIC? Which has better BIC? Which would you recommend and why?

**Task 4.** Write the interaction table from the liberal run to a Delta table called `training.module10.interaction_results`:

```python
spark.createDataFrame(
    liberal_table.to_pandas()
).write.mode("overwrite").saveAsTable("training.module10.interaction_results")
```

Then read it back with `spark.table("training.module10.interaction_results").show(5)` to confirm the write succeeded.

**Task 5.** Write a helper function `score_new_policies(new_X, enhanced_glm, interaction_pairs)` that takes a new Polars DataFrame (representing policies not seen during training) and returns their predicted claim frequencies from the enhanced GLM. Handle the case where a new policy has an unseen level for an interaction cell (use the string concatenation approach from Exercise 6 that does not create `pd.Categorical` on the interaction column).

<details>
<summary>Hint for Task 1</summary>

Structure your function around the objects you have been building in Exercises 1-5. The MLflow logging follows exactly the pattern from Part 15 of the tutorial. Log:
- `deviance`, `aic`, `bic`, `n_params` for both base and enhanced GLMs
- `n_interactions_suggested`, `n_interactions_recommended`
- The interaction table as a CSV artifact

</details>

<details>
<summary>Solution -- Exercise 9</summary>

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
from glum import GeneralizedLinearRegressor
from insurance_interactions import InteractionDetector, DetectorConfig, build_glm_with_interactions


def poisson_deviance(y_true, y_pred, weights):
    mu = np.clip(y_pred, 1e-8, None)
    log_term = np.where(y_true > 0, y_true * np.log(y_true / mu), 0.0)
    return 2.0 * float(np.sum(weights * (log_term - (y_true - mu))))


def run_interaction_pipeline(
    X: pl.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray,
    top_k_nid: int = 15,
    top_k_suggest: int = 3,
    experiment_name: str = "interaction_detection",
    run_name: str = "default",
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:

        # Step 1: Baseline GLM
        X_pd = X.to_pandas()
        for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
            X_pd[col] = pd.Categorical(X_pd[col].astype(str))

        glm_base = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True)
        glm_base.fit(X_pd, y, sample_weight=exposure)
        mu_glm   = glm_base.predict(X_pd)
        dev_base = poisson_deviance(y, mu_glm, exposure)
        n_params_base = len(glm_base.coef_) + 1
        n = len(X)
        aic_base = dev_base + 2 * n_params_base
        bic_base = dev_base + np.log(n) * n_params_base

        mlflow.log_param("top_k_nid", top_k_nid)
        mlflow.log_param("top_k_suggest", top_k_suggest)
        mlflow.log_param("family", "poisson")
        mlflow.log_metric("base_deviance", dev_base)
        mlflow.log_metric("base_aic",      aic_base)
        mlflow.log_metric("base_bic",      bic_base)
        mlflow.log_metric("base_n_params", n_params_base)

        # Step 2: InteractionDetector
        cfg = DetectorConfig(
            cann_hidden_dims=[32, 16],
            cann_n_epochs=300,
            cann_patience=30,
            cann_n_ensemble=3,
            top_k_nid=top_k_nid,
            top_k_final=top_k_suggest,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(X=X, y=y, glm_predictions=mu_glm, exposure=exposure)

        ix_table   = detector.interaction_table()
        suggested  = detector.suggest_interactions(top_k=top_k_suggest)
        n_rec = ix_table.filter(pl.col("recommended") == True).height if "recommended" in ix_table.columns else 0

        mlflow.log_metric("n_nid_tested",       ix_table.height)
        mlflow.log_metric("n_recommended",      n_rec)
        mlflow.log_metric("n_suggested",        len(suggested))
        mlflow.log_param("suggested_pairs",     str(suggested))

        # Step 3: Enhanced GLM
        enhanced_glm, comparison = build_glm_with_interactions(
            X=X, y=y, exposure=exposure,
            interaction_pairs=suggested,
            family="poisson",
        )
        dev_enh    = comparison.filter(pl.col("model") == "glm_with_interactions")["deviance"][0]
        n_params_enh = comparison.filter(pl.col("model") == "glm_with_interactions")["n_params"][0]
        aic_enh    = dev_enh + 2 * n_params_enh
        bic_enh    = dev_enh + np.log(n) * n_params_enh

        mlflow.log_metric("enhanced_deviance",    dev_enh)
        mlflow.log_metric("enhanced_aic",         aic_enh)
        mlflow.log_metric("enhanced_bic",         bic_enh)
        mlflow.log_metric("enhanced_n_params",    n_params_enh)
        mlflow.log_metric("delta_deviance",       dev_base - dev_enh)
        mlflow.log_metric("delta_deviance_pct",   100 * (dev_base - dev_enh) / dev_base)
        mlflow.log_metric("delta_aic",            aic_enh - aic_base)
        mlflow.log_metric("delta_bic",            bic_enh - bic_base)

        # Log interaction table as CSV artifact
        table_path = f"/tmp/interaction_table_{run_name}.csv"
        ix_table.to_pandas().to_csv(table_path, index=False)
        mlflow.log_artifact(table_path, "interaction_detection")

        mlflow.sklearn.log_model(glm_base,    "base_glm")
        mlflow.sklearn.log_model(enhanced_glm, "enhanced_glm")

        run_id = run.info.run_id
        print(f"Run '{run_name}' complete. Run ID: {run_id}")
        print(f"  Base GLM:     deviance={dev_base:,.1f}, AIC={aic_base:,.1f}, BIC={bic_base:,.1f}")
        print(f"  Enhanced GLM: deviance={dev_enh:,.1f},  AIC={aic_enh:,.1f},  BIC={bic_enh:,.1f}")
        print(f"  Suggested:    {suggested}")

    return enhanced_glm, suggested, comparison, ix_table


# Task 2: Run conservative and liberal pipelines
print("=== Conservative run ===")
enh_con, sug_con, comp_con, tbl_con = run_interaction_pipeline(
    X=X, y=y, exposure=exposure_arr,
    top_k_nid=10, top_k_suggest=2,
    experiment_name=EXPERIMENT_NAME,
    run_name="conservative",
)

print("\n=== Liberal run ===")
enh_lib, sug_lib, comp_lib, tbl_lib = run_interaction_pipeline(
    X=X, y=y, exposure=exposure_arr,
    top_k_nid=20, top_k_suggest=5,
    experiment_name=EXPERIMENT_NAME,
    run_name="liberal",
)

# Task 3: Compare in Python (also check MLflow UI)
print("\nTask 3: Conservative vs liberal comparison")
for label, comp in [("Conservative", comp_con), ("Liberal", comp_lib)]:
    row = comp.filter(pl.col("model") == "glm_with_interactions")
    print(f"  {label}: AIC={row['aic'][0]:,.1f}, BIC={row['bic'][0]:,.1f}, "
          f"n_params={row['n_params'][0]}")

# Task 4: Write to Delta
spark.createDataFrame(
    tbl_lib.to_pandas()
).write.mode("overwrite").saveAsTable("training.module10.interaction_results")
print("\nTask 4: Written to training.module10.interaction_results")
spark.table("training.module10.interaction_results").show(5)

# Task 5: Scoring function
def score_new_policies(
    new_X: pl.DataFrame,
    enhanced_glm,
    interaction_pairs: list,
) -> np.ndarray:
    """Score new policies with the interaction-enhanced GLM.

    Handles unseen interaction levels gracefully by using string
    concatenation (not pd.Categorical) for the interaction columns.
    """
    new_X_pd = new_X.to_pandas()
    for col in ["area", "vehicle_group", "age_band", "annual_mileage"]:
        if col in new_X_pd.columns:
            new_X_pd[col] = pd.Categorical(new_X_pd[col].astype(str))

    for f1, f2 in interaction_pairs:
        col_name = f"_ix_{f1}_{f2}"
        # String (not Categorical) -- glum treats unseen string levels as baseline
        new_X_pd[col_name] = (
            new_X_pd[f1].astype(str) + "_X_" + new_X_pd[f2].astype(str)
        )

    return np.clip(enhanced_glm.predict(new_X_pd), 1e-8, None)

# Test scoring on a small sample
sample = X[:10]
mu_sample = score_new_policies(sample, enh_lib, sug_lib)
print(f"\nTask 5: Sample predictions for first 10 policies:")
print(np.round(mu_sample, 5))
```

**What you should see:**

- The conservative run (top_k_nid=10, top_k_suggest=2) adds fewer interactions. Its AIC and BIC may be slightly higher than the liberal run if the additional interactions in the liberal run genuinely improve fit.
- The liberal run (top_k_nid=20, top_k_suggest=5) tests more candidates and adds more interactions. If all 5 suggested interactions are statistically significant (Bonferroni-corrected), both AIC and BIC should improve relative to the conservative run.
- In practice, we recommend the conservative approach for a first production run. It is easier to explain to a pricing committee why you added 2 interactions (clear evidence for each) than why you added 5.

</details>

---

## Exercise 10: Presenting interaction findings to a pricing committee

**Reference:** Tutorial Part 17

**What you will do:** Prepare a structured presentation of your interaction detection results, anticipate the questions a pricing committee will ask, and write the model governance documentation.

**Context.** You have run the full pipeline. The pricing committee meets on Thursday. They have approved the use of GBM-based methods for interaction identification but have not approved adding any interactions to the production GLM yet. Your job is to present the evidence clearly enough that they can make an informed decision.

This exercise is mostly written answers and chart production, not code. It tests whether you understand the methodology well enough to explain it.

### Tasks

**Task 1: The two-slide summary.**

Produce a two-panel chart suitable for a slide presentation. Panel A should be a horizontal bar chart of the top 10 NID scores (normalised) from `detector_careful`, colour-coded by whether the pair is `recommended == True` (blue) or not (grey). Panel B should be a scatter plot with `delta_deviance_pct` on the x-axis and `n_cells` on the y-axis, with recommended pairs labelled by name and non-recommended pairs as grey dots.

```python
import matplotlib.pyplot as plt
import polars as pl

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Interaction Detection Results -- UK Motor Pricing Review", fontsize=14, fontweight="bold")

# Panel A: NID scores
tbl = detector_careful.interaction_table()
top10 = tbl.head(10)
labels = [
    f"{r['feature_1']} x {r['feature_2']}"
    for r in top10.iter_rows(named=True)
]
scores = top10["nid_score_normalised"].to_list()
colours = [
    "#2271b3" if r.get("recommended", False) else "#aaaaaa"
    for r in top10.iter_rows(named=True)
]
ax1.barh(range(len(labels)), scores, color=colours)
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("NID score (normalised)")
ax1.set_title("Panel A: NID interaction ranking")
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color="#2271b3", label="Recommended (LR significant)"),
    Patch(color="#aaaaaa", label="Not recommended"),
], loc="lower right", fontsize=8)

# Your Panel B code here
```

**Task 2: The committee questions.**

Write answers to the following five questions that a UK pricing actuary on the committee might ask. Your answers should be concise (2-4 sentences each) and use the specific numbers from your `detector_careful` results.

1. "How do we know these interactions are real and not artefacts of the neural network training?"
2. "You are adding 20-odd parameters to the GLM. How do we know the data supports this? What if some cells have only 5 policies?"
3. "The FCA would ask us to demonstrate these interactions are not a proxy for a protected characteristic. How would you respond?"
4. "If we add these interactions and something goes wrong at the next audit, who is accountable for the decision?"
5. "The GBM has 30% lower deviance than our current GLM. Why are we only capturing 1.5% improvement through interaction detection? Where is the rest of the gap?"

**Task 3: The governance document.**

Write a markdown cell in your notebook that serves as the formal model change documentation for the interaction additions. A strong committee memo must address all of the following:

- **Statistical test result**: the LR test statistic, p-value (Bonferroni-corrected), and whether the result is robust to the choice of CANN architecture
- **Parameter cost**: the number of new GLM parameters (n_cells) and why the data credibility supports estimating them
- **AIC/BIC change**: confirm the model selection criteria improve; explain why a significant LR test can coexist with a deteriorating AIC
- **Underwriting plausibility**: why the interaction makes causal sense (or if it does not, why you are adding it anyway and what monitoring is planned)
- **Data credibility of affected cells**: minimum exposure in the cells, cells with fewer than 50 claims that will be shrunk or excluded
- **Monitoring plan**: what will be measured in the first 12 months to confirm the interaction is stable and not a spurious fit

Include:
- The model change description (what factors are being added and why)
- The evidence basis (NID scores, LR test results, deviance improvement)
- The limitations and known risks
- The testing performed (in-sample and out-of-sample)
- Sign-off requirements

**Task 4.** One of the committee members raises a concern: "Young drivers in high vehicle groups are already priced for separately -- we have age factors and vehicle group factors. Isn't the interaction just double-counting?"

Write a two-paragraph explanation (as a markdown cell) that explains why the interaction is NOT double-counting, using the multiplicative GLM formula. Use the specific numbers from your results to show how much the current GLM is underpricing the specific segment.

**Task 5.** The committee approves adding the two statistically significant interactions. Write the production scoring function as a clean Python module (not a notebook cell) with:
- A docstring explaining what the function does and its governance reference
- Input validation (check that all required columns are present)
- The interaction column construction
- The prediction call
- A brief test at the bottom using the first 100 rows of your synthetic data

<details>
<summary>Hint for Task 2, Question 5</summary>

The gap between GBM and interaction GLM can be decomposed into:
1. Interactions not in your shortlist (the NID ranked 66 pairs; you tested 15 and added 2)
2. Non-linear main effects that the GLM approximates with bands (the GBM uses the raw continuous variable and fits arbitrary non-linear functions)
3. Three-way and higher-order interactions (GBMs capture these; the pipeline tested only pairwise)
4. Structural non-multiplicativity (some risk factors may combine in ways that are not multiplicative even after adding pairwise terms)

The 1.5% deviance improvement from two interactions is not the ceiling -- it is the floor. Running the pipeline with `top_k_suggest=5` and adding more interactions would close more of the gap.

</details>

<details>
<summary>Solution -- Exercise 10</summary>

```python
# Task 1: Two-panel committee chart
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import polars as pl

tbl = detector_careful.interaction_table()
top10 = tbl.head(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Interaction Detection Results -- UK Motor Frequency Model\nPricing Committee, March 2026",
    fontsize=13, fontweight="bold"
)

# Panel A: NID bar chart
labels  = [f"{r['feature_1']} x {r['feature_2']}" for r in top10.iter_rows(named=True)]
scores  = top10["nid_score_normalised"].to_list()
colours = [
    "#2271b3" if r.get("recommended", False) else "#aaaaaa"
    for r in top10.iter_rows(named=True)
]
ax1.barh(range(len(labels)), scores, color=colours, edgecolor="white")
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("NID score (normalised)", fontsize=10)
ax1.set_title("Panel A: CANN-NID interaction ranking", fontsize=11)
ax1.legend(handles=[
    Patch(color="#2271b3", label="LR-significant (Bonferroni p < 0.003)"),
    Patch(color="#aaaaaa", label="Not significant"),
], fontsize=8, loc="lower right")

# Panel B: Deviance gain vs parameter cost
if "delta_deviance_pct" in tbl.columns and "n_cells" in tbl.columns:
    all_pairs = tbl.filter(pl.col("delta_deviance_pct").is_not_null())

    rec   = all_pairs.filter(pl.col("recommended") == True)
    norec = all_pairs.filter(pl.col("recommended") == False)

    ax2.scatter(
        norec["delta_deviance_pct"].to_list(),
        norec["n_cells"].to_list(),
        color="#aaaaaa", s=50, alpha=0.7, zorder=1, label="Not recommended"
    )
    ax2.scatter(
        rec["delta_deviance_pct"].to_list(),
        rec["n_cells"].to_list(),
        color="#2271b3", s=100, zorder=2, label="Recommended"
    )
    for r in rec.iter_rows(named=True):
        ax2.annotate(
            f"{r['feature_1']} x\n{r['feature_2']}",
            xy=(r["delta_deviance_pct"], r["n_cells"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="#2271b3",
        )

ax2.set_xlabel("Deviance improvement (%)", fontsize=10)
ax2.set_ylabel("Parameter cost (n_cells)", fontsize=10)
ax2.set_title("Panel B: Deviance gain vs parameter cost", fontsize=11)
ax2.legend(fontsize=8)
ax2.axhline(y=20, color="red", linestyle="--", alpha=0.4, linewidth=1,
            label="Credibility threshold (indicative)")

plt.tight_layout()
plt.show()
```

**Task 2 sample answers:**

**Q1: "How do we know these are real?"**
The CANN-NID pipeline identifies candidates, but the definitive test is the likelihood-ratio test against the GLM. Both planted interactions show chi-squared statistics in the hundreds-to-thousands range with Bonferroni-corrected p-values below 0.001. A false positive at that significance level, after correcting for 15 simultaneous tests, is extremely unlikely. The out-of-sample deviance improvement (Task 4, Exercise 6) confirms the interactions generalise to held-out data.

**Q2: "Data support for 20 parameters?"**
The `n_cells` column in the interaction table shows the parameter cost. The `age_band × vehicle_group` interaction has `n_cells = 20` (5 age levels minus 1, times 4 vehicle group levels minus 1). With 100,000 policies and roughly 20,000 young-driver claims, each interaction cell has adequate data on average -- but the sparsest cells (17-21, 41-50) may have only 200-400 policies. We would apply partial-pooling regularisation (covered in the Bayesian pricing module) for those cells. For now, the GLM estimates are regularisable via `alpha > 0` in glum.

**Q3: "FCA proxy check?"**
Age is not a protected characteristic under the Equality Act 2010 for motor insurance pricing. Vehicle group is not a protected characteristic at all. A proxy concern would arise if an interaction term acted as a surrogate for, say, ethnicity or disability -- which would require investigating whether the interaction's predictive power is concentrated in areas or demographic groups that correlate with protected characteristics. We would run this check by computing the A/E ratios by postcode sector for the interaction cells and flagging any patterns that correlate with ONS demographic data.

**Q4: "Who is accountable?"**
Under PRA SS1/23 model risk governance, the model owner (the Chief Pricing Actuary) is accountable for the model change. This pipeline produces a clear, reproducible audit trail: the MLflow experiment, the interaction table with LR test statistics, the deviance comparison, and the out-of-sample validation. The actuary reviewed and approved the shortlist; the pipeline provided the evidence but did not make the decision. That distinction -- human decision, automated evidence -- is exactly the SS1/23-compliant workflow.

**Q5: "Why only 1.5% improvement?"**
The GBM's advantage comes from three sources: non-linear main effects (the GLM uses bands; the GBM uses the full continuous variable), pairwise interactions beyond the two we added, and three-way-and-higher interactions. The two pairwise interactions capture the largest signal; the rest is distributed across many smaller effects. To close more of the gap, we would need to: (1) add more interactions from the NID shortlist, (2) refine the banding on continuous variables, and (3) eventually consider a CANN-based production model rather than a GLM. The 1.5% improvement at 24 new parameters has better parameter efficiency than the GBM's 1,000+ split points.

```python
# Task 5: Production scoring function
# Write this to a file for cleanliness

production_code = '''"""
interaction_scorer.py
---------------------
Production scoring for the UK Motor frequency GLM with detected interactions.

Governance reference: Pricing Committee approval, March 2026.
MLflow experiment: module_10_ex9_pipeline, run "conservative".
Model risk documentation: pricing/models/motor/frequency/v3.2/model_doc.pdf

This function is used by the nightly batch scoring pipeline in Databricks.
Do not modify without a model change request approved by the Chief Pricing Actuary.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Optional


REQUIRED_COLUMNS = [
    "area", "vehicle_group", "ncd_years", "age_band",
    "conviction_points", "annual_mileage",
]

CATEGORICAL_COLUMNS = ["area", "vehicle_group", "age_band", "annual_mileage"]


def score_policies(
    X: pl.DataFrame,
    enhanced_glm,
    interaction_pairs: list,
    exposure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score policies with the interaction-enhanced Poisson GLM.

    Parameters
    ----------
    X:
        Rating factors as Polars DataFrame. Must contain all columns in
        REQUIRED_COLUMNS.
    enhanced_glm:
        Fitted glum GeneralizedLinearRegressor from build_glm_with_interactions().
    interaction_pairs:
        List of (feature_1, feature_2) tuples. Must match the pairs used
        when fitting enhanced_glm.
    exposure:
        Optional exposure array. If provided, returns expected claim counts
        (frequency x exposure). If None, returns claim frequency.

    Returns
    -------
    np.ndarray of predicted claim frequencies (or counts if exposure provided).

    Raises
    ------
    ValueError:
        If required columns are missing from X.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_pd = X.to_pandas()
    for col in CATEGORICAL_COLUMNS:
        if col in X_pd.columns:
            X_pd[col] = pd.Categorical(X_pd[col].astype(str))

    for f1, f2 in interaction_pairs:
        col_name = f"_ix_{f1}_{f2}"
        X_pd[col_name] = (
            X_pd[f1].astype(str) + "_X_" + X_pd[f2].astype(str)
        )

    mu = np.clip(enhanced_glm.predict(X_pd), 1e-8, None)

    if exposure is not None:
        return mu * np.asarray(exposure, dtype=np.float64)
    return mu


if __name__ == "__main__":
    # Brief smoke test -- runs when file is executed directly
    import sys
    print("Smoke test: scoring first 100 rows of synthetic data")
    # (In practice, load from Unity Catalog)
    # This block is left as a placeholder for CI integration
    print("OK")
'''

with open("/tmp/interaction_scorer.py", "w") as f:
    f.write(production_code)

print("Production scoring module written to /tmp/interaction_scorer.py")

# Test it on our synthetic data
import importlib.util
spec = importlib.util.spec_from_file_location("interaction_scorer", "/tmp/interaction_scorer.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sample_100 = X[:100]
mu_sample  = mod.score_policies(sample_100, enh_con, sug_con)
print(f"Sample prediction range: [{mu_sample.min():.4f}, {mu_sample.max():.4f}]")
print(f"Mean predicted frequency: {mu_sample.mean():.4f}")
print("Scoring function: OK")
```

**Task 3: Governance document template (paste as markdown cell):**

```python
## Model Change Documentation

**Change reference:** PCM-2026-03-001
**Date:** March 2026
**Model:** UK Motor Frequency GLM v3.2
**Author:** [Your name], pricing actuary

### Change description
Addition of two pairwise interaction terms to the Poisson frequency GLM:
1. `age_band × vehicle_group` (20 new parameters)
2. `ncd_years × conviction_points` (4 new parameters)

### Evidence basis
Detected via CANN-NID pipeline (insurance-interactions v0.1.0) on 100,000-policy synthetic motor portfolio.

| Interaction | NID rank | delta_deviance | delta_deviance_pct | LR chi2 | Bonferroni p | Recommended |
|---|---|---|---|---|---|---|
| age_band × vehicle_group | 1 | see output | ~1.2% | high | <0.001 | True |
| ncd_years × conviction_points | 2 | see output | ~0.3% | high | <0.001 | True |

Joint deviance improvement: ~1.5%. Joint AIC improvement: confirmed. BIC improvement: confirmed.

### Limitations and risks
- Sparse cells (< 50 policies) in the age × vehicle group interaction may have high parameter uncertainty. Monitor A/E ratios for these cells at next quarterly review.
- The NID pipeline is sensitive to CANN training stability. Results were averaged over 3 ensemble runs with early stopping. Re-running the pipeline may produce small changes in NID scores.
- No regulatory proxy check has been completed. This must be performed before production deployment.

### Testing performed
- In-sample deviance improvement confirmed.
- Out-of-sample (20% hold-out) deviance improvement confirmed: ~1.2%.
- MLflow experiment logged: module_10_ex9_pipeline, run "conservative".

### Sign-off required
- Chief Pricing Actuary (model owner)
- Model Risk function (independent review per SS1/23)
- Compliance (FCA proxy check sign-off)
```

</details>

---

## Reference summary

The table below maps each exercise to the tutorial sections and library components it uses.

| Exercise | Tutorial parts | Key classes and functions |
|---|---|---|
| 1 | 1, 5, 7 | Polars `group_by`, manual A/E analysis |
| 2 | 8, 3 | `DetectorConfig`, `InteractionDetector`, `val_deviance_history` |
| 3 | 2, 3 | `DetectorConfig(mlp_m=True)`, `glm_test_table()` |
| 4 | 10 | `glum.GeneralizedLinearRegressor`, `scipy.stats.chi2` |
| 5 | 12, 13 | `build_glm_with_interactions()`, `suggest_interactions()` |
| 6 | 17 | Train/test split, OOS deviance, lift charts |
| 7 | 14 | `CatBoostRegressor`, `shapiq`, `shap_model=` parameter |
| 8 | Parts 1, 14 | `InteractionDetector(family="gamma")`, Gamma deviance |
| 9 | 15, 16, 17 | `mlflow`, Delta tables, `spark.createDataFrame()` |
| 10 | 17 | Charts, committee communication, governance documentation |

All code in these exercises uses:
- `polars` for data manipulation (not pandas, except at the glum/CatBoost boundary)
- `glum` for GLM fitting
- `insurance-interactions` for the CANN-NID pipeline
- `mlflow` for experiment tracking
- `matplotlib` for charts
