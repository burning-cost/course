## Part 14: Comparing the GBM to the GLM

Now we answer the original question: does the GBM beat the GLM, and where?

### Fit the benchmark GLM

We need a GLM fitted on the same training data for a fair comparison. If you ran Module 2, you have one already. If not, fit a quick one here:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

df_glm_train = df_train_final.to_pandas()
df_glm_test  = df_test_final.to_pandas()

df_glm_train["log_exposure"] = np.log(df_glm_train["exposure_years"].clip(lower=1e-6))
df_glm_test["log_exposure"]  = np.log(df_glm_test["exposure_years"].clip(lower=1e-6))

# Convert conviction_points to string so statsmodels treats it as categorical
df_glm_train["conviction_points"] = df_glm_train["conviction_points"].astype(str)
df_glm_test["conviction_points"]  = df_glm_test["conviction_points"].astype(str)

glm_formula = (
    "claim_count ~ C(area) + ncd_years + vehicle_group + "
    "driver_age + C(conviction_points)"
)

glm_model = smf.glm(
    formula=glm_formula,
    data=df_glm_train,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_glm_train["log_exposure"],
).fit()

glm_pred_counts = glm_model.predict(df_glm_test, offset=df_glm_test["log_exposure"])
print(f"GLM fitted. Deviance: {glm_model.deviance:.1f}")
```

**Why `statsmodels` is imported here rather than at the top:** We could have imported it at the start with everything else. In this tutorial we separate it to make clear that the GLM comparison is a separate step from the CatBoost modelling. In production notebooks, put all imports at the top.

### Gini coefficient

The Gini coefficient measures discrimination: how well the model separates high-risk from low-risk policies. We use the binary AUC formulation - binary because most policies have zero claims, so the discrimination problem is effectively a binary one (will this policy claim or not?):

```python
def gini_coefficient(y_counts: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Binary discrimination Gini, based on ROC-AUC of claim/no-claim.
    Formula: Gini = 2 * AUC - 1.
    A model that discriminates perfectly scores 1.0; random guessing scores 0.0.
    """
    y_binary = (y_counts > 0).astype(int)
    auc = roc_auc_score(y_binary, y_scores)
    return float(2 * auc - 1)

# Convert count predictions to frequency predictions for comparison
gbm_freq_pred = y_pred_freq / w_test_f
glm_freq_pred = glm_pred_counts.values / w_test_f

gini_gbm = gini_coefficient(y_test_f, gbm_freq_pred)
gini_glm = gini_coefficient(y_test_f, glm_freq_pred)

print(f"Gini - GBM: {gini_gbm:.3f}")
print(f"Gini - GLM: {gini_glm:.3f}")
print(f"Lift:       {gini_gbm - gini_glm:+.3f}")
```

**Interpreting the lift:** On UK motor data, a Gini lift of 0.03-0.05 is meaningful and reproducible. A lift of 0.01-0.02 is probably within the noise of the test set. A lift of 0.06+ on a well-developed GLM is unusual - if you see this, check whether the GLM formula is missing something obvious or whether the test set has a structural difference from the training data.

Now log the Gini comparison back to the frequency model's MLflow run, so the comparison sits alongside the model:

```python
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_metric("gini_gbm",  gini_gbm)
    mlflow.log_metric("gini_glm",  gini_glm)
    mlflow.log_metric("gini_lift", gini_gbm - gini_glm)
```

### Double lift chart

The Gini tells you how much the GBM outperforms the GLM overall. The double lift chart tells you *where* the GBM disagrees with the GLM and whether those disagreements correspond to genuine risk.

We bin the test set by decile of the ratio GBM predicted frequency / GLM predicted frequency. Within each decile, we compute the actual observed frequency. If the GBM is finding real additional signal, the deciles where the GBM predicts more than the GLM (high ratio) should show higher actual frequencies than the deciles where the GBM predicts less (low ratio). The chart should slope upward from left to right.

Create a new cell:

```python
ratio       = gbm_freq_pred / (glm_freq_pred + 1e-10)
actual_freq = y_test_f / w_test_f

n_bins    = 10
bin_edges = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
bin_idx   = np.digitize(ratio, bin_edges[1:-1])

ratio_means, actual_means, n_obs = [], [], []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    ratio_means.append(ratio[mask].mean())
    actual_means.append(actual_freq[mask].mean())
    n_obs.append(int(mask.sum()))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ratio_means, actual_means, "o-", color="steelblue", linewidth=2, label="Actual frequency")
ax.axhline(
    actual_freq.mean(),
    linestyle="--",
    color="grey",
    label=f"Portfolio mean ({actual_freq.mean():.4f})",
)
ax.set_xlabel("GBM / GLM predicted frequency ratio (by decile)")
ax.set_ylabel("Actual observed frequency")
ax.set_title("Double Lift Chart: CatBoost vs GLM - Motor Frequency")
ax.legend()
plt.tight_layout()
plt.show()
```

Run this. The chart appears in the cell output.

**Reading the chart:**
- A positively sloping line (actual frequency rises from left to right) confirms the GBM is finding real additional risk signal. The deciles where the GBM predicts more than the GLM actually are higher risk.
- A flat line means the GBM's additional complexity is not translating to better risk identification. The GLM is capturing the same information.
- A negatively sloping line means the GBM is finding spurious patterns. Do not deploy a model where this chart slopes downward.

On our synthetic data, the chart should slope upward because the DGP contains a superadditive interaction between young driver and high vehicle group that the multiplicative GLM underestimates. The GBM tree structure captures this interaction directly.

**What is happening in the top decile?** Look at the policies where the GBM predicts much higher frequency than the GLM. Run this:

```python
top_mask = bin_idx == (n_bins - 1)
print(f"Top decile: {top_mask.sum():,} policies")
print(f"Actual frequency in top decile:   {actual_freq[top_mask].mean():.4f}")
print(f"Portfolio mean actual frequency:  {actual_freq.mean():.4f}")
print(f"Ratio:                            {actual_freq[top_mask].mean() / actual_freq.mean():.2f}x portfolio")
print()

df_top = df_test_final.to_pandas().loc[top_mask]
df_all = df_test_final.to_pandas()
print(f"Mean driver age  - top decile: {df_top['driver_age'].mean():.1f}, portfolio: {df_all['driver_age'].mean():.1f}")
print(f"Mean vehicle grp - top decile: {df_top['vehicle_group'].mean():.1f}, portfolio: {df_all['vehicle_group'].mean():.1f}")
print(f"Pct age < 25     - top decile: {(df_top['driver_age'] < 25).mean():.1%}, portfolio: {(df_all['driver_age'] < 25).mean():.1%}")
```

You will find that the top decile is concentrated in young drivers in high vehicle groups - the interaction the DGP contains. This is the interaction the GLM misses and the GBM finds.

Save the chart as an MLflow artefact:

```python
fig.savefig("/tmp/double_lift.png", dpi=120, bbox_inches="tight")
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_artifact("/tmp/double_lift.png", artifact_path="charts")
print("Double lift chart saved to MLflow.")
```