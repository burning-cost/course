## Part 8: Training the InteractionDetector

Now we train the CANN. The `InteractionDetector` class orchestrates the full three-stage pipeline.

```python
from insurance_interactions import InteractionDetector, DetectorConfig

cfg = DetectorConfig(
    cann_hidden_dims=[32, 16],     # Two hidden layers: 32 then 16 units
    cann_n_epochs=300,             # Maximum epochs (early stopping will kick in sooner)
    cann_patience=30,              # Stop if val deviance does not improve for 30 epochs
    cann_n_ensemble=3,             # Train 3 runs and average NID scores
    top_k_nid=15,                  # Test top 15 NID pairs with LR tests
    top_k_final=5,                 # Suggest top 5 confirmed interactions
    mlp_m=False,                   # Standard CANN (not MLP-M variant)
    alpha_bonferroni=0.05,         # Significance level after Bonferroni correction
)

detector = InteractionDetector(family="poisson", config=cfg)

print("Training CANN ensemble (3 runs × up to 300 epochs each)...")
print("Expected time: 2-5 minutes on a single-node Databricks cluster.")
print()

detector.fit(
    X=X,
    y=y,
    glm_predictions=mu_glm,
    exposure=exposure_arr,
)

print("Training complete.")
```

**What is happening during training:**

The CANN trains three times with different random seeds (seeds 42, 43, 44). Each run trains on 80% of the data and validates on the other 20%, using early stopping to avoid overfitting. After training, the three sets of weight matrices are averaged in the NID scoring step, which smooths out noise from the stochastic training process.

Training 300 epochs of a small neural network on 100,000 rows takes a few minutes. You will not see progress output during training — the cell will appear to hang. That is expected. Do not interrupt it.

### Check the training history

```python
# Inspect validation deviance curves
val_histories = detector.cann.val_deviance_history

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, history in enumerate(val_histories):
    axes[i].plot(history)
    axes[i].set_title(f"Ensemble run {i+1}")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Validation deviance")
    axes[i].axvline(x=np.argmin(history), color="red", linestyle="--", alpha=0.5, label=f"Best: epoch {np.argmin(history)}")
    axes[i].legend()

plt.tight_layout()
plt.show()

# Print where each run stopped
for i, history in enumerate(val_histories):
    best_epoch = int(np.argmin(history))
    best_val   = min(history)
    print(f"Run {i+1}: best epoch {best_epoch}, val deviance {best_val:.4f}")
```

**What to look for:** Each run's validation deviance should decrease steadily then plateau. The red dashed line marks the best epoch, after which early stopping patience began. If the best epoch is near epoch 1 or 2, the CANN is not learning — which could mean your GLM predictions `mu_glm` are missing or incorrectly scaled (they should be on the response scale, not log scale).

If the deviance never improves from epoch 0, check that `glm_predictions` is `np.ndarray` of positive values summing approximately to `y.sum()`.