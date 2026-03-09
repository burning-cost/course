## Part 2: The CANN architecture

The core idea of the CANN is to give a neural network something specific to learn: the part of the data that the GLM cannot explain.

### The skip connection

The CANN (Combined Actuarial Neural Network, from Schelldorfer and Wüthrich, 2019) uses a skip connection. The GLM prediction is injected directly into the network's output in log space:

```sql
log(μ_CANN) = NN(x; θ) + log(μ_GLM)
```

Or equivalently:

```sql
μ_CANN = μ_GLM × exp(NN(x; θ))
```

At the start of training, the neural network's output layer is zero-initialised. This means at epoch 0, `NN(x; θ) = 0` for every observation, so `μ_CANN = μ_GLM` exactly. The network starts from the GLM prediction and only moves away from it during training.

Why does this matter? Because we want the network to learn only what the GLM is missing. If we trained a plain neural network from scratch, it would learn both the main effects (which the GLM already captures well) and any interactions (which it does not). The main effects would dominate the training signal, and the interaction signal would be small by comparison. The skip connection removes the main effect signal entirely: since the GLM already predicts those accurately, there is no residual to learn there. Only the interactions — where the GLM systematically under- or over-predicts — provide gradient signal.

### What the network is learning

After training, `NN(x; θ)` is a small additive correction to the log of the GLM prediction. For a policy where the GLM is well-calibrated, this correction is near zero. For a policy in a combination of factor levels where the GLM is systematically wrong — the young driver in a high vehicle group — the network learns a positive correction.

This correction is what we interrogate with NID. The network has learned something. NID asks: what structure did it learn, specifically which features had to co-participate in the same hidden units to produce that correction?

### The MLP-M variant

The standard CANN uses a single MLP that takes all features as input. This works, but it risks learning main effects from the residuals of an imperfect GLM (which will always have some residual structure in the main effects due to finite sample size and mild model misspecification).

The MLP-M variant addresses this by adding a separate small univariate network for each feature. These univariate networks absorb the main effect residuals, leaving only the interaction signal for the main MLP to learn. This reduces false positive interactions at the cost of more training parameters.

We will use the standard variant (`mlp_m=False`) in this module. For production work on a real motor portfolio, `mlp_m=True` is recommended, especially when features are correlated (age and NCD always are in UK motor).