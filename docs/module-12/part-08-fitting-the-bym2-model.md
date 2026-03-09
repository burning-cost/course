## Part 8: Fitting the BYM2 model

With the adjacency matrix built and spatial autocorrelation confirmed, we can fit the model. This is where PyMC does the work.

### Setting up the model

```python
from insurance_spatial import BYM2Model

model = BYM2Model(
    adjacency=adj,
    draws=1000,     # posterior samples per chain (post-warmup)
    chains=4,       # number of independent MCMC chains
    target_accept=0.9,  # NUTS acceptance rate; increase to 0.95 if divergences
    tune=1000,      # warmup steps per chain
)

print(f"BYM2Model configured:")
print(f"  Areas:          {adj.n}")
print(f"  Scaling factor: {adj.scaling_factor:.4f}")
print(f"  Draws per chain: {model.draws}")
print(f"  Chains:          {model.chains}")
print(f"  Total samples:   {model.draws * model.chains:,}")
```

### Running the sampler

```python
result = model.fit(
    claims=claims,
    exposure=exposure.astype(float),
    random_seed=42,
)

print("Fitting complete.")
print(f"Areas in result: {result.n_areas}")
print(f"Areas list (first 5): {result.areas[:5]}")
```

On a Databricks cluster (typically 4--16 cores), this 100-area model takes 2--5 minutes with 4 chains of 1,000 draws each. The progress bars show separately for the tuning phase and the sampling phase. A healthy run looks like:

```
Sampling: [==========] 100% 0:04:12
```

If you see warnings about "divergences" during sampling -- we address those in Part 9. If you see `RuntimeError: Chain 1 failed` -- this usually means PyMC cannot be found or there is a version conflict; restart the cluster and rerun the install cell.

### If the cluster runs out of memory

For a 100-area model, memory is not a concern. For 11,200 sectors, the ICAR precision matrix and gradient computations are substantial. If you hit out-of-memory errors on a larger dataset:

1. Reduce chains to 2 (minimum for R-hat diagnostics)
2. Reduce draws to 500 (enough for basic convergence checking)
3. Use a district-level model (N≈3,000) for exploratory work, then scale to sectors