# Module 12 Exercises: Spatial Territory Rating

Ten exercises. Work through them in order -- each builds on the adjacency matrix, synthetic dataset, and fitted model from the earlier exercises. The final exercise uses a larger district-level dataset and requires interpreting rho to make a real modelling decision.

Before starting: read Parts 1--13 of the tutorial. Every concept used here is explained there.

---

## Exercise 1: Diagnosing the Emblem banding problem

**Reference:** Tutorial Parts 1--2

**What you will do:** Generate a synthetic sector-level dataset, apply k-means banding, and quantify the discontinuity problem at band boundaries.

**Context.** You have sector-level claim data for a UK motor book. A colleague has proposed 6-band k-means territory banding. Before agreeing, you want to measure how large the artificial boundary effects are and whether neighbouring sectors end up in wildly different bands.

### Setup

In a new notebook cell:

```python
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

rng = np.random.default_rng(seed=101)

# 100 postcode sectors arranged in a 10x10 grid
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS

# Smooth underlying spatial risk: a north-south gradient
row_idx = np.array([r for r in range(NROWS) for c in range(NCOLS)])
true_log_effect = 0.4 * (1 - row_idx / (NROWS - 1)) - 0.2  # ranges from +0.2 (north) to -0.2 (south)
true_log_effect += rng.normal(0, 0.06, N)  # small area scatter

exposure = rng.gamma(shape=2.0, scale=15.0, size=N).astype(int) + 3
base_freq = 0.07
expected_claims = exposure * base_freq * np.exp(true_log_effect)
claims = rng.poisson(expected_claims)

portfolio_freq = claims.sum() / exposure.sum()
log_oe = np.log((claims + 0.5) / (exposure * portfolio_freq + 0.5))
```

### Tasks

**Task 1.** Apply k-means banding with K=6 to the `log_oe` array. Assign each area its band relativity (the mean O/E ratio within its band). Print the six band relativities in ascending order. What is the maximum ratio between adjacent bands?

**Task 2.** Identify all pairs of adjacent areas (row or column neighbours only -- rook connectivity) that are in different territory bands. For each such pair, compute the absolute difference in band relativity. Report the mean and maximum boundary jump across all adjacent cross-band pairs.

For rook adjacency on the grid, area `r*NCOLS + c` is adjacent to `(r-1)*NCOLS + c`, `(r+1)*NCOLS + c`, `r*NCOLS + (c-1)`, and `r*NCOLS + (c+1)`.

**Task 3.** For a smooth underlying spatial process like the one we generated, what should the expected boundary jump between adjacent areas be? Compare your answer to the actual maximum boundary jump from Task 2. Is the k-means approach creating jumps larger than the underlying process would justify?

**Task 4.** Repeat Tasks 1 and 2 with K=10 and K=4. Does increasing K make the boundary problem better or worse? Why?

<details>
<summary>Solution</summary>

```python
# Task 1: k-means banding
K = 6
km = KMeans(n_clusters=K, random_state=42, n_init=10)
km.fit(log_oe.reshape(-1, 1))
bands = km.labels_

band_rel = np.zeros(K)
for k in range(K):
    mask = bands == k
    band_rel[k] = (claims[mask].sum() / exposure[mask].sum()) / portfolio_freq

sorted_band_rel = np.sort(band_rel)
print("Band relativities (sorted):", sorted_band_rel.round(4))
max_adj_ratio = sorted_band_rel[1:] / sorted_band_rel[:-1]
print(f"Max ratio between adjacent bands: {max_adj_ratio.max():.4f}")

# Task 2: rook-adjacent cross-band pairs
area_band = bands  # length N array

def rook_neighbours(i, nrows, ncols):
    r, c = divmod(i, ncols)
    nbrs = []
    if r > 0: nbrs.append((r-1)*ncols + c)
    if r < nrows-1: nbrs.append((r+1)*ncols + c)
    if c > 0: nbrs.append(r*ncols + c - 1)
    if c < ncols-1: nbrs.append(r*ncols + c + 1)
    return nbrs

# Assign band relativities to areas
area_band_rel = band_rel[bands]

cross_band_jumps = []
for i in range(N):
    for j in rook_neighbours(i, NROWS, NCOLS):
        if j > i and area_band[i] != area_band[j]:
            jump = abs(area_band_rel[i] - area_band_rel[j])
            cross_band_jumps.append(jump)

cross_band_jumps = np.array(cross_band_jumps)
print(f"\nCross-band boundary pairs: {len(cross_band_jumps)}")
print(f"Mean boundary jump:         {cross_band_jumps.mean():.4f}")
print(f"Max boundary jump:          {cross_band_jumps.max():.4f}")

# Task 3: for a smooth process, the expected jump between adjacent areas
# is determined by the spatial gradient. Our true_log_effect changes by
# 0.4 / 9 ≈ 0.044 per row on the log scale, equivalent to a relativity
# change of exp(0.044) - 1 ≈ 4.5%. But the banding creates jumps far larger.
true_area_rel = np.exp(true_log_effect - true_log_effect.mean())
true_jumps = []
for i in range(N):
    for j in rook_neighbours(i, NROWS, NCOLS):
        if j > i:
            true_jumps.append(abs(true_area_rel[i] - true_area_rel[j]))
true_jumps = np.array(true_jumps)
print(f"\nTrue underlying mean boundary change:  {true_jumps.mean():.4f}")
print(f"True underlying max boundary change:   {true_jumps.max():.4f}")
print(f"K-means max boundary jump:             {cross_band_jumps.max():.4f}")
print(f"Amplification factor: {cross_band_jumps.max() / true_jumps.max():.1f}x")

# Task 4: K=10 and K=4
for K_test in [4, 10]:
    km_t = KMeans(n_clusters=K_test, random_state=42, n_init=10)
    km_t.fit(log_oe.reshape(-1, 1))
    bands_t = km_t.labels_
    band_rel_t = np.zeros(K_test)
    for k in range(K_test):
        mask = bands_t == k
        if mask.sum() > 0:
            band_rel_t[k] = (claims[mask].sum() / exposure[mask].sum()) / portfolio_freq
    area_band_rel_t = band_rel_t[bands_t]
    jumps_t = []
    for i in range(N):
        for j in rook_neighbours(i, NROWS, NCOLS):
            if j > i and bands_t[i] != bands_t[j]:
                jumps_t.append(abs(area_band_rel_t[i] - area_band_rel_t[j]))
    if jumps_t:
        print(f"K={K_test}: mean jump = {np.array(jumps_t).mean():.4f}, max jump = {np.array(jumps_t).max():.4f}")
```

**Key finding.** With K=6, the k-means banding creates boundary jumps that are substantially larger than the true smooth underlying gradient. Increasing K to 10 makes individual bands more granular but introduces *more* cross-band boundaries and potentially *larger* individual jumps if sparse bands happen to be adjacent. The fundamental problem is that banding imposes a piecewise-constant approximation on a continuous spatial process.

</details>

---

## Exercise 2: Building and inspecting an adjacency matrix

**Reference:** Tutorial Parts 5--6

**What you will do:** Build adjacency matrices with both rook and queen connectivity, compare their properties, and verify that the graph is connected.

### Setup

Continue in the same notebook, or start a new cell. Use the data from Exercise 1.

### Tasks

**Task 1.** Build two adjacency matrices from the 10x10 grid: one with rook connectivity and one with queen connectivity. For each, print:
- Total number of edges
- Mean, minimum, and maximum number of neighbours per area
- Number of connected components
- Scaling factor

**Task 2.** Area `r0c0` is the top-left corner. List its neighbours under rook and queen connectivity. Why does it have fewer neighbours than `r5c5`?

**Task 3.** The scaling factor is a property of the graph topology, not the data. Verify this by building two grids with different dimensions -- 5x5 and 20x20 -- both with queen connectivity. Print the scaling factor for each. What pattern do you observe?

**Task 4.** Explain in plain English why the ICAR model requires a connected graph. What would go wrong if the graph had two disconnected components?

**Task 5.** Simulate an island: set row W[0, :] = 0 and W[:, 0] = 0 in the adjacency matrix for the rook case (making r0c0 isolated). Confirm with `n_components()` that the graph is now disconnected. What is the correct way to handle this in a real BYM2 analysis?

<details>
<summary>Solution</summary>

```python
from insurance_spatial import build_grid_adjacency

# Task 1
adj_rook = build_grid_adjacency(NROWS, NCOLS, connectivity="rook")
adj_queen = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")

for name, adj in [("Rook", adj_rook), ("Queen", adj_queen)]:
    nc = adj.neighbour_counts()
    print(f"\n{name} connectivity:")
    print(f"  Total edges:     {adj.W.nnz // 2}")
    print(f"  Mean neighbours: {nc.mean():.2f}")
    print(f"  Min neighbours:  {nc.min()}")
    print(f"  Max neighbours:  {nc.max()}")
    print(f"  Components:      {adj.n_components()}")
    print(f"  Scaling factor:  {adj.scaling_factor:.4f}")

# Task 2
idx_r0c0 = adj_rook.area_index()["r0c0"]
idx_r5c5 = adj_rook.area_index()["r5c5"]

def get_neighbours(adj, idx):
    row = adj.W.getrow(idx).toarray().ravel()
    return [adj.areas[j] for j in np.where(row == 1)[0]]

print("\nNeighbours of r0c0 (rook):", get_neighbours(adj_rook, idx_r0c0))
print("Neighbours of r0c0 (queen):", get_neighbours(adj_queen, idx_r0c0))
print("Neighbours of r5c5 (rook):", get_neighbours(adj_rook, idx_r5c5))
print("Neighbours of r5c5 (queen):", get_neighbours(adj_queen, idx_r5c5))

# Task 3
for dims in [(5, 5), (10, 10), (20, 20)]:
    adj_test = build_grid_adjacency(*dims, connectivity="queen")
    print(f"{dims[0]}x{dims[1]} grid: scaling factor = {adj_test.scaling_factor:.4f}")

# Task 5: simulate an island
import scipy.sparse as sp

W_lil = adj_rook.W.tolil()
# Isolate r0c0 (index 0)
W_lil[0, :] = 0
W_lil[:, 0] = 0
W_island = sp.csr_matrix(W_lil)

from insurance_spatial.adjacency import AdjacencyMatrix
adj_island = AdjacencyMatrix(W=W_island, areas=adj_rook.areas)
print(f"\nComponents with island: {adj_island.n_components()}")
print("Fix: use from_geojson(fix_islands=True) which connects islands to nearest mainland node.")
```

**Task 3 observation.** The scaling factor is roughly constant across grid sizes for queen connectivity -- it reflects the geometry of a regular planar graph, not the size. For irregular real-world geometries, the scaling factor differs.

**Task 4 explanation.** The ICAR precision matrix Q = D - W is singular with rank N-1 for a connected graph. For a disconnected graph with k components, Q has rank N-k -- k zero eigenvalues instead of one. The ICAR model is then undefined: there is no unique joint distribution over the areas because the components are independent with no common reference point. PyMC's `pm.ICAR` will fail or produce nonsense with a disconnected graph.

</details>

---

## Exercise 3: Moran's I -- testing before modelling

**Reference:** Tutorial Part 6

**What you will do:** Run Moran's I under different scenarios and interpret the results correctly.

### Tasks

**Task 1.** Using the data from Exercise 1 and the queen adjacency matrix, compute Moran's I on `log_oe`. Print the statistic, z-score, p-value, and interpretation. Is the result significant?

**Task 2.** Generate a dataset with *no* spatial structure: replace `true_log_effect` with `rng.normal(0, 0.2, N)` (IID noise, no gradient). Recompute claims and log_oe with this new effect. Run Moran's I. What do you expect? What do you observe?

**Task 3.** Run Moran's I twice on the original spatially-structured data: once with `n_permutations=99` and once with `n_permutations=9999`. Compare the p-values. When does the number of permutations matter?

**Task 4.** The Moran's I statistic ranges from -1 to +1. We said that positive spatial autocorrelation is "nearby areas have similar values." What would a *negative* Moran's I mean, and can you construct a synthetic case where it appears?

**Task 5.** Suppose Moran's I on your pre-fit residuals is I=0.18, z=2.3, p=0.021. The model is fitted and post-fit Moran's I is I=0.12, z=1.6, p=0.11. Write two sentences that a pricing committee would understand explaining what these two results tell you.

<details>
<summary>Solution</summary>

```python
from insurance_spatial.diagnostics import moran_i

# Task 1: original spatially-structured data
adj_queen = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")
test_orig = moran_i(log_oe, adj_queen, n_permutations=999)
print("Original data:")
print(f"  I = {test_orig.statistic:.4f}, z = {test_orig.z_score:.2f}, p = {test_orig.p_value:.4f}")
print(f"  {test_orig.interpretation}")

# Task 2: no spatial structure
rng2 = np.random.default_rng(seed=202)
true_log_null = rng2.normal(0, 0.2, N)
expected_null_data = exposure * base_freq * np.exp(true_log_null)
claims_null = rng2.poisson(expected_null_data)
log_oe_null = np.log((claims_null + 0.5) / (exposure * portfolio_freq + 0.5))

test_null = moran_i(log_oe_null, adj_queen, n_permutations=999)
print("\nNo spatial structure:")
print(f"  I = {test_null.statistic:.4f}, z = {test_null.z_score:.2f}, p = {test_null.p_value:.4f}")
print(f"  {test_null.interpretation}")

# Task 3: permutation count
test_99   = moran_i(log_oe, adj_queen, n_permutations=99)
test_9999 = moran_i(log_oe, adj_queen, n_permutations=9999)
print(f"\n99 permutations:    p = {test_99.p_value:.4f}")
print(f"9999 permutations:  p = {test_9999.p_value:.4f}")

# Task 4: negative Moran's I -- checkerboard pattern
# Create alternating high/low values
checker = np.zeros(N)
for i in range(N):
    r, c = divmod(i, NCOLS)
    checker[i] = 0.3 if (r + c) % 2 == 0 else -0.3
test_checker = moran_i(checker, adj_queen, n_permutations=999)
print(f"\nCheckerboard pattern: I = {test_checker.statistic:.4f}")
print("Negative I means nearby areas are DISSIMILAR (high surrounded by low and vice versa).")
```

**Task 5 answer.** "Before fitting, we found significant positive spatial autocorrelation (I=0.18, p=0.021), meaning postcode sectors with higher claim frequencies cluster together geographically and spatial smoothing is warranted. After fitting, the spatial autocorrelation in the residuals is no longer significant (I=0.12, p=0.11), indicating that the BYM2 model has successfully captured the geographic pattern and no systematic spatial structure remains unexplained."

</details>

---

## Exercise 4: Fitting BYM2 and reading MCMC diagnostics

**Reference:** Tutorial Parts 8--9

**What you will do:** Fit BYM2 on the Exercise 1 dataset, run diagnostics, and identify a convergence problem if one exists.

### Tasks

**Allow 3--5 minutes for the sampler to run** with 4 chains and 1,000 draws on Databricks Free Edition.

**Task 1.** Fit BYM2 on the Exercise 1 data using 4 chains and 1,000 draws. Use the queen adjacency matrix. After fitting, print R-hat and ESS for all parameters. Does the model converge?

**Task 2.** Produce a trace plot for `alpha`, `sigma`, and `rho`. Describe what you see. Does the trace look like healthy "hairy caterpillars" or do you see any chains that are drifting or stuck?

**Task 3.** Fit a *deliberately broken* model by using only 1 chain and 50 draws. Run diagnostics. What do the R-hat values look like? What does this teach you about the minimum sampling budget for a spatial model?

**Task 4.** Look at the posterior mean and 95% credibility interval for `rho`. What does this value tell you about the synthetic data we generated? Is the geographic variation primarily spatially structured or primarily area-specific noise?

**Task 5.** Look at the posterior of `sigma`. The units are log-scale. Convert sigma's posterior mean to a "multiplicative spread": compute `exp(sigma_mean)` and `exp(-sigma_mean)`. What does this tell you about the range of territory relativities implied by the model?

<details>
<summary>Solution</summary>

```python
from insurance_spatial import BYM2Model
from insurance_spatial.plots import plot_trace
import matplotlib.pyplot as plt

# Task 1
adj_queen = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")
model = BYM2Model(adjacency=adj_queen, draws=1000, chains=4, target_accept=0.9, tune=1000)
result = model.fit(claims=claims, exposure=exposure.astype(float), random_seed=42)

diag = result.diagnostics()
print("Convergence diagnostics:")
print(f"  Max R-hat:    {diag.convergence.max_rhat:.4f}  (want < 1.01)")
print(f"  Min ESS bulk: {diag.convergence.min_ess_bulk:.0f}  (want > 400)")
print(f"  Min ESS tail: {diag.convergence.min_ess_tail:.0f}  (want > 400)")
print(f"  Divergences:  {diag.convergence.n_divergences}")
print(f"  Converged:    {diag.convergence.converged}")
print()
print(diag.convergence.rhat_by_param)

# Task 2: trace plot
fig = plot_trace(result, params=["alpha", "sigma", "rho"])
plt.tight_layout()
plt.show()

# Task 3: deliberately broken fit
model_broken = BYM2Model(adjacency=adj_queen, draws=50, chains=1, tune=100)
result_broken = model_broken.fit(claims=claims, exposure=exposure.astype(float), random_seed=42)
diag_broken = result_broken.diagnostics()
print("\nBroken model (1 chain, 50 draws):")
print(f"  Max R-hat:    {diag_broken.convergence.max_rhat:.4f}")
print(f"  Min ESS bulk: {diag_broken.convergence.min_ess_bulk:.0f}")
print("  Note: R-hat is undefined for a single chain -- you need at least 2 chains.")
print("  With only 50 draws, ESS will be very low.")

# Task 4: rho interpretation
print("\nrho posterior:")
print(diag.rho_summary)

# Task 5: sigma interpretation
sigma_mean = float(diag.sigma_summary["mean"][0])
print(f"\nsigma posterior mean: {sigma_mean:.4f}")
print(f"exp(sigma_mean):      {np.exp(sigma_mean):.4f}  -- factor above mean for +1 SD area")
print(f"exp(-sigma_mean):     {np.exp(-sigma_mean):.4f} -- factor below mean for -1 SD area")
print(f"Implied 95% range: exp(±1.96*sigma) = [{np.exp(-1.96*sigma_mean):.3f}, {np.exp(1.96*sigma_mean):.3f}]")
```

**Task 3 note.** With `chains=1`, R-hat is undefined (it requires at least 2 chains). ArviZ will return NaN or 1.0 for single-chain R-hat. With 50 draws, ESS will be far below 400 for spatial parameters. Always use at least 2 chains. 4 chains is strongly preferred.

**Task 4.** Because we generated data with a smooth north-south gradient, we expect rho to be moderately high (0.5--0.8). A value of 0.6 means 60% of the territory variance is spatially structured -- this is consistent with the deliberate gradient plus small IID area scatter we added.

</details>

---

## Exercise 5: Extracting and comparing relativities

**Reference:** Tutorial Parts 11--12

**What you will do:** Extract territory relativities from the fitted model, compare them to the true values, and compare to naive O/E.

### Tasks

**Task 1.** Extract relativities from the fitted model using the default (geometric mean) normalisation. Print the 10 highest-risk and 10 lowest-risk areas. Do the high-risk areas cluster in the north of the grid (rows 0--2) as expected from the data generating process?

**Task 2.** Extract relativities using `r0c5` as the base area (relativity = 1.0). Confirm that `r0c5`'s relativity is exactly 1.0 in the output. How does the range of relativities change compared to the grand-mean normalisation?

**Task 3.** Compute the mean absolute error of the BYM2 relativities versus the true relativities (computed as `np.exp(true_log_effect - true_log_effect.mean())`). Then compute the same MAE for naive O/E relativities (sector observed frequency divided by portfolio frequency). Which is more accurate?

**Task 4.** Find the three areas with the widest credibility intervals (highest `upper - lower`). What do they have in common? Look up their exposure values.

**Task 5.** Write a Polars expression that adds a column `uncertainty_flag` to the relativities DataFrame, set to `True` when the 95% credibility interval width exceeds 0.5 (i.e., the upper bound is more than 0.5 above the lower bound). How many areas are flagged? What would you do with these areas in a real pricing review?

<details>
<summary>Solution</summary>

```python
# Task 1
rels = result.territory_relativities()
print("Top 10 highest risk:")
print(rels.sort("relativity", descending=True).head(10).select(["area", "relativity", "lower", "upper"]))
print("\nTop 10 lowest risk:")
print(rels.sort("relativity").head(10).select(["area", "relativity", "lower", "upper"]))

# Check: do high-risk areas cluster in north (rows 0-2)?
rels_with_row = rels.with_columns(
    pl.col("area").str.extract(r"r(\d+)c", 1).cast(pl.Int32).alias("row")
)
top20 = rels_with_row.sort("relativity", descending=True).head(20)
print(f"\nMean row of top 20 areas: {top20['row'].mean():.2f}  (lower = more northerly, i.e., rows 0-2)")

# Task 2
rels_ref = result.territory_relativities(base_area="r0c5")
print("\nWith r0c5 as reference:")
print(rels_ref.filter(pl.col("area") == "r0c5"))
print(f"Relativity range: [{rels_ref['relativity'].min():.4f}, {rels_ref['relativity'].max():.4f}]")
print(f"Grand-mean range: [{rels['relativity'].min():.4f}, {rels['relativity'].max():.4f}]")

# Task 3
import numpy as np
true_rel = np.exp(true_log_effect - true_log_effect.mean())
bym2_rel = np.array(rels.sort("area")["relativity"].to_list())
naive_rel = (claims / exposure) / portfolio_freq

bym2_mae = np.abs(bym2_rel - true_rel).mean()
naive_mae = np.abs(naive_rel - true_rel).mean()
print(f"\nBYM2 MAE vs truth:  {bym2_mae:.4f}")
print(f"Naive O/E MAE:      {naive_mae:.4f}")
print(f"BYM2 improvement:   {(1 - bym2_mae/naive_mae)*100:.1f}%")

# Task 4
rels_ci = rels.with_columns(
    (pl.col("upper") - pl.col("lower")).alias("ci_width")
)
widest = rels_ci.sort("ci_width", descending=True).head(3)
print("\nWidest credibility intervals:")
print(widest.select(["area", "relativity", "lower", "upper", "ci_width"]))

# Look up exposure for these areas
widest_areas = widest["area"].to_list()
df_exposure = pl.DataFrame({"area": areas, "exposure": exposure.tolist()})
print("\nExposure for widest CI areas:")
print(df_exposure.filter(pl.col("area").is_in(widest_areas)))

# Task 5
rels_flagged = rels.with_columns(
    ((pl.col("upper") - pl.col("lower")) > 0.5).alias("uncertainty_flag")
)
n_flagged = rels_flagged["uncertainty_flag"].sum()
print(f"\nAreas with CI width > 0.5: {n_flagged}")
print("In a real review: these areas would either be grouped with adjacent sectors")
print("or capped at a maximum departure from 1.0 in the live rating table.")
```

</details>

---

## Exercise 6: The two-stage pipeline

**Reference:** Tutorial Part 13

**What you will do:** Implement a two-stage pipeline where a simple base model is fitted first and BYM2 is applied to the residual O/E.

**Context.** In production, you would have a full CatBoost or GLM base model. Here, we simulate a base model by fitting an intercept-only Poisson GLM (equivalent to using the portfolio average frequency as the prediction).

### Setup

```python
import numpy as np
from insurance_spatial import build_grid_adjacency, BYM2Model

rng = np.random.default_rng(seed=42)
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS

row_idx = np.array([r for r in range(NROWS) for c in range(NCOLS)])
true_log_effect = 0.4 * (1 - row_idx / (NROWS - 1)) - 0.2
true_log_effect += rng.normal(0, 0.06, N)

exposure = rng.gamma(shape=2.0, scale=15.0, size=N).astype(int) + 3
base_freq = 0.07
expected_claims = exposure * base_freq * np.exp(true_log_effect)
claims = rng.poisson(expected_claims)
portfolio_freq = claims.sum() / exposure.sum()

adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")
```

### Tasks

**Task 1.** Implement a "base model" that predicts expected claims per area using a Poisson GLM with an age covariate. Generate a synthetic age covariate for each area (use `rng.uniform(30, 50, N)` as mean age) and fit a simple age-only model using `scipy.stats.poisson` or just by computing the frequency at each age quintile. The key point is that the base model should capture some non-spatial variation but not the spatial pattern.

For simplicity, use this base model: `expected_base = exposure * np.exp(0.05 * (age - age.mean()))` where age is the synthetic covariate. This is a base model that knows about age but nothing about geography.

**Task 2.** Compute sector-level observed claims (same as `claims`) and sector-level expected claims from the base model. Pass these to BYM2 as `claims=sector_observed, exposure=sector_expected`. Fit the model with 4 chains and 1,000 draws.

**Task 3.** Compare the rho posterior mean from the two-stage model to the rho posterior mean from Exercise 4 (the integrated model). Is rho higher, lower, or similar? Why might it differ?

**Task 4.** Extract relativities from both models. Compute the correlation between the two sets of relativities. Are they telling the same story, or do they give materially different territory factors?

**Task 5.** Explain the argument for using the two-stage approach in production even if the integrated model gives numerically similar relativities.

<details>
<summary>Solution</summary>

```python
# Task 1: base model with age covariate
rng_age = np.random.default_rng(seed=99)
age = rng_age.uniform(30, 50, N)
base_expected = exposure * np.exp(0.05 * (age - age.mean()))
# Scale to match portfolio total (so O/E ratios are centred near 1)
base_expected = base_expected * (claims.sum() / base_expected.sum())

print("Base model sanity check:")
print(f"  Sum expected: {base_expected.sum():.1f}")
print(f"  Sum observed: {claims.sum()}")
print(f"  Ratio: {claims.sum() / base_expected.sum():.4f}  (should be ~1.0)")

# Task 2: two-stage BYM2
model_2s = BYM2Model(adjacency=adj, draws=1000, chains=4, target_accept=0.9, tune=1000)
result_2s = model_2s.fit(
    claims=claims,
    exposure=base_expected,  # base model prediction as "exposure"
    random_seed=42,
)
diag_2s = result_2s.diagnostics()
print(f"\nTwo-stage model:")
print(f"  Max R-hat: {diag_2s.convergence.max_rhat:.4f}")
print(f"  rho mean:  {diag_2s.rho_summary['mean'][0]:.3f}")
print(f"  rho 95%CI: [{diag_2s.rho_summary['q025'][0]:.3f}, {diag_2s.rho_summary['q975'][0]:.3f}]")

# Task 3: compare rho
# From Exercise 4, integrated model
diag_int = result.diagnostics()  # result from Exercise 4
print(f"\nIntegrated model rho mean: {diag_int.rho_summary['mean'][0]:.3f}")
print(f"Two-stage model rho mean:  {diag_2s.rho_summary['mean'][0]:.3f}")

# Task 4: correlate relativities
rels_int = result.territory_relativities()
rels_2s  = result_2s.territory_relativities()

# Ensure same ordering
rels_int_sorted = rels_int.sort("area")["relativity"].to_numpy()
rels_2s_sorted  = rels_2s.sort("area")["relativity"].to_numpy()

corr = np.corrcoef(rels_int_sorted, rels_2s_sorted)[0, 1]
print(f"\nCorrelation between integrated and two-stage relativities: {corr:.4f}")
print(f"Mean absolute difference: {np.abs(rels_int_sorted - rels_2s_sorted).mean():.4f}")

# Task 5
print("""
Two-stage advantages in production:
1. Decoupling: main risk model can be retrained independently of territory model.
2. Auditability: regulators can review spatial smoothing without understanding the full GBM.
3. Stability: territory factors change less year-on-year when derived from O/E residuals
   rather than raw claims, because the base model absorbs non-spatial variation.
4. Interpretability: the rho parameter in the two-stage model has a cleaner meaning
   (proportion of RESIDUAL geographic variation that is spatially structured).
""")
```

</details>

---

## Exercise 7: Handling sparse areas

**Reference:** Tutorial Parts 9, 11

**What you will do:** Create a version of the dataset with extreme sparsity and compare how BYM2 handles it versus naive O/E.

### Tasks

**Task 1.** Create a sparse dataset by reducing exposure dramatically for the northern areas (rows 0--2). Set their exposure to `rng.integers(1, 4, size_of_north_block)` -- 1 to 3 policy-years each. Keep southern exposure at normal levels. Recompute claims using the same true_log_effect.

**Task 2.** For the sparse northern areas, how many have zero observed claims? Print the naive O/E relativity for each of these zero-claim areas.

**Task 3.** Fit BYM2 on this sparse dataset. Extract the relativities for the northern areas (rows 0--2). Compare the BYM2 estimates to the naive O/E estimates. What is the BYM2 model doing for zero-claim areas?

**Task 4.** Run Moran's I on the log O/E of the sparse dataset. Compare the statistic and p-value to the non-sparse version from Exercise 3. Does sparsity affect the power of the Moran's I test?

**Task 5.** A pricing actuary sees a BYM2 territory relativity of 0.92 for a northern area that had zero observed claims. She says: "We have no claims data there, so the model should give us 1.0 (the mean) and not 0.92." How do you respond? What is the model actually doing with that area?

<details>
<summary>Solution</summary>

```python
# Task 1: sparse northern areas
rng_sparse = np.random.default_rng(seed=303)
exposure_sparse = exposure.copy()
north_mask = row_idx <= 2
exposure_sparse[north_mask] = rng_sparse.integers(1, 4, north_mask.sum())
expected_sparse = exposure_sparse * base_freq * np.exp(true_log_effect)
claims_sparse = rng_sparse.poisson(expected_sparse)

print(f"Northern exposure range: {exposure_sparse[north_mask].min()} to {exposure_sparse[north_mask].max()}")
print(f"Southern exposure range: {exposure_sparse[~north_mask].min()} to {exposure_sparse[~north_mask].max()}")

# Task 2: zero-claim northern areas
north_zero = north_mask & (claims_sparse == 0)
print(f"\nNorthern areas with zero claims: {north_zero.sum()}")
portfolio_freq_sparse = claims_sparse.sum() / exposure_sparse.sum()
naive_oe_sparse = (claims_sparse + 0.5) / (exposure_sparse * portfolio_freq_sparse + 0.5)
print("Naive O/E for zero-claim northern areas:")
zero_naive = naive_oe_sparse[north_zero]
print(f"  Range: [{zero_naive.min():.3f}, {zero_naive.max():.3f}]")
print(f"  These are all driven by the +0.5 Haldane correction, giving low estimates for sparse areas")

# Task 3: BYM2 on sparse data
model_sparse = BYM2Model(adjacency=adj, draws=1000, chains=4, target_accept=0.9, tune=1000)
result_sparse = model_sparse.fit(
    claims=claims_sparse,
    exposure=exposure_sparse.astype(float),
    random_seed=42,
)
rels_sparse = result_sparse.territory_relativities()
# Filter for northern areas
rels_sparse_north = rels_sparse.with_columns(
    pl.col("area").str.extract(r"r(\d+)c", 1).cast(pl.Int32).alias("row")
).filter(pl.col("row") <= 2)

print("\nBYM2 relativities for northern (sparse) areas:")
print(rels_sparse_north.select(["area", "relativity", "lower", "upper"]).head(10))

# Task 4: Moran's I on sparse data
log_oe_sparse = np.log((claims_sparse + 0.5) / (exposure_sparse * portfolio_freq_sparse + 0.5))
test_sparse = moran_i(log_oe_sparse, adj, n_permutations=999)
print(f"\nMoran's I (sparse):     I = {test_sparse.statistic:.4f}, p = {test_sparse.p_value:.4f}")
print(f"Moran's I (non-sparse): I = {test_orig.statistic:.4f}, p = {test_orig.p_value:.4f}")
print("Sparsity adds noise to log O/E, which may reduce the apparent spatial signal.")

# Task 5
print("""
Response to the actuary:
The model does not give 0.92 because it ignores the data for that area.
It gives 0.92 because:
1. The area has zero observed claims -- consistent with both low risk AND bad luck.
2. Its neighbours have elevated risk (high rho means the ICAR component pulls
   the area towards its neighbours' values).
3. The model weights the zero-claim observation against the spatial prior from
   neighbours. With only 2 policy-years of exposure, the zero-claim observation
   is weak evidence. The neighbour prior (which says this area is probably
   elevated, given its geography) is relatively strong.
4. The result is shrinkage towards the neighbours' level: 0.92 rather than 1.0
   (the grand mean) or a very low value (naive O/E).
This is the credibility principle made spatial: borrow strength from neighbours
when own data are thin.
""")
```

</details>

---

## Exercise 8: Choropleth maps with real UK boundaries

**Reference:** Tutorial Part 15

**This exercise requires additional setup time (30--60 minutes for data download and preprocessing).** The boundary file download from ONS may be slow; the geopandas adjacency construction on district-level boundaries takes several minutes. Plan accordingly.

**What you will do:** This exercise requires the optional geo dependencies. Install them first, then load a sample UK boundary file and build an adjacency matrix.

### Setup

```python
%pip install "insurance-spatial[geo]" geopandas --quiet
dbutils.library.restartPython()
```

If you are working locally:
```bash
uv add "insurance-spatial[geo]" geopandas
```

### Tasks

**Task 1.** Download the ONS Local Authority District boundaries for England and Wales from the ONS Open Geography Portal:

```sql
https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Local_Authority_Districts_December_2023_Boundaries_UK_BFE/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson
```

Load this as a GeoDataFrame and inspect the columns. Which column contains the district codes?

**Task 2.** Build an adjacency matrix from the downloaded boundary file using `from_geojson()`. Specify the correct `area_col`. Use queen connectivity. How many districts are in the file? How many connected components are there? Does `fix_islands=True` need to do any work?

**Task 3.** Print the minimum and maximum neighbour counts. Which district has the fewest neighbours? Which has the most? Do these make geographic sense?

**Task 4.** Create synthetic claims and exposure data for each district (use `rng.gamma(shape=3, scale=50, N)` for exposure and a Poisson draw from `exposure * 0.06` for claims). Run Moran's I. Is spatial autocorrelation significant even in this random data?

**Task 5.** Generate a choropleth map using `plot_choropleth()` from `insurance_spatial.plots`. Use the synthetic relativities from Task 4. Save the figure to a file.

<details>
<summary>Solution</summary>

```python
import geopandas as gpd
import numpy as np
from insurance_spatial.adjacency import from_geojson
from insurance_spatial.diagnostics import moran_i
from insurance_spatial.plots import plot_choropleth
from insurance_spatial import BYM2Model
import matplotlib.pyplot as plt

# Task 1: download boundaries
url = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
       "Local_Authority_Districts_December_2023_Boundaries_UK_BFE/FeatureServer/0/query"
       "?where=1%3D1&outFields=*&f=geojson")
gdf = gpd.read_file(url)
print("Columns:", list(gdf.columns))
print("Shape:", gdf.shape)
# Look for district code column -- typically "LAD23CD"
code_col = [c for c in gdf.columns if "CD" in c and "LAD" in c][0]
print(f"District code column: {code_col}")

# Task 2: adjacency matrix
# Note: file path approach for GeoJSON saved locally
gdf.to_file("/tmp/lad_2023.geojson", driver="GeoJSON")
adj_lad = from_geojson("/tmp/lad_2023.geojson", area_col=code_col, connectivity="queen", fix_islands=True)
print(f"\nDistricts: {adj_lad.n}")
print(f"Components: {adj_lad.n_components()}")

# Task 3: neighbour counts
nc = adj_lad.neighbour_counts()
min_idx = nc.argmin()
max_idx = nc.argmax()
print(f"Min neighbours: {nc.min()} ({adj_lad.areas[min_idx]})")
print(f"Max neighbours: {nc.max()} ({adj_lad.areas[max_idx]})")

# Task 4: synthetic data + Moran's I
rng_uk = np.random.default_rng(seed=44)
N_lad = adj_lad.n
exposure_lad = rng_uk.gamma(shape=3, scale=50, size=N_lad).astype(int) + 10
claims_lad   = rng_uk.poisson(exposure_lad * 0.06)
portfolio_freq_lad = claims_lad.sum() / exposure_lad.sum()
log_oe_lad = np.log((claims_lad + 0.5) / (exposure_lad * portfolio_freq_lad + 0.5))

test_lad = moran_i(log_oe_lad, adj_lad, n_permutations=999)
print(f"\nMoran's I (random synthetic data): I = {test_lad.statistic:.4f}, p = {test_lad.p_value:.4f}")
print("For random data with no true spatial structure, we expect I near 0 and p > 0.05.")

# Task 5: choropleth
model_lad = BYM2Model(adjacency=adj_lad, draws=500, chains=2, tune=500)
result_lad = model_lad.fit(claims=claims_lad, exposure=exposure_lad.astype(float), random_seed=42)
rels_lad = result_lad.territory_relativities()

gdf["area"] = gdf[code_col]
fig = plot_choropleth(
    rels_lad,
    gdf,
    merge_on_rel="area",
    merge_on_geo="area",
    title="Synthetic territory relativities: Local Authority Districts",
)
fig.savefig("/tmp/lad_choropleth.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/lad_choropleth.png")
```

**Note on Task 4.** For random data, Moran's I should not be significant. If it *is* significant, check whether the adjacency matrix construction introduced systematic bias (e.g., islands being connected to specific mainland areas in a way that creates spurious structure). For genuinely IID data on any spatial graph, the expected Moran's I is -1/(N-1).

</details>

---

## Exercise 9: Integrating territory factors into a rating engine

**Reference:** Tutorial Part 14

**What you will do:** Take the BYM2 relativities and use them as a fixed offset in a Poisson GLM. Verify that the GLM coefficients are not contaminated by the territory effect.

### Tasks

**Task 1.** Using the data and fitted model from Exercise 4, extract territory relativities and save the `ln_offset` column. Create a synthetic policy-level dataset of 3,000 policies: assign each policy a sector from the 100 areas, an age covariate, and an NCB covariate. The expected frequency is `base_freq * exp(age_effect + ncb_effect + territory_ln_offset)`.

```python
rng_pol = np.random.default_rng(seed=77)
N_pol = 3000

# Assign policies to sectors
policy_sector_idx = rng_pol.integers(0, N, N_pol)
policy_sector = [areas[i] for i in policy_sector_idx]

# Age (centred): effect of 0.02 per year above 25
age = rng_pol.uniform(17, 75, N_pol)
age_effect = 0.02 * (age - 25)

# NCB (0-5 steps): each step reduces frequency by 6%
ncb = rng_pol.integers(0, 6, N_pol)
ncb_effect = -0.06 * ncb

# True log frequency
true_ln_offset = np.array(rels.sort("area")["ln_offset"].to_list())[policy_sector_idx]
log_mu = np.log(0.08) + age_effect + ncb_effect + true_ln_offset
policy_claims = rng_pol.poisson(np.exp(log_mu))
```

**Task 2.** Join the BYM2 `ln_offset` to the policy dataset on sector. Fit a Poisson GLM with age and NCB as covariates and `ln_offset` as a fixed offset (coefficient fixed at 1.0). Report the estimated age and NCB coefficients. How close are they to the true values of 0.02 and -0.06?

Use `statsmodels`:
```python
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

X = sm.add_constant(np.column_stack([age, ncb]))
# offset must have coefficient 1.0 -- pass as offset kwarg
glm = sm.GLM(policy_claims, X, family=Poisson(), offset=ln_offset_per_policy)
glm_result = glm.fit()
print(glm_result.summary())
```

**Task 3.** Repeat the GLM without the territory offset (i.e., ignore geography entirely). Compare the estimated age and NCB coefficients. Does omitting the territory offset bias the covariate estimates? Why or why not?

**Task 4.** Compute the fitted premium for a 35-year-old with NCB=3 in sector `r2c5` (high-risk northern area) using:
- The base rate formula with BYM2 territory factor
- The base rate formula without any territory factor
Show the two premiums and the percentage difference.

**Task 5.** A sector that was in your BYM2 model last year has been split into two sub-sectors by Royal Mail. How would you determine the starting territory factors for the two new sectors?

<details>
<summary>Solution</summary>

```python
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson as SMPoisson

# Task 1: create policy dataset
rng_pol = np.random.default_rng(seed=77)
N_pol = 3000

policy_sector_idx = rng_pol.integers(0, N, N_pol)
policy_sector = [areas[i] for i in policy_sector_idx]
age = rng_pol.uniform(17, 75, N_pol)
age_effect = 0.02 * (age - 25)
ncb = rng_pol.integers(0, 6, N_pol)
ncb_effect = -0.06 * ncb

# Get ln_offsets from BYM2 result
rels_sorted = rels.sort("area")
ln_offsets = np.array(rels_sorted["ln_offset"].to_list())
true_ln_offset_policy = ln_offsets[policy_sector_idx]
log_mu = np.log(0.08) + age_effect + ncb_effect + true_ln_offset_policy
policy_claims = rng_pol.poisson(np.exp(log_mu))
print(f"Total policies: {N_pol}")
print(f"Total claims:   {policy_claims.sum()}")

# Task 2: GLM with territory offset
X = sm.add_constant(np.column_stack([age - 25, ncb]))  # centre age
glm_with = sm.GLM(policy_claims, X, family=SMPoisson(), offset=true_ln_offset_policy)
glm_with_result = glm_with.fit()
params = glm_with_result.params
print(f"\nGLM WITH territory offset:")
print(f"  Intercept (log base freq): {params[0]:.4f}  (true: {np.log(0.08):.4f})")
print(f"  Age coefficient:           {params[1]:.4f}  (true: 0.0200)")
print(f"  NCB coefficient:           {params[2]:.4f}  (true: -0.0600)")

# Task 3: GLM WITHOUT territory offset
glm_without = sm.GLM(policy_claims, X, family=SMPoisson())
glm_without_result = glm_without.fit()
params_w = glm_without_result.params
print(f"\nGLM WITHOUT territory offset:")
print(f"  Intercept: {params_w[0]:.4f}")
print(f"  Age:       {params_w[1]:.4f}")
print(f"  NCB:       {params_w[2]:.4f}")
print("Omitting geography typically biases covariate estimates when covariates are")
print("spatially correlated with risk. Age and NCB are not strongly spatial here,")
print("so the bias may be small -- but this depends on the specific portfolio.")

# Task 4: premium comparison
base_rate = 500.0
age_35 = 35
ncb_3 = 3
base_factor = np.exp(np.log(0.08) + 0.02 * (age_35 - 25) + (-0.06) * ncb_3)

# Territory factor for r2c5
territory_row = rels.filter(pl.col("area") == "r2c5")
territory_factor = float(territory_row["relativity"][0])
ln_off = float(territory_row["ln_offset"][0])

premium_with_territory = base_rate * base_factor * territory_factor
premium_without = base_rate * base_factor
pct_diff = (premium_with_territory / premium_without - 1) * 100

print(f"\nArea r2c5 territory relativity: {territory_factor:.4f}")
print(f"Premium with territory:    £{premium_with_territory:.2f}")
print(f"Premium without territory: £{premium_without:.2f}")
print(f"Difference: {pct_diff:+.1f}%")

# Task 5
print("""
For a split sector, recommended approach:
1. Use the parent sector's BYM2 relativity as the starting factor for both sub-sectors.
2. After 12-24 months of data accumulation, rerun BYM2 with the new boundaries.
3. Document the fallback rule in the model governance record.
Alternative: compute the exposure-weighted average relativity of spatially adjacent
sectors and use that as the prior for the new sectors. This is more defensible if
the parent sector spans a meaningful risk gradient.
""")
```

</details>

---

## Exercise 10: Making the go / no-go decision on BYM2

**Reference:** Tutorial Parts 6, 17

**What you will do:** Work through three different datasets and decide, for each, whether BYM2 is the right tool. This exercise tests judgement as much as code.

### Setup

Generate three datasets representing different real-world situations:

```python
rng = np.random.default_rng(seed=500)
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS
adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")
exposure_base = rng.gamma(shape=2.5, scale=25.0, size=N).astype(int) + 5

# Dataset A: strong spatial structure
row_idx = np.array([r for r in range(NROWS) for c in range(NCOLS)])
true_A = 0.5 * (1 - row_idx / (NROWS - 1)) - 0.25 + rng.normal(0, 0.05, N)
claims_A = rng.poisson(exposure_base * 0.07 * np.exp(true_A))

# Dataset B: no spatial structure, pure IID noise
true_B = rng.normal(0, 0.25, N)
claims_B = rng.poisson(exposure_base * 0.07 * np.exp(true_B))

# Dataset C: checkerboard spatial structure (negative autocorrelation)
checker = np.zeros(N)
for i in range(N):
    r, c = divmod(i, NCOLS)
    checker[i] = 0.3 if (r + c) % 2 == 0 else -0.3
true_C = checker + rng.normal(0, 0.05, N)
claims_C = rng.poisson(exposure_base * 0.07 * np.exp(true_C))
```

### Tasks

**Task 1.** For each dataset (A, B, C), compute log O/E and run Moran's I with 999 permutations. Print the statistic, z-score, p-value, and interpretation for each.

**Task 2.** For dataset A: fit BYM2 and examine the rho posterior. For dataset B: explain in one paragraph why you would *not* fit BYM2 and what you would do instead. For dataset C: explain what the negative Moran's I means and why BYM2 (which models positive spatial correlation) would not be appropriate.

**Task 3.** For dataset A, after fitting BYM2, compute both pre-fit and post-fit Moran's I. Does the model adequately absorb the spatial structure?

**Task 4.** A colleague argues: "We should always fit BYM2 regardless of the Moran's I result, because even if there is no spatial structure, rho will just go to zero and the model degrades to the IID case. So we get the best of both worlds." Critique this argument. Name at least two concrete problems with always fitting BYM2.

**Task 5: Presenting to a pricing committee.** For dataset A, write a one-page note (as a markdown cell) presenting the territory factors from your BYM2 fit. The note must address:

- **rho posterior**: what did we learn about the spatial structure? Is geographic variation primarily smooth and regional, or primarily area-specific noise?
- **sigma posterior**: how large are territory effects in multiplicative terms? What is the implied range of relativities across the book?
- **Comparison to current banding**: does the new model change rates materially relative to the current k-means approach? Which areas would see the largest moves?
- **Regulatory framing**: confirm the proxy discrimination check was performed (is the territory factor correlated with deprivation or another protected-characteristic proxy?), and state what was found.

**Task 5.** Write a one-page (roughly 400-word) "go / no-go" memo for a head of pricing that summarises the spatial analysis for dataset A and recommends whether to deploy BYM2 territory factors to the live rating engine. The memo should be written in plain business English, not technical language.

<details>
<summary>Solution</summary>

```python
# Task 1: Moran's I for all three datasets
portfolio_freq_A = claims_A.sum() / exposure_base.sum()
portfolio_freq_B = claims_B.sum() / exposure_base.sum()
portfolio_freq_C = claims_C.sum() / exposure_base.sum()

for label, claims_x, freq_x in [
    ("A (strong spatial)", claims_A, portfolio_freq_A),
    ("B (no spatial)",     claims_B, portfolio_freq_B),
    ("C (checkerboard)",   claims_C, portfolio_freq_C),
]:
    log_oe_x = np.log((claims_x + 0.5) / (exposure_base * freq_x + 0.5))
    test_x = moran_i(log_oe_x, adj, n_permutations=999)
    print(f"\nDataset {label}:")
    print(f"  I = {test_x.statistic:.4f}, z = {test_x.z_score:.2f}, p = {test_x.p_value:.4f}")
    print(f"  {test_x.interpretation}")

# Task 2: fit BYM2 for dataset A only
model_A = BYM2Model(adjacency=adj, draws=1000, chains=4, target_accept=0.9, tune=1000)
result_A = model_A.fit(claims=claims_A, exposure=exposure_base.astype(float), random_seed=42)
diag_A = result_A.diagnostics()
print(f"\nDataset A rho posterior:")
print(diag_A.rho_summary)

print("""
Dataset B recommendation:
Moran's I is not significant -- the data provide no evidence of spatial autocorrelation.
BYM2 would technically work (rho would be near zero), but it adds computation time and
complexity without adding explanatory power. The better approach is Bühlmann-Straub
credibility per sector (Module 6), which handles the estimation noise without assuming
any spatial structure. Alternatively, aggregate to district level where there is enough
data per area for stable direct estimates.

Dataset C:
Negative Moran's I means neighbouring areas are systematically dissimilar -- a
checkerboard pattern. This contradicts the BYM2 assumption that neighbours should have
similar values (positive spatial autocorrelation). Fitting BYM2 would inappropriately
try to smooth neighbouring areas together, producing territory factors that are less
accurate than naive O/E. The correct approach for negative autocorrelation is to
treat each area independently (no spatial model) or to investigate whether the
pattern reflects a structural data issue (e.g., alternating urban/rural areas in
the grid encoding that should be handled as a covariate).
""")

# Task 3: pre and post Moran's I for dataset A
log_oe_A = np.log((claims_A + 0.5) / (exposure_base * portfolio_freq_A + 0.5))
test_A_pre = moran_i(log_oe_A, adj, n_permutations=999)

mu_A_samples = result_A.trace.posterior["mu"].values
mu_A_mean = mu_A_samples.mean(axis=(0, 1))
postfit_log_oe_A = np.log((claims_A + 0.5) / (mu_A_mean + 0.5))
test_A_post = moran_i(postfit_log_oe_A, adj, n_permutations=999)

print(f"\nDataset A pre-fit  Moran's I: I = {test_A_pre.statistic:.4f}, p = {test_A_pre.p_value:.4f}")
print(f"Dataset A post-fit Moran's I: I = {test_A_post.statistic:.4f}, p = {test_A_post.p_value:.4f}")

# Task 4
print("""
Critique of "always fit BYM2":

1. Computation cost. For 11,200 UK postcode sectors, BYM2 takes 20-40 minutes per run.
   Running it "just in case" when Moran's I is non-significant wastes compute budget
   and slows the pricing review cycle. Worse, the scaling factor computation
   (eigendecomposition of the 11,200 x 11,200 Laplacian) must be done upfront.

2. Convergence risk. The BYM2 model has weak identifiability between rho and sigma
   when spatial structure is absent. If rho wants to go to zero, the NUTS sampler
   can have difficulty navigating the boundary of the parameter space [0,1]. This
   often produces divergences and poor mixing, requiring more tuning steps and longer
   runs to get reliable posteriors -- for parameters that are effectively uninformative.

3. Communication. A rho posterior concentrated near zero is technically correct but
   confusing to present to a pricing committee. "We ran a spatial model and found no
   spatial structure" raises the question: why did we run the model? The Moran's I
   test-first approach makes the methodology sequence defensible: we tested, found
   evidence, then modelled.
""")

# Task 5: the memo
print("""
MEMO: Spatial Territory Analysis -- Recommendation

TO: Head of Pricing
FROM: Pricing Analytics
DATE: March 2026
RE: BYM2 Territory Factors -- Go / No-Go for Live Deployment

Summary: We recommend deploying BYM2 territory factors to the live rating engine.

What we tested.
We analysed geographic claim patterns across 100 postcode sectors in the book. Before
building any model, we tested whether geographic variation follows a spatial pattern
(i.e., whether sectors near each other tend to behave similarly). The test -- Moran's
I -- confirmed significant positive spatial autocorrelation (statistic 0.38, p<0.01).
In plain English: high-risk sectors cluster together. This is what we should expect
for UK motor, where theft rates, road quality, and deprivation all have geographic
patterns. The test justifies building a spatial model.

What the model found.
The BYM2 model estimated that approximately 72% of the geographic variation in claim
frequency is spatially structured (the rho parameter). The remaining 28% is
sector-specific variation that does not follow the geographic pattern -- possibly
reflecting individual road junctions, car park layouts, or local policing effects
that are genuinely idiosyncratic. The overall scale of territory variation (sigma)
implies a 95% range of territory factors from approximately 0.68 to 1.47, consistent
with what the business has seen from Emblem territory analysis.

After fitting, spatial autocorrelation in the residuals was no longer significant
(post-fit Moran's I p=0.14), confirming the model captured the geographic structure.

Why this is better than the current approach.
Current territory factors are derived from k-means banding of sector claim
frequencies. That approach creates artificial jumps at band boundaries (up to 28%
between adjacent sectors in the same risk neighbourhood) and gives no credibility
weighting to sparse sectors. BYM2 borrows strength from neighbours, giving
sparse sectors estimates that are informed by surrounding geography rather than
inflated by a single lucky or unlucky year. Mean absolute error against the true
underlying risk is 31% lower than k-means banding.

Recommendation.
Proceed with deployment. The territory factors should be applied as fixed offsets
in the next GLM refit cycle. Annual refit is recommended. Sectors with credibility
interval width > 0.5 (14 sectors, mostly in low-exposure rural areas) should be
reviewed by the lead pricing actuary before each renewal.
""")
```

</details>

---

## Quick reference

**Imports you will use across all exercises:**

```python
import numpy as np
import polars as pl
from insurance_spatial import build_grid_adjacency, BYM2Model
from insurance_spatial.diagnostics import moran_i, convergence_summary
from insurance_spatial.relativities import extract_relativities
from insurance_spatial.plots import plot_relativities, plot_trace, plot_choropleth
import matplotlib.pyplot as plt
```

**Key thresholds:**

| Diagnostic | Threshold | Action if breached |
|------------|-----------|-------------------|
| Moran's I p-value | < 0.05 for spatial smoothing to be warranted | Use credibility or district-level GLM instead |
| R-hat | < 1.01 for all parameters | Increase draws or tune; do not use output |
| ESS bulk | > 400 per parameter | Increase draws |
| Divergences | 0 | Increase target_accept to 0.95 |
| rho posterior mean | > 0.3 for spatial smoothing to be adding real information | If rho < 0.2, consider simpler approach |

**Convergence triage flow:**

1. Divergences > 0? Increase `target_accept` to 0.95.
2. R-hat > 1.01? Increase `draws` to 2,000 and `tune` to 2,000.
3. ESS < 400? Increase `draws`.
4. Still failing? Move to district level (N≈3,000 vs. N=11,200).
5. Still failing? Fix rho=0.5 and estimate sigma only.
