## Part 15: Real UK postcode geography

Everything above has used a synthetic grid. For real UK territory rating, you need actual postcode sector boundaries.

### Data sources

**Postcode sector boundaries.** ONS does not publish postcode sector boundaries directly. The recommended source is to derive them from OS CodePoint Open (free for commercial use under OS OpenData licence). CodePoint Open gives you a point location for every postcode unit in Great Britain. You Voronoi-tessellate these points to get approximate sector boundaries. The result is not identical to Royal Mail's proprietary sector polygons but is accurate enough for adjacency construction.

Alternatively, the UK Data Service and some actuarial data vendors (e.g., Barbican) publish pre-built sector boundary files. Check whether your organisation has an existing licence.

**ONSPD (Office for National Statistics Postcode Directory).** This maps every UK postcode unit to its sector, district, LSOA, MSOA, and local authority. It is updated quarterly and available free from the ONS Open Geography Portal.

### Loading real boundaries

```python
from insurance_spatial.adjacency import from_geojson

# With a real boundary file
adj_real = from_geojson(
    "postcode_sectors_england_wales.geojson",
    area_col="PC_SECTOR",          # column name in your file
    connectivity="queen",           # standard for UK postcode geography
    fix_islands=True,               # connect Scilly Isles, IoW, etc.
)

print(f"Sectors loaded:  {adj_real.n}")
print(f"Components:      {adj_real.n_components()}  (want 1)")
print(f"Mean neighbours: {adj_real.neighbour_counts().mean():.2f}")
print(f"Scaling factor:  {adj_real.scaling_factor:.4f}")
```

**Computing the scaling factor for 11,200 sectors.** The scaling factor computation requires eigenvalues of the N x N Laplacian matrix. The Laplacian of the postcode sector adjacency graph is sparse -- UK postcode sectors have on average 5--6 neighbours each, giving roughly 67,000 non-zero entries in an 11,200 x 11,200 matrix. A sparse eigensolver handles this in 30--120 seconds without forming the dense matrix. Run it once, cache the result, and pass it directly in subsequent runs:

```python
# First run: compute and cache
scaling = adj_real.scaling_factor  # triggers computation
np.save("sector_scaling_factor.npy", np.array([scaling]))

# Subsequent runs: load from cache
from insurance_spatial.adjacency import AdjacencyMatrix
scaling_cached = float(np.load("sector_scaling_factor.npy")[0])
adj_real_cached = AdjacencyMatrix(
    W=adj_real.W,
    areas=adj_real.areas,
    _scaling_factor=scaling_cached,
)
```

### Computation time expectations

For N=11,200 postcode sectors with 4 chains of 1,000 draws on a 4-core Databricks cluster:

- Scaling factor: 30--120 seconds (one-off, uses sparse eigensolver)
- Sampling: 20--40 minutes total
- Diagnostics and relativity extraction: 2--5 minutes

Use nutpie for a 2--5x speedup on the sampling phase:

```python
%pip install nutpie --quiet
dbutils.library.restartPython()
```

After restart, nutpie is detected automatically by `insurance-spatial`. The sampler selection is silent if nutpie is found and warns if it is not. You do not need to change any other code.

For district-level analysis (N≈3,000 UK postcode districts), the full pipeline runs in under 15 minutes without nutpie.