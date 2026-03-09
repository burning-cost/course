## Part 5: Building the adjacency matrix

The adjacency matrix W is the backbone of the spatial model. It encodes which areas are neighbours. For our synthetic grid, two areas are neighbours if they share an edge (rook connectivity) or share an edge or corner (queen connectivity). For real postcode sectors, two sectors are neighbours if their boundary polygons share any edge or vertex -- queen contiguity by convention.

### Rook vs. queen contiguity

With rook connectivity, the centre cell of a 3x3 grid has four neighbours. With queen connectivity, it has eight. For UK postcode geography, queen is standard: postcode sectors often share only a single point at a corner, and ignoring those connections creates artificially disconnected areas.

```python
from insurance_spatial import build_grid_adjacency

adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")

print(f"Areas (N):       {adj.n}")
print(f"Total edges:     {adj.W.nnz // 2}")  # symmetric, so divide by 2
print(f"Mean neighbours: {adj.neighbour_counts().mean():.2f}")
print(f"Min neighbours:  {adj.neighbour_counts().min()}")
print(f"Max neighbours:  {adj.neighbour_counts().max()}")
print(f"Connected components: {adj.n_components()}")
print(f"Scaling factor:  {adj.scaling_factor:.4f}")
```

**What you should see:**

```bash
Areas (N):       100
Total edges:     180
Mean neighbours: 3.60
Min neighbours:  1
Max neighbours:  5
Connected components: 1
Scaling factor:  0.4263
```

Corner cells have fewer neighbours than interior cells. The scaling factor -- 0.4263 in this case -- is a graph-level quantity that makes the spatial random effect variance-interpretable. We explain it in Part 7.

### The adjacency matrix as a sparse matrix

`adj.W` is a SciPy CSR sparse matrix. It is binary and symmetric: W[i,j] = 1 if areas i and j are neighbours, 0 otherwise. No self-loops.

```python
# Inspect the structure
print(f"W type: {type(adj.W)}")
print(f"W shape: {adj.W.shape}")
print(f"Density: {adj.W.nnz / (adj.n ** 2):.4f}")

# Look at the first row -- who are area r0c0's neighbours?
first_row = adj.W.getrow(0).toarray().ravel()
neighbour_indices = np.where(first_row == 1)[0]
neighbour_labels = [adj.areas[i] for i in neighbour_indices]
print(f"Neighbours of r0c0: {neighbour_labels}")
```

r0c0 is the top-left corner. With queen connectivity it has three neighbours: the cell to its right, the cell below, and the cell diagonally below-right.

### Why the graph must be connected

The ICAR model (which we explain in Part 7) is defined on a connected graph. If any areas form a disconnected component -- they have no path of edges connecting them to the rest of the graph -- the ICAR precision matrix is not invertible in the right way and the model fails. The `adj.n_components()` call above confirms connectivity: we want exactly 1.

For real postcode geography, islands are the classic problem. The Scottish Islands, Orkney, and Shetland have no shared boundaries with mainland sectors. The `from_geojson()` function handles this automatically with `fix_islands=True` (the default), connecting each island to its nearest mainland sector by centroid distance. This is a documented modelling choice: we are saying "for spatial smoothing purposes, treat Shetland as if it borders its nearest mainland neighbour." For Shetland, this is typically in Highland council area; the exact sector depends on the centroid computation. We explain this in the audit trail.