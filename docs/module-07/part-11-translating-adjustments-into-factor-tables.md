## Part 11: Translating adjustments into factor tables

The factor adjustment multipliers apply uniformly to every level of each factor table. This section shows how to produce the updated tables and what to include in the pricing committee pack.

```python
# Current factor tables (Polars DataFrames)
# In production, these come from your rating system's data store
current_tables = {
    "f_age": pl.DataFrame({
        "band":       ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "relativity": [2.00, 1.50, 1.20, 1.00, 0.92, 0.95, 1.10],
    }),
    "f_ncb": pl.DataFrame({
        "ncd_years":  [0, 1, 2, 3, 4, 5],
        "relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
    "f_vehicle": pl.DataFrame({
        "group":      ["Standard", "Performance", "High-perf", "Prestige"],
        "relativity": [0.90, 1.00, 1.10, 1.30],
    }),
    "f_region": pl.DataFrame({
        "region":     ["Rural", "National", "Urban", "London"],
        "relativity": [0.85, 1.00, 1.10, 1.20],
    }),
    "f_tenure_discount": pl.DataFrame({
        "tenure_years": list(range(10)),
        "relativity":   [1.00] * 10,
    }),
}

factor_adj = result.factor_adjustments

# Apply adjustments and produce updated tables (all in Polars)
updated_tables = {}
for fname, tbl in current_tables.items():
    m = factor_adj.get(fname, 1.0)
    updated = tbl.with_columns([
        (pl.col("relativity") * m).alias("new_relativity"),
        ((m - 1) * 100 * pl.lit(1.0)).alias("pct_change"),
    ]).rename({"relativity": "current_relativity"})
    updated_tables[fname] = updated
    print(f"\n{fname}  (adjustment {m:.4f} = {(m-1)*100:+.1f}%):")
    print(updated)
```

**What you should see for f\_age:**

```sql
f_age  (adjustment 1.0368 = +3.7%):

band    current_relativity  new_relativity  pct_change
17-21                 2.00          2.0736         3.7
22-24                 1.50          1.5552         3.7
25-29                 1.20          1.2442         3.7
30-39                 1.00          1.0368         3.7
40-54                 0.92          0.9539         3.7
55-69                 0.95          0.9850         3.7
70+                   1.10          1.1405         3.7
```

All rows show the same percentage change. This is correct and expected for a uniform factor adjustment: the shape of the table is preserved; only the scale changes.

**What this means for customers.** A 19-year-old in the 17-21 band has a current factor of 2.00. After the rate action, it is 2.074. Their premium increases by 3.7%, same as a 45-year-old in the 40-54 band. In absolute terms, the 19-year-old's premium increases by more (because they start from a higher base), but the percentage change is identical across every age band.

This is the correct reading of a uniform rate action, and it matters for the cross-subsidy analysis in the next part.