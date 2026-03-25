## Part 9: Reading the NID scores

```python
# Raw NID table (before GLM testing)
nid_table = detector.nid_table()
print("Top 10 NID candidates:")
print(nid_table.head(10))
```

The NID table contains:

| Column | Meaning |
|---|---|
| `feature_1`, `feature_2` | The candidate pair |
| `nid_score` | Raw NID score (unnormalised) |
| `nid_score_normalised` | Score rescaled to [0, 1] — 1.0 = highest-ranked pair |

```python
# Plot the NID scores
top_n = 15
top_nid = nid_table.head(top_n)
labels = [f"{r['feature_1']} × {r['feature_2']}" for r in top_nid.iter_rows(named=True)]

plt.figure(figsize=(10, 5))
plt.barh(range(top_n), top_nid["nid_score_normalised"].to_list(), color="#2271b3")
plt.yticks(range(top_n), labels)
plt.xlabel("NID score (normalised)")
plt.title("Top interaction candidates from NID")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**What to look for:** The pair `age_band × vehicle_group` should rank first or second. The second planted interaction — `ncd_years × has_convictions` (representing the extra risk for convicted drivers with substantial NCD) — should also appear in the top 5. If neither appears in the top 10, the CANN training may not have converged — try increasing `cann_n_epochs` to 500 or reducing the learning rate to `5e-4`.

### Understanding the NID scores in context

The NID scores are relative, not absolute. A pair with `nid_score_normalised = 1.0` is the strongest interaction detected by the CANN. A pair with `nid_score_normalised = 0.2` has one-fifth the detected signal. But "strongest detected" does not mean "statistically significant" — some pairs with high NID scores will fail the LR test because the CANN found a pattern that does not survive GLM testing.

The NID step is a fast screening tool. We then take the top 15 candidates to the slower, more rigorous LR test step.
