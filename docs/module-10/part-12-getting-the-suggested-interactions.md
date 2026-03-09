## Part 12: Getting the suggested interactions

```python
# Simple API to get the recommended pairs
suggested = detector.suggest_interactions(top_k=3)
print("Suggested interactions:", suggested)
```

`suggest_interactions()` returns only the pairs where `recommended == True`, sorted by consensus score. The consensus score combines NID rank and (if SHAP validation was run) SHAP rank — it is a composite ranking that tends to be more stable than either method alone.

You can also ask for the top-K regardless of whether they are statistically significant:

```python
# Top 3 by NID score regardless of significance
top3_nid = detector.suggest_interactions(top_k=3, require_significant=False)
print("Top 3 by NID score (significance not required):", top3_nid)
```

For production use, always use the default `require_significant=True`. For exploratory analysis, `require_significant=False` helps you understand which interactions the CANN found most strongly, even if they did not survive GLM testing on this sample size.