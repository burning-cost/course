## Part 13: Protected characteristics and proxy discrimination

Before exporting anything, we need to address a regulatory requirement.

The FCA's Consumer Duty and the Equality Act 2010 place constraints on UK personal lines rating factors. Driver age is currently permitted as a motor rating factor where actuarially justified. But the more important question for any GBM is whether features that are not protected characteristics are acting as proxies for protected characteristics.

SHAP gives you a quantitative way to check this. The general method is to compute the correlation between one feature's SHAP values and another variable you are concerned about. In a real portfolio, you would check whether a rating factor's SHAP values correlate with a protected characteristic (such as an IMD deprivation score as a proxy for ethnicity, or a telematics speed metric as a proxy for disability). The example below demonstrates the method by checking area SHAP values against a synthetic deprivation proxy — the structure of the check is identical to what you would run against a real protected characteristic.

In a new cell, type this and run it (Shift+Enter):

```python
# Example: check if area SHAP values correlate with simulated deprivation index
# In a real portfolio you would use actual deprivation scores or demographic data
# Here we create a synthetic deprivation proxy to demonstrate the method

rng_check = np.random.default_rng(99)
area_deprivation = {"A": 0.3, "B": 0.4, "C": 0.5, "D": 0.6, "E": 0.7, "F": 0.8}
dep_score = np.array([area_deprivation[a] for a in df_pd["area"]]) + rng_check.normal(0, 0.1, len(df_pd))

area_idx  = sr.feature_names_.index("area")
area_shap = sr.shap_values()[:, area_idx]

correlation = np.corrcoef(area_shap, dep_score)[0, 1]
print(f"Correlation between area SHAP values and deprivation proxy: {correlation:.3f}")

if abs(correlation) > 0.5:
    print("Correlation above 0.5 - review with compliance before using these relativities.")
    print("This threshold is illustrative: agree your own threshold with your compliance function.")
else:
    print("Correlation is modest. Document this check in your model governance file.")
```

You will see a correlation of around 0.55-0.65. In this synthetic example, area and deprivation are by construction correlated - we built the data that way. In a real portfolio, you would use actual deprivation indices (e.g. the Index of Multiple Deprivation) matched to postcode.

**What to do with this result:** A high correlation does not automatically disqualify the feature. Area is a legitimate risk factor in UK motor (urban areas have higher claim frequency due to traffic density, not due to poverty). But you need to document the check, document the justification, and be prepared to discuss it with the FCA if asked. The Consumer Duty requires evidence that your pricing is fair; SHAP gives you the evidence layer.

The 0.5 threshold in the code above is illustrative. There is no industry-standard cutoff. Agree your threshold with your compliance function before running this check in production — what counts as "high" depends on the feature, the protected characteristic being proxied, and your firm's risk appetite.