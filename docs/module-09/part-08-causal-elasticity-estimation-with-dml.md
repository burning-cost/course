## Part 8: Group average treatment effects

The portfolio ATE tells you the average elasticity across all customers. The commercial question is more specific: which segments are elastic, and by how much? An NCD-5 direct customer and an NCD-0 PCW customer have very different price sensitivities. Pricing them using the same elasticity is leaving money on the table.

`RenewalElasticityEstimator.gate()` computes Group Average Treatment Effects: the average CATE within a segment, with confidence intervals.

### GATEs by NCD band

```python
%md
## Part 8: Group average treatment effects
```

```python
gate_ncd = est.gate(df, by="ncd_years")
print("GATE by NCD band:")
print(gate_ncd)
```

Compare against ground truth:

```python
# Side-by-side comparison: estimated vs. true
true_ncd = true_gate_by_ncd(df)

comparison_ncd = gate_ncd.join(
    true_ncd.rename({"true_elasticity_mean": "true_ate"}),
    on="ncd_years",
    how="left",
).with_columns(
    (pl.col("elasticity") - pl.col("true_ate")).alias("bias")
)
print("\nEstimated vs. true by NCD band:")
print(comparison_ncd.select(["ncd_years", "elasticity", "ci_lower", "ci_upper",
                              "true_ate", "bias", "n"]))
```

On 50,000 observations with sufficient price variation, the estimated GATEs should lie within their 95% CIs of the true values for all NCD bands. The NCD gradient (most elastic at NCD=0, least at NCD=5) should be clear and significant: the CI for NCD=0 should not overlap with the CI for NCD=5.

### GATEs by age band

```python
gate_age = est.gate(df, by="age_band")
print("\nGATE by age band:")
print(gate_age)
```

```python
true_age = true_gate_by_age(df)

comparison_age = gate_age.join(
    true_age.rename({"true_elasticity_mean": "true_ate"}),
    on="age_band",
    how="left",
)
print("\nEstimated vs. true by age band:")
print(comparison_age.select(["age_band", "elasticity", "ci_lower", "ci_upper", "true_ate"]))
```

### GATEs by channel

```python
gate_channel = est.gate(df, by="channel")
print("\nGATE by channel:")
print(gate_channel)
```

PCW customers should be substantially more elastic (more negative elasticity) than direct or broker customers. In the DGP, PCW customers have 30% higher elasticity than the same customer through direct or broker. The CausalForestDML should recover this.

### What these GATEs tell a pricing actuary

These segment elasticities are the input to differentiated pricing strategy:

- **NCD-0, PCW channel** (most elastic): price increases will drive high lapse. If ENBP permits headroom, aggressive retention discounts may be justified.
- **NCD-5, direct channel** (least elastic): these customers are unlikely to leave for moderate price increases. The ENBP constraint, not elasticity, is typically the binding constraint for this group.

The GATE table does not tell you what price to set — that is the optimiser's job in Part 12. It tells you the relative sensitivity of each segment, which determines how the optimiser balances across the book.
