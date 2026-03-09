## Part 11: The Tweedie model (alternative approach)

The frequency-severity split (Poisson frequency times Gamma severity) is the standard UK personal lines approach. An alternative is the Tweedie pure premium model, which models the total incurred amount directly using a compound Poisson-Gamma distribution.

The Tweedie fits one model, not two. Its power parameter `p` controls the compound distribution: `p=1` is Poisson, `p=2` is Gamma, and `1 < p < 2` is the compound Poisson-Gamma relevant for insurance.

```python
# Compute pure premium in Polars before converting to pandas
df_pp = df.with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
)
df_pp_pd = df_pp.to_pandas()
df_pp_pd["log_exposure"] = np.log(df_pp_pd["exposure"].clip(lower=1e-6))

pp_formula = (
    "pure_premium ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_tweedie = smf.glm(
    formula=pp_formula,
    data=df_pp_pd,
    family=sm.families.Tweedie(
        var_power=1.5,
        link=sm.families.links.Log(),
    ),
    offset=df_pp_pd["log_exposure"],
).fit()

print(f"Tweedie GLM deviance: {glm_tweedie.deviance:,.1f}")
print(f"Pseudo R2: {1 - glm_tweedie.deviance/glm_tweedie.null_deviance:.4f}")
```

**When to use Tweedie vs frequency-severity split:**

**Tweedie** is simpler - one model, not two - and handles the zero-inflated pure premium directly. It is appropriate when you only want a pure premium prediction and are not trying to understand whether a risk is high frequency or high severity.

**Frequency-severity split** gives you more diagnostic power. You can see whether your area F uplift is driven by frequency (more accidents) or severity (more expensive accidents). That distinction matters: high-frequency/low-severity areas have a different risk management profile from low-frequency/high-severity areas, and reinsurance structuring is designed around that distinction.

The Tweedie also requires you to commit to a value of `p`. When the regulator asks why you chose `p=1.5`, you need an answer. The frequency-severity split has a cleaner actuarial justification: claims arrive as a Poisson process and each claim has a Gamma-distributed cost.

We recommend the frequency-severity split for production personal lines pricing.