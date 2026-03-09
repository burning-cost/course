## Part 7: The BYM2 model -- what it is and why

### The components

BYM2 (Riebler et al., 2016) is a Bayesian hierarchical model for areal data. "Areal" means that the data are associated with geographic areas (postcode sectors, districts, counties) rather than individual GPS coordinates.

The model has two spatial components that are combined into a single territory effect:

**The ICAR component (phi):** This is the spatially structured part. ICAR stands for Intrinsic Conditional Auto-Regressive. It encodes the prior belief that neighbouring areas have similar values. The ICAR distribution for area i says: the expected value of phi[i], given all its neighbours, is the average of the neighbours' phi values. Areas that are isolated from high-phi areas will be pulled towards zero; areas surrounded by high-phi areas will be pulled upwards.

**The IID component (theta):** This is the unstructured part -- an independent Normal(0,1) effect for each area. It captures area-specific variation that does not follow the spatial pattern. A sector with genuinely anomalous risk (a major new road junction, a large car park that attracts opportunistic theft) might have a high theta even though its neighbours are unremarkable.

The full territory effect for area i is:

```
b_i = sigma * (sqrt(rho / s) * phi_i + sqrt(1 - rho) * theta_i)
```

**sigma** is the overall scale of geographic variation. A sigma of 0.3 means territory effects have a standard deviation of about 0.3 on the log scale -- roughly a 30% spread around the mean.

**rho** is the mixing parameter. It controls how much of the total variance is allocated to the spatially structured component versus the IID component:
- rho = 1: all geographic variation is spatially smooth
- rho = 0: all geographic variation is area-specific noise, no spatial pattern
- rho = 0.7: 70% of the territory variance is spatially structured

**s** is the BYM2 scaling factor. This is a technical correction for the ICAR precision matrix. The ICAR distribution does not have a natural unit variance -- its scale depends on the graph topology. The scaling factor normalises phi to unit marginal variance, which means rho and sigma are interpretable regardless of whether you have a 10x10 grid or an 11,200-sector postcode map.

### Why rho is the most important output

After fitting, the posterior distribution of rho tells you directly whether spatial smoothing was useful. If the posterior of rho is concentrated near 1.0, the data support strong spatial structure and your territory factors are doing real work. If rho is concentrated near 0.0, the data say that geographic variation is essentially random noise -- BYM2 smoothing is not adding information.

A rho posterior mean of 0.3 with a wide credibility interval from 0.05 to 0.65 tells a different story from a posterior concentrated at 0.85 to 0.95. In the first case you might report: "We find weak and uncertain evidence of spatial structure; territory factors have wide credibility intervals and should be applied with caution." In the second: "Strong, consistent spatial structure is present; BYM2 territory factors are robustly estimated."

### The full model specification

```
y_i ~ Poisson(mu_i)
log(mu_i) = log(E_i) + alpha + b_i

b_i = sigma * (sqrt(rho / s) * phi_i + sqrt(1 - rho) * theta_i)

phi   ~ ICAR(W)           # structured; prior encodes neighbour similarity
theta ~ Normal(0, 1)      # unstructured; independent per area
sigma ~ HalfNormal(1)     # total territory scale
rho   ~ Beta(0.5, 0.5)    # proportion due to spatial structure; Jeffrey's prior
alpha ~ Normal(0, 1)      # intercept (overall log-rate)
```

E_i is exposure (policy-years). The log(E_i) term is a fixed offset -- it is not estimated, just added to the linear predictor to convert from log-rate to log-expected-claims.

The Beta(0.5, 0.5) prior on rho is Jeffrey's prior: it is flat on the logit scale and gives equal prior probability to "entirely spatial" and "entirely IID". It does not push the model towards either extreme.

### Two-stage vs. integrated

There are two ways to use this model in production.

**Integrated:** Pass raw area-level claims and exposure. The alpha parameter absorbs the overall rate, and b captures all geographic variation.

**Two-stage (recommended):** Fit a non-spatial GLM or GBM first. Compute sector-level observed vs. expected claims from the base model. Pass those sector-level O/E counts to BYM2. This way:

1. The spatial model is decoupled from the main risk model. You can update them independently.
2. The territory factor is auditable separately. A regulator can scrutinise the spatial smoothing methodology without also having to audit the full GBM.
3. The rho parameter has a cleaner interpretation: it tells you how much of the *residual* geographic variation (after removing all non-spatial risk factors) is spatially structured.

We use the integrated approach in this tutorial because we do not have a fitted base model. Part 13 discusses how to do the two-stage pipeline in practice.