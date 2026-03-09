## Part 1: Why GLMs miss interactions

Before touching any code, we need to be precise about what a GLM assumes and where that assumption breaks down.

### The multiplicative GLM

A standard Poisson frequency GLM for UK motor insurance predicts:

```
μ_i = exposure_i × exp(α + β_area × area_i + β_vg × vg_i + β_age × age_i + ...)
```

This is a multiplicative model. The total risk for policy `i` is the product of the base rate with a factor for each rating variable. Young driver gets multiplied by 1.8. High vehicle group gets multiplied by 1.4. Young driver in a high vehicle group gets multiplied by 1.8 × 1.4 = 2.52.

That is what the GLM predicts. The actual multiplier for that combination — if it were a genuinely supermultiplicative interaction — might be 3.1. The GLM is undercharging that combination by 19%.

### How interactions enter the data

An interaction term in a GLM looks like this:

```
μ_i = exposure_i × exp(... + β_age_vg × [age_band_i × vehicle_group_i] + ...)
```

For a categorical-by-categorical interaction with `L_age` age bands and `L_vg` vehicle groups, this adds `(L_age - 1) × (L_vg - 1)` parameters to the model. A 6-band age variable with a 50-group vehicle group variable adds 245 parameters. The GLM can represent the interaction exactly if the data supports it — but the parameter cost is high, and the GLM never learns the interaction unless you tell it to look for it.

This is the fundamental problem: a GLM will never spontaneously discover that age × vehicle group is important. You must add the interaction term explicitly. And to add it explicitly, you must first know to look for it.

### Why manual 2D A/E plots miss interactions

The standard initial approach is to produce 2D actual-versus-expected plots: for each pair of factors (i, j), split the data by (i, j) cells, compute the ratio of observed claims to GLM-predicted claims within each cell, and look for systematic off-diagonal patterns.

This works for detecting the obvious interactions — young driver × high vehicle group produces a very clear pattern where the top-right corner of the 2D plot (high age risk, high vehicle group) has A/E ratios well above 100%.

It fails in two ways. First, it requires you to check all `n(n-1)/2` pairs. With 12 rating factors, that is 66 plots. Actuaries selectively check the pairs they expect to be interesting. The unexpected ones are missed by construction.

Second, some interactions only manifest after controlling for other factors. An interaction between `driver_age` and `annual_mileage` might be invisible in the raw 2D plot because high-mileage young drivers cluster in urban areas (high `area` factor), and the area effect drowns out the age-mileage signal. The interaction is real but the marginal 2D plot does not show it.

The CANN + NID pipeline addresses both failures: it checks all pairs simultaneously, and it learns the interactions from the GLM's residuals (which already control for the main effects).