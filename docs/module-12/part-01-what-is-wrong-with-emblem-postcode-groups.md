## Part 1: What is wrong with Emblem postcode groups

### The standard workflow

You are pricing UK motor. Your data has postcode sector for each policy: SW1A 1, SW1A 2, E1 1, and so on. There are roughly 11,200 postcode sectors in Great Britain. They vary enormously in size: EC2Y 9 covers a few streets in the City of London; IV27 4 covers several hundred square kilometres in Sutherland.

The standard Emblem workflow goes like this:

1. Aggregate claims and exposure to postcode sector level
2. Compute observed claim frequency per sector
3. Run k-means clustering (or geographic smoothing in Emblem's "spatial" module) to group sectors into 8--15 territory bands
4. Use band as a categorical rating factor, one relativity per band
5. Call it done

This is widely practised. The CII pricing exam tests it. Vendor documentation endorses it. And it has three structural problems that compound each other.

### Problem 1: Sharp boundaries where no sharp boundary exists

Risk does not care about territory band boundaries. The underlying drivers of claims frequency -- road network density, vehicle theft rates, weather patterns, deprivation -- vary continuously across space. A postcode sector in band 6 that borders a sector in band 4 does not suddenly become a different level of risk at the boundary. But the rating model treats it that way.

In practice, the gap between adjacent bands can be 20--35%. A customer in sector SL3 7 (Slough, band 6) might pay 28% more than a customer in sector SL3 8 (also Slough, band 4) for the same vehicle and driver profile. The sectors are geographically adjacent. The underlying risk difference, once sampling noise is accounted for, might be 6--8%.

That premium gap is not wrong because territory is unimportant. It is wrong because the banding process converts a continuous spatial process into a piecewise constant approximation and adds its own discretisation error on top of the underlying estimation noise.

### Problem 2: Thin data, no borrowing

A typical UK motor book has 50,000--150,000 policies in force spread across 11,200 postcode sectors. That averages out to 4--13 policies per sector per year. Most sectors have zero claims in any given year. Many sectors with the highest apparent claim frequencies have only one or two claims -- estimates with coefficients of variation exceeding 100%.

k-means clustering the sector claim frequencies does not solve this problem. It averages noisy estimates into noisy bands. A sector with two claims and two policy-years has a 100% observed frequency. It drags its band upwards. A neighbouring sector with zero claims and four policy-years has a 0% observed frequency. It drags its band downwards. Neither estimate is meaningful, and combining them does not produce a meaningful band estimate.

What you actually want is to borrow strength: let sectors with sparse data learn from their neighbours. This is exactly what spatial Bayesian models do. The Emblem approach has no mechanism for it.

### Problem 3: The methodology is opaque

When the FCA asks how your territory factors are derived, the answer "postcode sectors were clustered by historical claim frequency using k-means with k=10" raises follow-up questions. Why k=10? How were ties broken? Which years of data? What smoothing was applied before clustering? The answers are usually "judgment," "default settings," and "two or three years."

The BYM2 model has documented prior distributions, published theoretical justification (Riebler et al., 2016, *Statistical Methods in Medical Research*), and a single explicit spatial smoothing parameter rho that quantifies how much the data support the spatial structure. You can reproduce every step from inputs to outputs with a fixed random seed. That is what a defensible methodology looks like.