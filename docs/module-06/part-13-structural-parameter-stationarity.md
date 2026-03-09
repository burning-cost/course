## Part 13: Structural parameter stationarity

The EPV (v) and VHM (a) are estimated from historical data. Which historical data matters.

The 2020-2022 period contained the COVID shock: claim frequencies fell sharply in 2020 (reduced driving), rebounded unevenly in 2021-2022, and claim costs inflated due to supply chain disruption on parts and courtesy cars. Estimating v from that period inflates the EPV, because some of the within-group year-to-year variance is a portfolio-wide shock rather than genuine group-level volatility.

Inflated v → inflated K → Z values too low → more shrinkage than warranted.

In practice:
- Re-estimate structural parameters at each model rebuild cycle, not just when you rebuild the main pricing model
- Consider excluding shock years (or downweighting them) when estimating v and a, if the portfolio experienced a clear external distortion
- Monitor K stability over time: a sudden increase in K without a change in portfolio composition is a signal that the EPV estimate has been contaminated by a portfolio-wide event

For the v estimate specifically: if your within-group variance in 2019-2023 is substantially higher than 2015-2019, investigate whether the difference is genuine geographic volatility or a COVID-era artefact before using the inflated K in production.