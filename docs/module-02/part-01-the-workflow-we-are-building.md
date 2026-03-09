## Part 1: The workflow we are building

We are building a motor frequency-severity model: a Poisson GLM for claim frequency and a Gamma GLM for average severity, both with log link and exposure offset. The pure premium estimate is the product: frequency times severity.

The data pipeline, in order:

1. Generate a synthetic UK motor dataset with known true parameters
2. Prepare features: encode categorical factors, handle base levels, check for data quality issues
3. Fit the frequency GLM (Poisson with log link and exposure offset)
4. Fit the severity GLM (Gamma with log link, on claimed policies only)
5. Run diagnostics: deviance residuals, actual-versus-expected by factor level
6. Validate against known true parameters (and later, against Emblem output)
7. Export factor tables in the format Radar expects

We use synthetic data throughout because it has known true parameters. This lets us verify that our GLM is recovering the right answers - if the model works correctly on synthetic data, we can trust it on real data.