# Module 12: Spatial Territory Rating

This is the final module. Every module to this point has built a piece of a production pricing system: data preparation, GLM fitting, gradient boosting, credibility weighting, rate optimisation. Territory has been a background character throughout -- a regional factor column, a categorical variable in Emblem, a postcode sector dropped into the model and forgotten.

Territory deserves better treatment than that. Based on published pricing model benchmarks, geography typically accounts for 15--25% of the explainable variation in claims frequency in UK personal lines books, though the figure varies substantially by class and portfolio composition. It is also the dimension where standard approaches fail most quietly. The Emblem postcode group approach produces numbers. The numbers look plausible. But the methodology has structural problems that spatial statistics resolved in the epidemiology literature three decades ago.

This module applies those solutions to insurance. By the end, you will have built a BYM2 spatial smoothing model on Databricks, extracted territory relativities with proper uncertainty estimates, and understood how to slot those relativities into a production rating engine as a fixed offset. You will also know when not to use the approach -- because the posterior will tell you.

By the end of this module you will have:

- Diagnosed the structural problems with Emblem-style postcode group rating
- Built and inspected an adjacency matrix from UK geography
- Run Moran's I to confirm spatial structure before fitting anything
- Understood the BYM2 model (ICAR + IID components) well enough to explain it to a non-statistician
- Fitted BYM2 via PyMC 5 on Databricks and interpreted MCMC diagnostics
- Extracted territory relativities with 95% credibility intervals
- Compared smoothed factors to Emblem bands and understood the differences
- Built a choropleth map of territory risk
- Integrated territory factors into a downstream GLM as a log-offset
- Made an informed decision about the two-stage vs. integrated pipeline
[Download the notebook for this module](notebook.py)

---
