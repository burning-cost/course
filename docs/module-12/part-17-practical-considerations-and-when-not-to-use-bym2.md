## Part 17: Practical considerations and when not to use BYM2

### Prior sensitivity

The default priors in the BYM2 model are weakly informative:

- `sigma ~ HalfNormal(1)`: this allows territory SDs up to about 2 on the log scale, corresponding to relativities from roughly 0.13 to 7.4. For UK personal lines, the actual range is much narrower -- roughly 0.6 to 1.6. So this prior is effectively uninformative.
- `rho ~ Beta(0.5, 0.5)`: Jeffreys' prior, symmetric around 0.5.

We recommend checking prior sensitivity by fitting with `sigma ~ HalfNormal(0.5)` (more regularisation) and comparing the posterior of sigma and rho. If the posteriors are similar, the data are informative and the prior does not matter much. If they shift substantially, the data are sparse and you should use the more regularising prior.

```python
# Check: does the prior on sigma matter?
# You would need to modify the model or implement this as a custom PyMC model
# For most real UK datasets with > 500 claims spread across 200+ sectors,
# HalfNormal(1) and HalfNormal(0.5) give similar posteriors.
print("Prior sensitivity check: run with sigma ~ HalfNormal(0.5) and compare")
print("rho and sigma posteriors to the default HalfNormal(1) fit.")
print("If mean(rho) shifts by > 0.10, the data are sparse enough that the prior matters.")
```

### Computation time in production

For N=11,200 UK postcode sectors with nutpie on a 16-core cluster: 8--15 minutes for 4 chains of 1,000 draws. Without nutpie: 25--50 minutes. These are the published benchmarks from the insurance-spatial README.

The scaling factor computation (an N x N dense eigendecomposition) is the only step that must be done offline for large N. For district-level analysis (N≈3,000) the scaling factor computes in under 60 seconds.

Territory models should be refitted annually, not monthly. The computation budget is not the constraint -- it is the data: claim patterns at sector level are noisy, and refitting more frequently than annually introduces spurious fluctuation.

### Convergence failure and what to do

If R-hat > 1.01 after 4 chains of 1,000 draws:

1. Increase draws to 2,000 and refit
2. Increase tune to 2,000 (more warmup)
3. Increase target_accept to 0.95
4. If persistent: the data may be very sparse for the resolution chosen. Move to district level.

### When BYM2 is not the right choice

BYM2 is appropriate when:
- Moran's I is significantly positive (spatial structure is present)
- You have at least 50--100 areas with meaningful exposure
- You can afford 15--60 minutes of compute time for the annual refit
- You need interpretable, auditable territory factors

Simpler alternatives that may be better:

- **Bühlmann-Straub credibility per sector** (Module 6): a parametric shrinkage method that blends each area's observed frequency towards the portfolio mean, weighted by exposure. Faster than BYM2 and requires no spatial computation, but treats each area as independent -- it does not borrow strength from neighbouring sectors
- **District-level GLM factor**: fewer areas, more data per area, cleaner estimates; appropriate when the portfolio is small
- **GBM with postcode sector encoded**: implicitly captures territory but produces uninterpretable effects; suitable if interpretability is not required

Do not build BYM2 just because it is available. Build it when it is justified by the data and useful for the business question.

### A note on the Emblem spatial module

Emblem (as of Emblem 4.x) includes a "spatial" module that applies kernel-based geographic smoothing to territory factors. We are describing Emblem 4.x behaviour based on available documentation; methodology may differ in later versions. The methodology is proprietary and not published. The smoothing is applied post-estimation, not as part of the likelihood. This means the uncertainty estimates from Emblem do not account for the spatial smoothing -- the credibility intervals are too narrow because they treat the smoothed factor as a point estimate rather than a posterior.

BYM2 propagates uncertainty through the spatial model. The credibility intervals on the relativities account for both estimation noise and the uncertainty in how much spatial smoothing is appropriate (through the posterior of rho).