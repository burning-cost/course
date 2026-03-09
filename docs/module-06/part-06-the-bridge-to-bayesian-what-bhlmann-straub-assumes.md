## Part 6: The bridge to Bayesian — what Bühlmann-Straub assumes

### B-S is empirical Bayes

Bühlmann-Straub is an empirical Bayes method. This is rarely stated clearly in actuarial textbooks, but it motivates everything in the second half of this module.

The credibility premium P_i is exactly the posterior mean of this Bayesian model:

```
X_{ij} | theta_i  ~  Normal(theta_i,  v / w_{ij})    [observation model]
theta_i            ~  Normal(mu, a)                    [prior on group mean]
```

The posterior mean of theta_i is:

```
E[theta_i | data] = Z_i × X̄_i + (1 - Z_i) × mu
```

where Z_i = w_i / (w_i + K), K = v/a. That is exactly the Bühlmann-Straub formula.

B-S plugs in point estimates of v and a (the structural parameters we estimated above). Full Bayesian treats v and a as uncertain — it places priors on them and integrates over that uncertainty. Three things follow from this:

**When B-S is sufficient:**
- Many groups (20+) so that v and a are estimated reliably from data
- One grouping variable
- Results needed in seconds, not minutes
- Regulatory documentation needs to be simple and traceable

**When full Bayesian is better:**
- Few groups (fewer than 10 affinity schemes) — with few groups, the estimate of the between-group variance is unreliable, and that uncertainty propagates into Z. Full Bayesian propagates it correctly; B-S ignores it.
- Multiple crossed grouping variables simultaneously (area AND vehicle group AND NCD band)
- Proper Poisson or Gamma likelihood — B-S assumes Normal errors in its derivation
- Credible intervals on individual segment rates are required for regulatory evidence or pricing decisions on thin segments