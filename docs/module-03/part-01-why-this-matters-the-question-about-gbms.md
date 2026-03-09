## Part 1: Why this matters - the question about GBMs

The question most pricing teams ask about GBMs is: "does it beat the GLM?" That is the wrong question. It leads to cherry-picked test splits, optimistic Gini comparisons, and over-confident deployment decisions.

The right question is: "does the GBM find genuine risk signal that the GLM misses, and is that signal large enough to justify the additional governance overhead?"

The difference matters. A GBM always fits the training data better than a GLM - it is more flexible by design. What you need to establish is whether that additional flexibility corresponds to real out-of-sample discrimination, and whether the improvement is large enough to justify a more complex model that requires a more expensive audit trail.

This module shows you how to establish that rigorously.

### Why CatBoost specifically

There are three major GBM libraries in common use: XGBoost, LightGBM, and CatBoost. For insurance pricing, CatBoost has two properties that make it the better choice.

**Native categorical handling.** Rating factors in UK motor pricing are categorical: area band, vehicle group, NCD years. XGBoost and LightGBM require you to encode these as integers or one-hot dummy variables before training. One-hot encoding a 50-level vehicle group creates 50 binary features and destroys any ordinal structure. Integer encoding imposes an assumption that the factor has a linear effect in the encoded order. CatBoost uses ordered target statistics - a form of regularised mean encoding computed on a permuted training set - which handles categorical factors correctly without either assumption.

**Proper Poisson loss with exposure offset.** CatBoost implements a Poisson log-likelihood loss and integrates the exposure offset cleanly into its Pool object (explained in Part 3). This is the right objective for claim count modelling and the cleanest integration for insurance data.