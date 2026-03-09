## Part 3: Neural Interaction Detection

After the CANN trains, we have a weight matrix `W1` connecting the input features to the first hidden layer. NID reads the interaction structure directly from this matrix.

### The key insight

Two features can only interact in a feedforward neural network if they both contribute to the same hidden unit in the first layer. If feature `i` has weight near zero into hidden unit `s`, and feature `j` has a large weight into unit `s`, they are not interacting through that unit — it is effectively a function of `j` alone.

Genuine interaction requires both features to have non-negligible weights into the same hidden unit.

### The NID formula

For a pair of features (i, j), the NID score is:

```sql
d(i, j) = Σ_s  z_s × min(|W1[s,i]|, |W1[s,j]|)
```

The sum is over all first-layer hidden units `s`. For each unit, `z_s` is how much that unit influences the output — computed as the product of absolute weight matrices from layer 2 through to the output (this tells you whether unit `s` actually matters for the prediction). The `min(|W1[s,i]|, |W1[s,j]|)` term is the bottleneck: it is large only if both features have large weights into unit `s`.

The result is a scalar score for each pair. Higher scores mean stronger detected interaction in the trained CANN.

### Why the min operator?

Consider unit `s` with `|W1[s,i]| = 10` and `|W1[s,j]| = 0.01`. Feature `j` barely enters unit `s`. There is no genuine interaction through that unit. Using a product `|W1[s,i]| × |W1[s,j]|` would give 0.1, suggesting a weak interaction. Using the min gives 0.01, correctly indicating that the bottleneck on feature `j` prevents a real interaction.

### Feature-level aggregation for categoricals

After one-hot encoding, a categorical feature with 6 levels expands to 5 binary columns in the input layer. NID collapses these back to a single feature-level importance by taking the L2 norm of the weights for those 5 columns per hidden unit. This is correct: the categorical variable is conceptually one thing, and its weight into a hidden unit is the combined effect of all its levels.

### NID is a ranking, not a test

The NID score produces a ranked list of candidate interactions. It is not a statistical test. A high NID score means the trained CANN learned interaction structure involving those features. It does not mean the interaction is statistically significant in the GLM sense.

That is what Stage 3 — the likelihood-ratio tests — provides.