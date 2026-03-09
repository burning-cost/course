## Part 12: Applying B-S in log-rate space — why it matters

### The Jensen's inequality problem

For a multiplicative pricing framework (Poisson log-link), the correct approach is to blend in log-rate space. The `log_transform=True` argument in `buhlmann_straub()` does this automatically, but it is worth understanding why.

If you apply B-S directly to rates and then convert the estimate to a log-scale relativity, you introduce a bias. The log of the expected value does not equal the expected value of the log:

```python
log(Z × X̄ + (1-Z) × mu)  ≠  Z × log(X̄) + (1-Z) × log(mu)
```

For typical motor insurance frequencies (5-10%), the difference is small. For extreme relativities — a very thin district with an observed rate of 0.5% against a portfolio mean of 7% — the bias can shift the estimate by 5-10% relative to the log-space calculation. Use `log_transform=True` as the default for any Poisson/multiplicative application.