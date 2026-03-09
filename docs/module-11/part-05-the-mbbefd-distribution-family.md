## Part 5: The MBBEFD distribution family

### Origins: Bernegger 1997

Stefan Bernegger's 1997 paper in the ASTIN Bulletin introduced the MBBEFD class as an analytic framework for insurance exposure curves. The name refers to three physics distribution families -- Maxwell-Boltzmann, Bose-Einstein, and Fermi-Dirac -- whose mathematical structures Bernegger unified into a single parametric family suited for insurance severity.

The insight was that the Swiss Re standard curves, which had been in use for decades as lookup tables, could be captured analytically by a two-parameter family. This meant the curves could be fitted to data, compared using information criteria, and used without the discretisation error of a table.

The MBBEFD distribution is **mixed continuous-discrete**:

- It has a **continuous density f(x) on [0, 1)** representing partial losses
- It has a **point mass P(X = 1) = 1/g** at x = 1 representing total losses

The parameter g controls the total loss probability: large g means total losses are rare. The parameter b controls the shape of the partial loss distribution: it determines how concentrated losses are at the low end versus the high end of the severity spectrum.

### The exposure curve formula

For MBBEFD(g, b) with b not equal to 1, the exposure curve has an analytic closed form:

```python
G(x) = ln[(g-1)*b / (1-b) + (1-g*b) / (1-b) * b^x] / ln(g*b)
```

This looks intimidating. Let us unpack it.

Define two constants:

```python
A = (g-1)*b / (1-b)
C = (1-g*b) / (1-b)
```

Note that A + C = 1. These constants partition the numerator: A handles the "intercept" of the curve at x = 0, and C modulates how b^x (the power of b) contributes as x increases.

The formula becomes:

```python
G(x) = ln[A + C * b^x] / ln(g*b)
```

At x = 0: G(0) = ln[A + C * 1] / ln(g*b) = ln[A + C] / ln(g*b) = ln(1) / ln(g*b) = 0. Correct.

At x = 1: G(1) = ln[A + C * b] / ln(g*b). Substituting the definitions of A and C:

- A = (g-1)b / (1-b)
- C*b = [(1-g*b)/(1-b)] * b = b(1-g*b)/(1-b)
- A + C*b = [(g-1)b + b(1-g*b)] / (1-b) = [g*b - b + b - g*b^2] / (1-b) = g*b*(1-b)/(1-b) = g*b

So ln(A + C*b) = ln(g*b), and G(1) = ln(g*b) / ln(g*b) = 1. Correct.

The formula is analytically exact at both boundaries: G(0) = 0 and G(1) = 1. No clamping is required for correctness. We can verify numerically:

```python
y2 = swiss_re_curve(2.0)
print(y2.exposure_curve(0.0))   # Should be 0.0
print(y2.exposure_curve(1.0))   # Should be 1.0
print(y2.exposure_curve(0.5))   # Something between
```

### The CDF and point mass

The cumulative distribution function for x in [0, 1) is:

```python
F(x) = 1 - (1-b) / [(g-1)*b^(1-x) + 1 - g*b]
```

At x = 1, F(1) = 1 (inclusive of the total-loss atom). The point mass at x = 1 is:

```sql
P(X = 1) = 1/g
```

This is why g > 1 is required: if g were 1, the point mass would be 1.0 (certain total loss), and the continuous component would have zero weight. The constraint g > 1 also ensures the exposure curve is not trivially degenerate.

### The Bose-Einstein degenerate case

When b approaches 1, the MBBEFD family degenerates to the uniform distribution on [0, 1]. In this limit:

```sql
G(x) = x   (for all x in [0, 1])
```

This is the exposure curve of a distribution where every level of loss is equally likely. In practice, b is never exactly 1 in insurance data, but the library handles this limiting case explicitly to avoid division by zero in the formula.

### The mean

The mean destruction rate is computed numerically. There is no closed-form expression analogous to the exposure curve formula. The mean is:

    E[Z] = integral from 0 to 1 of z * f(z) dz  +  1.0 * P(Z = 1)

where f(z) is the MBBEFD density on [0, 1) and P(Z = 1) = 1/g is the point mass at total loss. The library handles this computation internally. Higher c-parameter values (heavier, more industrial risks) produce lower mean destruction rates because the loss distribution is concentrated at small partial losses.