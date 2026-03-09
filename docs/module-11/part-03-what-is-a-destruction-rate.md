## Part 3: What is a destruction rate?

Before we touch the maths, we need a consistent unit. Every claim in a commercial property portfolio can be expressed as a destruction rate:

```sql
z = loss / MPL
```

where MPL is the maximum possible loss for that risk. The destruction rate z lies in [0, 1]. A partial loss of £300,000 on a risk with MPL £2 million gives z = 0.15. A total loss gives z = 1.0.

The destruction rate is the fundamental unit of the MBBEFD framework because it removes size heterogeneity from the data. A small fish and chip shop and a large distribution centre are not directly comparable in loss amount terms, but they are comparable in destruction rate terms. Both can be either a 10% partial loss or a total loss. By working in z-space, we can pool claims data across risks of different sizes to fit a single curve.

**What destruction rates look like in practice:**

For commercial property (fire), you typically see:
- Many small partial losses (z in the range 0.01-0.15): these are contained fires, minor water damage, small break-ins
- A spread of medium partial losses (z 0.15-0.60): significant fire damage, major flood events
- A small number of large partials (z 0.60-0.99): fires that consumed most of the building
- Total losses (z = 1.0): the building is destroyed

The proportion of total losses is a key parameter. For Y2 (standard commercial property), about 13% of losses that enter the data are total losses. For Y4 (heavy industrial), only about 0.6% are total losses. This is intuitive: a warehouse of consumer goods burns to nothing more readily than a reinforced concrete industrial plant.

### Computing destruction rates from claims data

In code:

```python
# Suppose you have a claims DataFrame with loss amount and MPL per claim
# We show this in Polars, as per the course convention

claims = pl.DataFrame({
    "claim_id":  ["C001", "C002", "C003", "C004", "C005"],
    "loss":      [150_000, 800_000, 50_000, 2_000_000, 300_000],
    "mpl":       [2_000_000, 2_000_000, 500_000, 2_000_000, 1_000_000],
})

claims = claims.with_columns(
    (pl.col("loss") / pl.col("mpl")).alias("destruction_rate")
)

print(claims)
```

**Output:**

```python
shape: (5, 4)
┌──────────┬───────────┬───────────┬──────────────────┐
│ claim_id ┆ loss      ┆ mpl       ┆ destruction_rate │
│ ---      ┆ ---       ┆ ---       ┆ ---              │
│ str      ┆ i64       ┆ i64       ┆ f64              │
╞══════════╪═══════════╪═══════════╪══════════════════╡
│ C001     ┆ 150000    ┆ 2000000   ┆ 0.075            │
│ C002     ┆ 800000    ┆ 2000000   ┆ 0.4              │
│ C003     ┆ 50000     ┆ 500000    ┆ 0.1              │
│ C004     ┆ 2000000   ┆ 2000000   ┆ 1.0              │
│ C005     ┆ 300000    ┆ 1000000   ┆ 0.3              │
└──────────┴───────────┴───────────┴──────────────────┘
```

C004 is a total loss (z = 1.0). C001 and C003, despite very different loss amounts (£150k and £50k), have destruction rates of 0.075 and 0.10 -- both small partial losses in the same region of the severity distribution. This is the normalisation that makes pooling valid.