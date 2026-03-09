## Part 6: Wrapping the data in PolicyData and FactorStructure

The `rate-optimiser` library needs the data in two specific wrapper objects. This step deserves more attention than it usually gets, because getting it wrong silently corrupts the ENBP calculation.

### PolicyData

`PolicyData` validates the input and exposes summary statistics that are the inputs to every constraint. It requires a pandas DataFrame (the library boundary where we convert from Polars):

```python
from rate_optimiser import PolicyData, FactorStructure

# Convert to pandas at the library boundary
df_pd = df.to_pandas()

data = PolicyData(df_pd)

print(f"n_policies: {data.n_policies:,}")
print(f"n_renewals: {data.n_renewals:,}")
print(f"channels:   {data.channels}")
print(f"Current LR: {data.current_loss_ratio():.4f}")
```

**What you should see:**

```
n_policies: 5,000
n_renewals: 3,250 (approximately)
channels:   ['PCW', 'direct']
Current LR: 0.7502
```

Check the LR against your own calculation: `df["technical_premium"].sum() / df["current_premium"].sum()` should match `data.current_loss_ratio()`. If they do not match, there is a mismatch between what you put in the DataFrame and what the library is using. Do not proceed until they agree.

### FactorStructure

`FactorStructure` tells the library which columns are rating factors, and which of those factors apply only to renewals (not new business). This is the most consequential configuration decision in the entire module.

```python
FACTOR_NAMES = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=FACTOR_NAMES,
    factor_values=df_pd[FACTOR_NAMES],
    renewal_factor_names=["f_tenure_discount"],
)

print(f"n_factors:            {fs.n_factors}")
print(f"shared factors:       {[f for f in FACTOR_NAMES if f not in fs.renewal_factor_names]}")
print(f"renewal-only factors: {fs.renewal_factor_names}")
```

**What you should see:**

```
n_factors:            5
shared factors:       ['f_age', 'f_ncb', 'f_vehicle', 'f_region']
renewal_only factors: ['f_tenure_discount']
```

### Why renewal\_factor\_names matters so much

The ENBP constraint requires: for each renewal policy, the adjusted renewal premium must not exceed what a new customer with the same risk profile would be quoted.

The "same risk profile" for a new customer means the same age, NCB, vehicle, and region — but a new customer does not get the tenure discount, because tenure requires being a customer for some years. The tenure discount is renewal-only.

So the new business equivalent premium is computed with all factor adjustments except the renewal-only ones:

```
NB equivalent premium = current_premium x m_age x m_ncb x m_vehicle x m_region
Adjusted renewal premium = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure_discount
```

The ENBP constraint requires: `adjusted_renewal_premium <= NB_equivalent_premium`

Which simplifies to: `m_tenure_discount <= 1.0`

This means the optimiser can never increase the tenure discount factor above 1.0 for renewal customers. The discount can only stay flat or increase. If the optimiser wants to improve LR by reducing tenure discounts (increasing m\_tenure above 1.0), the ENBP constraint blocks it. This is intentional — it is the FCA's requirement.

**What if you get renewal\_factor\_names wrong?**

If you forget to put `f_tenure_discount` in `renewal_factor_names`, the library computes the NB equivalent premium using all five factors, including the tenure discount. The ENBP constraint then compares:

```
adjusted_renewal = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure
NB equivalent    = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure
```

Both sides include m\_tenure. The constraint reduces to `1 <= 1`, which is always satisfied. ENBP is trivially satisfied regardless of what the optimiser does with the tenure discount. The solver can set m\_tenure = 1.15 (a 15% reduction in renewal discounts) and the ENBP check will pass — even though in reality, this would be an ENBP breach for every renewal customer receiving a tenure discount. You would have a regulatory breach disguised as compliance. This is why the configuration matters.