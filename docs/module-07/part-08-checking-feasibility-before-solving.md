## Part 8: Checking feasibility before solving

Before running the solver, always verify that the problem has a solution. This takes one line:

```python
print(opt.feasibility_report())
```

**What you should see (at current rates, m = 1 for all factors):**

```python
Feasibility report at current rates (m = 1.0 for all factors):

  LR constraint:      VIOLATED   current=0.750, target=0.720, gap=-0.030
  Volume constraint:  SATISFIED  current=1.000, floor=0.970
  ENBP constraint:    SATISFIED  (no rate change, no breach possible)
  Factor bounds:      SATISFIED  all within [0.90, 1.15]

Feasibility of the full problem:
  A solution exists within the factor bounds at the given LR target and volume floor.
  Estimated minimum rate change required: ~4.1% uniform across all factors.
```

The LR constraint is violated at current rates — that is expected. It is why you are running a rate action. The question the feasibility check answers is: does a solution exist within the constraint set? Can you simultaneously achieve 72% LR, 97% volume, ENBP compliance, and factor movements within the caps?

If the answer is "No feasible solution found," you need to relax one or more constraints before proceeding. The most common choices are:

1. **Loosen the volume floor**: changing from 97% to 96% gives the optimiser an extra 1% volume to trade against LR improvement. This is often substantial.
2. **Widen the factor caps**: changing from [0.90, 1.15] to [0.85, 1.20] gives more rate-taking capacity.
3. **Accept a less ambitious LR target**: if 72% is genuinely infeasible within the approved parameters, present the frontier to the pricing committee and let them choose a feasible point.

Never relax the ENBP constraint. It is a regulatory requirement. Relaxing it to achieve a better LR is a regulatory breach.