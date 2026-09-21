# 09 — Timing stability development experiment

This is the last phase of [05's controlled study](../05_policy/README.md), using
the [same prespecified manifest](../05_policy/protocol.json).

Run 5000 original development inputs at 60³, cap 50, RL from input 1001, with
three separately initialized execution repetitions. Every repetition contains
three independently adapting Joint branches:

| Variant | Selection rule |
|---|---|
| `raw` | Original score minimum, including the original random numerical-tie rule |
| `stable` | Numerical-tie set; retain the previous reference if eligible, otherwise fixed arm-ID priority |
| `near_tie` | Same stable priority, with an independently calibrated score band |

All branches retain the original structured candidate generator and RNG streams.
The score-audit callback reproduces the raw decision and RNG sequence under
identical observations. It logs candidate hashes, score margins, RNG states,
reference retention and band width. Logged audit computation is included in
selection overhead and is present in all three branches. Do not compare these
new timings to historical timings as though instrumentation were absent.

Let `d` be the current reference-versus-minimum feature difference. The band is

`h = 2 sigma sqrt(max(0, d' V^-1 d - lambda ||V^-1 d||²))`.

Here `sigma` is the 75th percentile of within-input timing SDs from the separate
Default calibration. This is a declared homoscedastic fixed-design heuristic;
the actual measurements may be correlated, heteroscedastic and action dependent.
It is not a statistical confidence interval or a bound on excess physical cost.
The reference is always rescored on the current problem. The full calibration
cost is charged once to each near-tie logical execution. No threshold is selected
using the held-out performance or proximity to the earlier favorable trajectory.

The primary evidence is cumulative algorithm cost, final failures, and the mean
and sample SD across the three new executions. The `stable` control separates
the effect of deterministic tie priority from the additional band. Three repeats
are a diagnostic starting point, not a precise uncertainty estimate. Reusing the
development stream is explicit. No RNG re-indexing, ridge change, feature change,
failure-target change or elite-search change is mixed into this experiment.

Outputs retain full trajectories on disk; only compact scalar summaries stay in
memory across repetitions, limiting accumulated memory-pressure differences.
