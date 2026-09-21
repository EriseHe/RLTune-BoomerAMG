# 09 — Timing stability development experiment

The corrected two-execution study completed successfully on September 21.
The [completed results](../05_policy/RESULTS.md) exclude the discarded initial runs.

## September 21 correction

The initial timing runs at commit `75fdbbd` are **invalid**. The experiment
passed the setup learner's seven-value context into the solve interface, which
requires the full eight-value PDE context. The interface exception was treated
as a setup failure by the legacy recovery path. Fallback completed most inputs,
while the saved RL controllers had zero training steps. The initial smoke
checks verified completion and numerical consistency but failed to require
actual RL learning. Their earlier validation claim was insufficient.

The invalid timing runs and old smoke artifacts were removed from the repository
at the user's request. Do not use their costs, apparent variant differences or
execution variance. The frozen evaluation and common-hierarchy online phase use
the encoder's matrix-derived context and remain separate; all their recorded
attempts, hierarchy matches and learning evidence were checked independently.

The corrected path rebuilds the full PDE stream with the production helper,
verifies its alignment with the recorded setup context, and supplies the two
views to their respective interfaces. Non-native execution exceptions now abort
the experiment even when fallback succeeded. Completed active branches must
have nonzero RL training steps; progress includes those counters. Both 8³ and
60³ integration checks verify actual training and changed model parameters.

The user requested **two** corrected repetitions for development screening.
They start from scratch with a fresh independent calibration; completed
attribution experiments are not rerun:

```bash
python -m experiments.paper_final.run_05_controlled --run \
  --timing-only-from results/paper_final/05_policy/controlled \
  --output-root results/paper_final/09_algorithms/timing_corrected \
  --repetitions 2
```

Read `timing_corrected/status.json` and `timing_corrected/runs/` for the new
results. Two executions support preliminary development decisions, not a claim
that variance has reliably decreased.

This is the last phase of [05's controlled study](../05_policy/README.md), using
the [same prespecified manifest](../05_policy/protocol.json).

Run 5000 original development inputs at 60³, cap 50, RL from input 1001, with
two separately initialized execution repetitions. Every repetition contains
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
and sample SD across the two new executions. The `stable` control separates
the effect of deterministic tie priority from the additional band. Two repeats
are a development screen, not a precise uncertainty estimate. Reusing the
development stream is explicit. No RNG re-indexing, ridge change, feature change,
failure-target change or elite-search change is mixed into this experiment.

Outputs retain full trajectories on disk; only compact scalar summaries stay in
memory across repetitions, limiting accumulated memory-pressure differences.
