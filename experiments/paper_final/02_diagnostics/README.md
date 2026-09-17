# 02 — Development diagnostics

[The prespecified protocol](protocol.json) is emitted by
`python -m experiments.paper_final.run_02_diagnostics`. Add `--run` to execute.
Outputs: `results/paper_final/02_diagnostics/`.

## Figures

The [figure index](../../../results/paper_final/02_diagnostics/figures/README.md)
links the held-out cost comparison, frozen per-cycle actions and time-feature
saturation plots. Generate them from saved artifacts without running solvers:

```bash
MPLBACKEND=Agg python -m experiments.paper_final.plot_02_diagnostics
```

Figures use evaluation cases 7–12; baseline selection remains on cases 1–6.
Plotted cost totals and time-feature fractions are checked against `summary.json`.

## Current status and amendment

Both screens are complete. Diffusion used 12 inputs × 2 setups × 49 policies
= 1176 primary executions. Advection used the original 12 inputs × 2 setups ×
14 policies = 336 additional primary executions. Frozen-state checks passed.
The 58/64 advection short-training rows retained before the user requested stage
03 first remain in the data; that short training was not resumed and produced
no checkpoint. The original protocol and prior trajectory bytes remain intact.

The completed advection diagnostic used stage 03 replicate 1's
`start_1000_final.npz` with its encoder metadata, chosen before seeing 03 results.
Keep the original fresh evaluation inputs and fixed setup tuples. This replaces
the planned 64-problem short training; do not rerun completed diffusion cells.
Stage 02 results did not tune the stage 03 protocol or algorithm.
The result directory contains the full summary, Chinese report, continuation
provenance, completion marker, exact source snapshot and independent final audit.

The continuation command was:

```bash
python -m experiments.paper_final.run_02_diagnostics --resume-advection \
  results/paper_final/03_activation/advection_80_s1/checkpoints/start_1000_final.npz
```

The continuation validates the original record order and input hashes, retains
existing cells, rejects duplicate cells, and uses a process lock. Its five tests
passed without native PDE solves. Diagnostic RL output now includes the actual
fallback residual for final-tolerance checks; the solve algorithm is unchanged.

## Original protocol

- **60³ diffusion:** 12 fresh matrix/RHS inputs, two fixed setup tuples, all 41
  fixed weights, six short periodic schedules, frozen LCB and mean-only RL.
- **80³ diffusion–advection:** 64 short-training problems for a new 34-state /
  306-feature controller, then 12 different fresh inputs, two setup tuples,
  six fixed weights, the same six schedules and two frozen scoring rules.
- The learned setup tuple is the most frequent tuple in the completed
  diffusion pilot's last 1000 problems. It is selected before fresh outcomes.
  Its advection use is a transfer diagnostic, not an advection-trained setup
  learner. The reference tuple provides a second hierarchy source.
- Each policy reconstructs the hierarchy from identical matrix/RHS and setup
  parameters. This matches configuration, not a reused in-memory AMG object.
- Cases 1–6 select one fixed weight and one schedule per setup. Cases 7–12
  evaluate those selected baselines and frozen RL. These six-case estimates are
  a development screen, not final statistical evidence.
- Policy execution order is randomized within each case. Native solves are
  sequential, single-threaded. Primary failures retain their costs and trigger
  one reference fallback; there is no setup reselection in this fixed-hierarchy
  experiment. Final experiments use the appropriate full production protocol.
- Selection cost includes native solve, controller/decision time and fallback
  setup time. Only the shared primary setup cost is excluded. Full recorded
  completion costs are reported separately.

The run records cycle traces, primary and unrecovered failures, cycle caps,
time-feature saturation, recovery cost and frozen-state checks. The final stage
03 checkpoint passes numerical audits before and after the frozen evaluation.
The abandoned short training passed audits at cases 16/32/48 in its original
execution, but the numerical values were not saved and are not reconstructed.
No activation-time comparison is performed in stage 02.

The existing schedule helper now records its cycle traces and decision time,
so comparisons include that measured overhead. Assembly, Python/native call
boundary costs outside existing timers, logging and checkpoint I/O are outside
the per-problem metric, as in the main runner.

`summary.json` is the machine-readable result. `REPORT.md` records interpretation
after the run. A formal matched-hierarchy study using final training checkpoints
and more independent inputs remains stage 05.
