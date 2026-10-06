# Reproducing the SISC studies

Run commands from the repository root in the activated Python environment.
Install with `python -m pip install -e '.[artifacts]'` and build with
`make -C hypre JOBS=4`; see the [README](../README.md) for native prerequisites.
See the [organization report](sisc_repository_cleanup.md) for validation results
and environment limitations.

## Choose a study

This is the command guide for the two official paper studies. Detailed protocol
and input notes are linked alongside each study; code ownership is documented
separately in the [repository layout](repository_layout.md).

- [Module 04: online autotuning](#module-04-exact-configurations) — validate, run,
  and analyze the formal single-seed design or the accepted six-seed batches.
- [Module 05: matched-hierarchy comparison](#module-05-verify-and-retime-the-frozen-bundle)
  — verify the accepted bundle, then prepare, run, plot, package, and check a
  fresh retiming.
- [Timing and recovery](#timing-and-recovery) — cost definitions, failure
  handling, fixed comparators, and interpretation limits shared by the studies.
- [Module 06: recovery stress test](#module-06-recovery-stress-test) — exploratory
  four-worker comparison on a failure-heavy recorded Module 04 stream.

## Included evidence

| Study | Included protocol and evidence | Fresh execution |
|---|---|---|
| Module 04: Default, LinUCB and LinUCB–LSTDQ over 5000 paired inputs for diffusion and diffusion–advection at 40³/60³/80³ | [Formal design](../experiments/paper_final/04_online/20260920_formal/README.md), exact online configurations, [accepted six-seed aggregate](../experiments/paper_final/reproduction/online/six_seeds.json) | `run_04_online`, `analyze_04_online` |
| Module 05 / Run 05: frozen policies on matched diffusion-60³ hierarchies | [Accepted record](../experiments/paper_final/05_policy/RUN05.md), [protocol and compact results](../experiments/paper_final/reproduction/matched_policy/README.md), verified frozen-input bundle | `run_05_policy_minimax`, `plot_05_policy_minimax`, `package_05_policy_minimax` |

The clone contains compact accepted measurements and the inputs needed to run the
studies. It does not contain the original full trajectories or generated figure
releases. New measurements use current source and the recorded current runtime;
they do not replace the September results or claim identical timings.
The [capture manifest](../experiments/paper_final/reproduction/manifest.json)
preserves original configuration/result hashes. The
[Run 05 bundle manifest](../experiments/paper_final/reproduction/matched_policy/frozen_inputs/manifest.json)
separately verifies its frozen inputs and derived evidence.

## Module 04: exact configurations

The [online input captures](../experiments/paper_final/reproduction/online/streams/README.md)
contain 36 distinct streams (180,000 rows, approximately 20 MB compressed). Their
canonical hashes exactly match the 42 recorded configurations. Accepted runs
replay these matrix arguments and contexts to avoid platform-dependent rounding
in logarithms; changed sampling settings with an old hash are rejected. Unpinned
or unrecognized configurations continue to use the original sampler.

There are 42 tracked experiment configurations: six in the formal September 20
design and 36 in the accepted six-seed batches. These are distinct protocol
captures, not 42 independent seeds to pool. All use 5000 inputs per PDE/grid group,
the three paired methods, tolerance `1e-6`, the recorded 50-cycle cap, and solve
learning from problem 1001.

The three methods are Default, LinUCB with default solve, and LinUCB with
recursive LSTDQ. They use the shared online experiment engine and its
recovery/cost accounting. The [formal protocol](../experiments/paper_final/04_online/20260920_formal/README.md)
records group order, context sizes, RNG seeds, and the optional `run.command`
launcher. The [accepted six-seed aggregate](../experiments/paper_final/reproduction/online/six_seeds.json)
records the later captured batches.

Current code uses the unversioned names `SharedLinUCB`, `RecursiveLstdqController`,
and solve kind `recursive_lstdq`. Frozen configurations and checkpoint bytes retain
their original recorded identities. The reader translates `recursive_lstdq_v3`
and its `lstdq_v3` confidence settings into the same method; fresh configurations
use `recursive_lstdq` and the single `lstdq` settings section.

Validate all three captured suites without native solves:

```sh
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --validate-only
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/reproduction/online/seeds_1_to_3/suite.json --validate-only
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/reproduction/online/seeds_4_to_6/suite.json --validate-only
```

For the six-group formal single-seed design, use a fresh output root:

```sh
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --output-root results/paper_final/04_online/formal_retiming --run
python -m experiments.paper_final.analyze_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --output-root results/paper_final/04_online/formal_retiming
```

For the accepted six-seed design, use the six `suite_seed_N.json` files:

| Global seed labels | Suite directory | Configuration count |
|---|---|---:|
| 1–3 | `experiments/paper_final/reproduction/online/seeds_1_to_3/` | 18 |
| 4–6 | `experiments/paper_final/reproduction/online/seeds_4_to_6/` | 18 |

Each seed suite contains the six PDE/grid groups. For example:

```sh
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/reproduction/online/seeds_1_to_3/suite_seed_1.json --output-root results/paper_final/04_online/six_seed_retiming/seed_1 --run
python -m experiments.paper_final.analyze_04_online --suite experiments/paper_final/reproduction/online/seeds_1_to_3/suite_seed_1.json --output-root results/paper_final/04_online/six_seed_retiming/seed_1
```

Repeat with the corresponding suite and a separate output root for seeds 2–6.
The accepted measurements used three concurrent single-thread seed processes per
batch, with each seed's groups run serially and no claimed CPU affinity. Preserve
that concurrency when comparing timings; sequential execution is a different
recorded timing condition. Global seeds 4–6 retain local runner labels 1–3 and
explicit global mappings in their captured metadata.

Execution records configurations, source/binary provenance, paired input streams,
per-method trajectories and final checkpoints. Completed analysis writes the
current `analysis/module04.json` and `.md` reports under each selected output
root; these files contain the Module 04 online and cost-breakdown
results. The runner audits completed groups before reuse and refuses changed
source/configuration or an incomplete group's automatic restart.

## Module 05: verify and retime the frozen bundle

The tracked bundle is approximately 14 MB. It contains six final controller
checkpoints and encoders, their configurations, exact accepted inputs/jobs,
frozen hierarchy and fixed-weight choices, theory records, the accepted reference
environment, and 29,520 compact retained fixed-grid records. No earlier study
folders or setup-bandit checkpoint reconstruction are required.

Verify it without HYPRE solves:

```sh
python -c 'from experiments.paper_final.common.frozen_inputs import DEFAULT_BUNDLE, verify_bundle; print(verify_bundle(DEFAULT_BUNDLE))'
```

Verification checks every bundled file against its manifest, ties exact accepted
input/checkpoint bytes to their original prepared hashes, and independently
reconstructs the successful stream-wide and per-instance fixed-grid selections.
Compact scan files preserve ten fields and every recorded timing float. Their
original raw source hashes and new derived-file hashes are recorded separately
in `fixed_scan_evidence.json`; omitted raw fields are not represented as identical
original scan files.

After building the native interfaces, prepare and execute a fresh output:

```sh
python -m experiments.paper_final.run_05_policy_minimax prepare --output results/paper_final/05_policy/run05_retiming
python -m experiments.paper_final.run_05_policy_minimax run --output results/paper_final/05_policy/run05_retiming
python -m experiments.paper_final.plot_05_policy_minimax --output results/paper_final/05_policy/run05_retiming
python -m experiments.paper_final.package_05_policy_minimax --run results/paper_final/05_policy/run05_retiming --output results/paper_final/05_policy/releases/run05_retiming
python -m experiments.paper_final.verify_05_policy_minimax --root results/paper_final/05_policy/releases/run05_retiming
```

Use `--bundle PATH` on preparation or packaging for another copy of the verified
accepted bundle. Preparation requires an empty new directory. Packaging requires
a new release directory. Neither command replaces an existing accepted result.

Preparation copies accepted checkpoint/input/job/protocol bytes unchanged and
captures complete current project source plus current execution environment.
Comparison with the accepted environment is recorded without claiming September
binary or package identity. Resume refuses changes to prepared inputs, evidence,
source files, native binaries, Python, packages, MPI or thread settings.

The six controllers remain frozen. The same 100 cases and three prescribed
repetitions retain the accepted hierarchy tuples, chosen weights, shared
execution cells and randomized policy order. The run audits repeated reference
and periodic solves before evaluation, then automatically analyzes completed raw
logs into summary/cost/trace files. The `analyze_05_policy_minimax.analyze(output)`
helper can rebuild that analysis from audited raw logs; it has no separate CLI.
Figure generation reads completed analysis and labels the illustrative best
observed seed explicitly, while retaining all six in aggregate figures.

The release contains fresh raw evaluations, compact fixed-grid selection evidence,
checkpoints, frozen numerical source snapshots, separate current presentation
sources, theory, tables and figures. Its copied `reproduction/verify_archive.py`
uses only Python's standard library. From an extracted release:

```sh
python reproduction/verify_archive.py
```

It checks file hashes, execution coverage, prescribed schedules, all reported
cost arrays and both fixed comparator selections. Original prepared outputs retain
strict source guards; changed execution source cannot resume historical runs.

## Timing and recovery

Online native cost includes all attempted setup and solve work, including
recovery. Total online cost adds recurring bandit/controller work. One-time model
initialization, candidate preparation, matrix/RHS assembly and output I/O are
outside these declared components; method-call wall time is recorded separately.

Online construction failure permits at most three learned setup attempts total,
excluding failed exact configurations on the same problem. Solve nonconvergence
invokes default recovery directly. At most one default attempt follows, and all
attempted work remains charged. Under `rollback_unrecovered`, provisional learning
feedback is rolled back if the final recovery also fails. Unrecovered counts must
accompany cost comparisons; attempted-solve cost alone does not establish that
every problem was solved.

Frozen Run 05 uses one primary matched hierarchy and the same single default
fallback after numerical failure, with no setup reselection or learning.
Programming/execution errors abort. Native continuation excludes common initial
setup and controller dispatch but includes recovery setup/solve; inclusive
continuation adds controller work. Average the three repetitions within case
before forming ratios of summed costs.

The fixed comparators are successful best-observed choices on the 41-weight grid,
then independently retimed. The accepted checkpoints/cases were already examined,
and seed 4 was refreshed in earlier development; this is a prescribed-policy
follow-up. The `(2.85, 1.10)` pair is selected by the normalized SPD smoothing
surrogate, not by current timing outcomes. Its multilevel runtime comparison is
empirical. See the [theory index](theory/README.md).

## Module 06: recovery stress test

This exploratory module lives on `experiments/module06-recovery`. It reuses the
existing learners and native solve engine, with its own retry coordination and
analysis. Module 04/05 source and accepted records are unchanged.

The selection ranks the 18 accepted diffusion-advection runs by full-stream
LinUCB-LSTDQ unrecovered failures. Global seed **5**, base seed **339923787**, on
the **40³** grid ranks first with **62/5000** failures (191 summed over the three
original methods). The exact stream hash is
`c3f7866f341b204de0b884df00345a5313a35262a2e7788e6edae625732fa4b4`.
The complete ranking and configuration hash are saved in each run's protocol.

| Variant | Learned attempts | Retry trigger | RL target after a failed attempt |
|---|---:|---|---|
| `original` | Up to 3 | Construction failure only; solve failure goes directly to default | Native cycle cost plus default recovery |
| `shared3_cost` | Up to 3 | Construction or solve failure | Native cycle cost plus all subsequent native recovery work |
| `shared3_no_cost` | Up to 3 | Construction or solve failure | Native cycle cost only |
| `shared5_no_cost` | Up to 5 | Construction or solve failure | Native cycle cost only |

These counts include the initial attempt. One default attempt follows exhaustion
of the learned budget. Each failed exact setup is excluded on that problem,
with immediate bandit failure-risk feedback. Bandit runtime labels include the
remaining work through completion. All variants keep the existing final-failure
rollback rule, the 50-cycle cap, tolerance `1e-6`, fresh learner initialization,
and RL activation at problem 1001.

The unified variants provisionally learn cycle costs during each attempt. Once
the whole problem is recovered, `shared3_cost` replaces those provisional updates
by replaying the recorded transitions from the preproblem learning snapshot,
adding the realized subsequent native setup/solve time at each failed terminal.
This preserves the LSTDQ equations and episode covariance without duplicating
samples, native solves, or random action draws. Replay overhead is measured and
included in total time. No-recovery-cost variants retain their attempt-cost
updates. Fully unrecovered problems roll back all provisional learning, while
their attempted work remains in the reported costs.

The original three candidate rows per problem are preserved in every variant.
Attempts 4/5 use a separate two-row schedule with the original candidate seed
plus 600006. Extra attempts do not shift later problems' original candidate rows.

Launch all four workers with a new output directory:

```sh
python -m experiments.paper_final.run_06_recovery --output results/paper_final/06_recovery/seed5_advection40
```

The launcher verifies identical input streams, initial models and base candidate
schedules before releasing all four workers. Each uses one numerical-library
thread and one MPI rank; macOS workers request user-initiated QoS, which does not
guarantee hard core affinity. Source snapshots, native-library hashes, environment
metadata, per-attempt trajectories, progress and final checkpoints stay with the
run. Existing outputs are never overwritten or automatically resumed.

At completion, the command audits retry limits, failure feedback, target costs,
and cost decomposition, then writes `comparison.md`, `.json`, `.csv`, `.png`,
and `.pdf`. Analysis can be repeated with:

```sh
python -m experiments.paper_final.analyze_06_recovery --output results/paper_final/06_recovery/seed5_advection40
```

Full-stream totals retain failed work. Costs on the common successful input
intersection and the final-1000/RL-active windows are also recorded. This single
seed was deliberately selected using earlier failures; its results are diagnostic.

`--smoke-cases 20` runs a shortened prefix and activates RL at problem 2 for
native validation. Smoke outputs are marked and must not be pooled with the
full experiment. The permanent `test_06_recovery.py` checks mixed failure
triggers, three/five-attempt budgets, exclusions, rollback, candidate pairing,
and delayed-target equivalence. Temporary smoke outputs are removed after use.

### Recorded stress-test results

The 2026-10-06 run completed all four 5000-problem streams concurrently using
source commit `62e19b9`. Pairing, retry limits, target costs and timing accounting
passed validation. The [compact capture](../experiments/paper_final/reproduction/recovery/20261006_seed5_advection40.json)
contains the protocol, all analysis windows, attempt counts, failure indices and
raw-output hashes. Full trajectories and checkpoints remain in
`results/paper_final/06_recovery/20261006_seed5_advection40/` outside Git.

| Variant | Unrecovered / 5000 | Default fallbacks | Native (s) | Total (s) |
|---|---:|---:|---:|---:|
| Original recovery | 61 | 381 | 502.774 | 591.982 |
| Shared 3, with RL recovery cost | 60 | 61 | 528.159 | 623.558 |
| Shared 3, without RL recovery cost | 60 | 60 | 524.611 | 618.368 |
| Shared 5, without RL recovery cost | 59 | 59 | 545.290 | 642.217 |

These full-stream totals include unsuccessful work. Relative to the original
policy, total cost increased by 5.33%, 4.46% and 8.49%, respectively. On the 4937
inputs completed by every variant, the corresponding total-cost changes were
+0.23%, -0.55% and -1.25%; repeated work on unsuccessful inputs accounts for most
of the full-stream increase.

The five-attempt policy reached attempts four and five on 59 problems, and all
59 remained unrecovered. Thus no success directly required either extra attempt.
Its one fewer failure than the three-attempt no-recovery-cost policy occurred
under a different learned trajectory; it is not evidence that attempts four or
five rescued that input.

In the final 1000 problems, total costs were 105.741, 104.342, 105.464 and
102.115 seconds, with 13, 12, 13 and 12 unrecovered failures in the same order.
Neither RL target is consistently preferable across the reported windows.
This selected single-seed concurrent run supports further investigation of the
shared three-attempt rule, but establishes no broad performance advantage or
reason to increase the budget to five. It does not replace the accepted paper
experiments.
