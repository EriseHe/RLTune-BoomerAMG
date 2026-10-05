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
