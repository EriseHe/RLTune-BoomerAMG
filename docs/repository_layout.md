# Repository layout

This submission tree contains only the sources, tests and computational evidence
needed for Module 04 online autotuning and accepted Module 05 matched-hierarchy
Run 05. The official entry modules keep their names under
`experiments/paper_final/`.

Start with the [root guide](../README.md) for installation and checks, or the
[reproduction guide](reproduction.md) to run either official study. This page
maps the source files and their responsibilities. Implementation details are in
the [setup guide](../setup/README.md), [solve guide](../solve/README.md), and
[native-binding guide](../hypre/bindings/README.md).

| Path | Responsibility |
|---|---|
| `problems/` | PDE coefficients, context encoders and deterministic input streams |
| `setup/` | Shared context-action LinUCB V4, hierarchy action spaces, candidate schedules and checkpoint persistence |
| `solve/` | Recursive LSTDQ V3, state/action encoders, cycle execution and recovery accounting |
| `hypre/source/` | Preserved HYPRE implementation and upstream notices |
| `hypre/interfaces/` | Project-owned C wiring into HYPRE and native regression tests |
| `hypre/bindings/` | Python native adapter, typed statuses and bounded fallback |
| `experiments/paper_final/online/` | Composition of setup and solve for the official online study; protocol loading, execution and reporting |
| `experiments/paper_final/common/` | Durable files, frozen evaluation, matched cost aggregation, worker supervision, input-bundle verification and rendering helpers |
| `experiments/paper_final/reproduction/` | Exact captured configurations, compact accepted results and the verified Run 05 input bundle |
| `experiments/runtime.py` | Explicit thread configuration and optional sleep prevention for command-line runs |
| `experiments/tests/` | Import/runtime and shared experiment checks |
| `scripts/check_repository.py` | Static checks and the five unit/native test groups |
| `docs/` | Reproduction, component layout and smoothing-theory references |
| `results/` | Generated outputs from fresh runs; large logs and figures are ignored |

## PDE problems

`problems/` defines the two PDE streams and does not import learners,
controllers, or experiment runners.

| File in `problems/` | Responsibility |
|---|---|
| `scalar_anisotropic_diffusion.py` | Scalar anisotropic diffusion with zero advection coefficients |
| `scalar_anisotropic_diffusion_advection.py` | The same diffusion family with independently sampled advection components |
| `amg.py` | Shared stencil-0 native coefficient mapping, coefficient sampler, and normalized PDE contexts |
| `streams.py` | Deterministic instance streams generated from independent child seeds in their original order |
| `registry.py` | The two problem identities and shared setup context views |

The underlying stream retains its eight-field coefficient representation.
Official diffusion experiments select `diffusion3d`; diffusion-advection
experiments select `canonical_no_c_mean`. Both retain an intercept and remove
redundant mean features from the learner-visible view. The original default
V4 context projection remains available for existing configuration defaults.

## Online experiment engine

The public Module 04 commands are `experiments.paper_final.run_04_online` and
`experiments.paper_final.analyze_04_online`. Their support package,
`experiments/paper_final/online/`, implements configuration, execution,
provenance, and reporting for Default, LinUCB, and LinUCB–LSTDQ.

| Files in `experiments/paper_final/online/` | Responsibility |
|---|---|
| `suite.py`, `analyze_suite.py` | Validate frozen group configurations, supervise runs, audit completed outputs |
| `run.py`, `runner.py` | Resolve one JSON configuration and assemble its method branches |
| `configuration.py`, `method_spec.py`, `methods.py`, `controllers.py` | Typed experiment settings and component construction |
| `execution.py`, `case_loop.py` | Randomized paired case loop, update order, recovery accounting |
| `setup_branches.py`, `action_spaces.py`, `problem_stream.py` | Setup candidates and accepted PDE contexts/streams |
| `native_case.py`, `native_evaluation.py` | Native default/fixed/periodic execution, reused by Module 05 |
| `feedback.py`, `comparison.py` | Observed costs and paired method comparisons |
| `protocol.py`, `artifacts.py`, `io.py`, `reporting.py` | Provenance, records, checkpoints and tables |
| `plot_run.py`, `plotting.py`, `action_trajectory_plot.py` | Figures from completed runs |

Learning algorithms remain under `setup/` and `solve/`; the experiment engine
composes them and owns experiment policy. Shared study helpers in
`experiments/paper_final/common/` handle durable artifact I/O, frozen evaluation,
matched cost aggregation, worker supervision, input verification, and figure
presentation. The [reproduction guide](reproduction.md#timing-and-recovery)
defines the accepted recovery rules, timing boundaries, and required outputs.

## Dependency direction

`problems`, `setup` and `solve` own their respective numerical logic.
Project-owned HYPRE adapters expose native operations. The paper experiment layer
combines these components into the three online methods and the matched frozen
comparison. Learner implementations do not import executable experiment scripts.

`setup.registry` constructs `default` and `linucb` setup branches.
`solve.registry` constructs `default`, `fixed` and `recursive_lstdq_v3` solve
branches. Controller and encoder pairing, numerical updates, candidate schedules
and checkpoint data stay with their owning packages. Setup/solve coordination
and study-specific policy selection stay in the experiment layer.

Cycle execution and transactional learning live in `solve/core/episode.py`.
The online engine supplies setup reselection, the reference fallback and complete
per-problem reporting. Frozen evaluation uses saved hierarchy tuples and learned
controllers without changing their learning state.

Use package-qualified imports and invoke the official entry points with
`python -m experiments.paper_final.MODULE`. Importing a reusable helper does not
start an experiment or change thread settings. CLI entry points configure worker
threads before numerical imports; child workers receive explicit thread settings.

## Source, evidence and generated output

The [reproduction manifest](../experiments/paper_final/reproduction/manifest.json)
records the unchanged configuration and compact-result captures. The Run 05
[frozen-input manifest](../experiments/paper_final/reproduction/matched_policy/frozen_inputs/manifest.json)
records its checkpoint, input and evidence files. Original fixed-scan hashes and
derived compact-file hashes are separate provenance records.

Fresh output belongs under `results/paper_final/04_online/` or
`results/paper_final/05_policy/`. Prepared runs capture their current source and
execution environment and refuse changes on resume. Historical dates and paths
inside accepted metadata describe original provenance; fresh execution does not
read those earlier directories.

HYPRE build/install trees and compiled interfaces are generated locally. Changes
to our wiring belong in `hypre/interfaces/` or `hypre/bindings/`; the vendored
implementation and its license notices remain preserved.

Development tools and archives are retained on the
[`online-bandit-rl`](https://github.com/EriseHe/RLTune-BoomerAMG/tree/online-bandit-rl)
and [`cleanup/sisc-repository-20261004`](https://github.com/EriseHe/RLTune-BoomerAMG/tree/cleanup/sisc-repository-20261004)
branches. The [reproduction guide](reproduction.md) describes only this submission.
