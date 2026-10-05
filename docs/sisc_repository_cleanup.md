# SISC repository organization

The `main` tree is the submission tree for Module 04 online autotuning and
accepted Module 05 matched-hierarchy Run 05. The full organized development
tree, historical experiments, diagnostic runners, readers and archives remain
on `online-bandit-rl` and `cleanup/sisc-repository-20261004`.

## What changed

1. **Separate submission scope.** Only the official experiment configurations,
   methods, numerical support, evidence and relevant checks are present here.
   Unrelated algorithm families and executable studies are absent. The full
   development cleanup is documented in the
   [development report](https://github.com/EriseHe/RLTune-BoomerAMG/blob/online-bandit-rl/docs/repository_cleanup_20261004.md).
2. **Canonical component ownership.** PDE definitions live in `problems`, setup
   learning and candidate selection in `setup`, and solve learning and episode
   transactions in `solve`. Package imports do not depend on search-path
   mutation or load the native library during configuration inspection.
3. **Focused experiment orchestration.** The former general joint runners are
   reduced to the official methods under `experiments/paper_final/online`.
   Configuration, construction, execution, cost feedback, records, analysis and
   plotting have named modules. Historical `4k` filenames and development
   compatibility facades are removed. Public Module 04 and 05 commands remain
   under `experiments/paper_final`.
4. **Simple shared study helpers.** Durable files, verified frozen inputs,
   frozen evaluation, worker supervision and figures use ordinary modules and
   functions. No new public API or checkpoint framework was introduced. The
   accepted NPZ formats and encoder sidecars are preserved.
5. **Submission documentation and validation.** The root guide describes fresh
   builds, official studies, input validation and evidence boundaries. Package
   tests remain available through Python's standard `unittest` commands. The
   GitHub workflow, repository-check script, and associated Ruff dependency and
   settings were removed from `main` at the author's request. Build files and
   official experiment verifiers remain. The Linux wrapper preserves its literal
   loader-relative HYPRE path.

## Documentation navigation

The root README is the entry point for installation, input validation, and both
official studies. Experiment commands are collected in the reproduction guide; source
ownership and file responsibilities are collected in the repository layout.
Separate setup, solve, native-binding, theory, formal-protocol, and input-bundle
notes retain their implementation or provenance details.

Six overlapping READMEs were consolidated, reducing the project-owned count
from 14 to 8:

| Former README | Information retained in |
|---|---|
| `problems/README.md` | [Repository layout: PDE problems](repository_layout.md#pde-problems) |
| `setup/learners/README.md` | [Setup guide](../setup/README.md#linucb-implementation) |
| `solve/controllers/README.md` | [Solve guide](../solve/README.md#lstdq-implementation) |
| `experiments/paper_final/online/README.md` | [Repository layout: online experiment engine](repository_layout.md#online-experiment-engine) |
| `experiments/paper_final/04_online/README.md` | [Module 04 reproduction](reproduction.md#module-04-exact-configurations) |
| `experiments/paper_final/README.md` | [Reproduction guide](reproduction.md), with study links in the root README |

This consolidation changes documentation only. The frozen protocols, input
captures, accepted results, checkpoint files, and implementation remain intact.

## Scientific preservation

The vendored `hypre/source` tree is unchanged. Its recorded Git tree is
`ab1f65095a820e6ecfa4473218985779df654a83`. Project-owned interfaces remain in
`hypre/interfaces` and `hypre/bindings`.

The paper implementations are named `SharedLinUCB` and `RecursiveLstdqController`.
The previously separate LSTDQ numerical base and episode-cluster implementation
are merged in `solve/controllers/recursive_lstdq/controller.py`. Mean updates,
action selection, episode covariance, and saved checkpoint schemas are preserved.
LinUCB's implementation is in `setup/learners/linucb/learner.py`; development
version labels have been removed from class, factory, and current method names.
Original labels inside frozen configuration/provenance records remain unchanged
and are translated at the experiment input boundary. Unreachable Monte Carlo
episode paths and unused setup reporting helpers were removed from this submission tree.

Differential checks against the development implementation preserved actions,
learner/controller arrays, random states, method order, update order, native
calls, recovery outcomes and timer calls. They covered both PDE contexts,
construction reselection, solve fallback and unrecovered rollback. Nine further
episode probes covered native errors, nonfinite recovery and frozen evaluation.

All 42 online configuration bytes and all 47 captured configuration/result
hashes remain unchanged. The 36 distinct online input streams are captured as
canonical JSONL and replayed under their original hashes. This preserves all
180,000 matrix arguments and contexts across platform logarithm implementations.
Both compressed-file and canonical-content hashes are checked, and every
sampling setting is bound to the accepted identity. The original sampler remains
available for unpinned and unrecognized configurations. The Module 05 bundle preserves the exact accepted
checkpoint, encoder, input, job and choice bytes. Its derived compact scans
record separate hashes and reconstruct all 29,520 fixed-grid observations and
choices. Packaging was checked against the original accepted run without
changing its measurements: the extracted release verifies 8,460 evaluations,
cost arrays, schedules, coverage and fixed choices using the standard library.

This cleanup preserves the accepted recovery policy. Construction failure
allows up to three learned attempts and one default fallback; solve failure
goes directly to default recovery. A final unrecovered case rolls provisional
learning observations back. A revised failure-learning experiment would be a
separate protocol and is not part of this organization change.

## Validation record

After merging LSTDQ and removing development labels, **114 local tests** passed
without skips: problems 9, setup 14, solve/native/recovery 52, experiment
infrastructure/protocols 19, and official paper helpers 20. The existing input
tests now cover equivalence between current and captured LSTDQ settings,
including default and nondefault confidence values.

Temporary differential checks compared all 42 LinUCB method bodies, 40 LinUCB
cases and 168 LSTDQ transitions with the previous implementation. Actions,
learned arrays, uncertainty scores, rollback state, RNG state and every saved
checkpoint field matched exactly; all six accepted controller checkpoints loaded.
All three captured online suites and the frozen bundle verified. Python 3.10
syntax passed for 112 project sources, and 70 maintained local documentation
links and anchors resolved. All 130 protected evidence files and the HYPRE
source tree remained unchanged. Temporary comparison scripts and baseline
copies were removed after verification.

The initial curated tree passed **113 local tests** without skips: problems 9,
setup 14, solve/native/recovery 52, experiment infrastructure/protocols 18, and official
paper helpers 20. Python 3.10 syntax and fatal Ruff checks passed on the then-114
source files, including the subsequently removed repository-check script.
The formal six-group CLI validated without native solves; all 42 recorded streams
validated, and all six accepted LSTDQ checkpoints loaded with their encoders.
HYPRE and project interfaces built from a fresh checkout, and wheel construction
passed. The full Module 05 packaging smoke and standalone release verifier passed.

The separate development branch passed its 158 required Linux CI tests in
[run 37251317768](https://github.com/EriseHe/RLTune-BoomerAMG/actions/runs/37251317768).
Its full 421 local tests included historical diagnostics and archive checks.
Those optional historical fixtures remain platform-specific and are outside
this submission tree. Before workflow removal,
[main run 37252065358](https://github.com/EriseHe/RLTune-BoomerAMG/actions/runs/37252065358)
tested all five retained groups from a fresh Linux dependency installation/build.
A fresh pip-only macOS environment exposed a loader error in the third-party
SciPy 1.15.3 `_spropack` wheel on this host. The existing working `rl` environment
passes the local source, learner and native checks. The scientific dependency
versions are preserved; this repository does not patch SciPy binaries. The
recorded Linux run checked the fresh pip installation separately. A new macOS
installation requires a working SciPy build before the experiment packages can import.

See the [reproduction guide](reproduction.md) for fresh-run commands and timing
definitions. Existing accepted results remain distinct from any new retiming.

The project-owned license remains undecided. Upstream HYPRE license and notice
files are preserved. No manuscript or Overleaf files are part of this cleanup.
