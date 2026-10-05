# Repository cleanup, October 4, 2026

The development branch preserves the full research tree. The SISC `main` branch
is curated separately for official Module 04 online autotuning and Module 05
matched-hierarchy evaluation; unrelated studies, runners, data, and archives
belong on the development branch.

## Implemented changes

| Step | Result |
|---|---|
| 1. Submission and reproduction contents | Added a reproduction index, component ownership guide, and 47 unchanged captured configuration/numerical records with SHA-256 hashes. Large local outputs and temporary files are excluded. |
| 2. Imports and portable startup | Replaced 256 bare experiment imports with canonical package imports; native loading is deferred until execution. CLI thread setup and optional sleep protection are explicit. Added package metadata and pinned dependencies. |
| 3. Shared code and long entry points | Shared controller episodes moved to `solve/core/episode.py`, common TD settings to `solve/controllers/common/td_config.py`; action spaces, setup branches, native evaluation, and study evaluation have separate owners. Three long CLI functions now delegate preparation, execution, and reporting to named private helpers. |
| 4. Paper infrastructure | Only official 04/05 remain active. Moved 129 development files to the archive. Shared frozen execution, durable I/O, statistics, worker supervision, and figure helpers have canonical modules. Historical numerical source and current rendering source have separate provenance records. |
| 5. Validation | Added a small check runner and Linux CI with pinned tools, out-of-source native builds, and isolated test processes. Development tests use explicit archived fixtures instead of assuming the official default is a historical suite. |

No new checkpoint API framework was introduced. Project licensing remains
undecided; existing third-party notices remain in place.

## Verification

- All **420 tests** pass: problems 14, setup 30, solve/native 87,
  infrastructure 10, official paper 16, diagnostics 209, and archived paper 54.
- Python 3.10 syntax compilation and fatal Ruff checks pass on 314 project files.
  Wheel construction and the check runner's required/optional group rules pass.
- The 40 extracted experiment definitions and action defaults match the original
  AST. Shared episode comparisons match outputs, learner state, random state,
  and mocked timings across 28 success/failure/recovery cases. All three refactored
  CLI executions match the original mocked call order and accounting.
- Formatting preserves the complete AST of all 28 formatted active/helper files.
- All 47 captured evidence hashes still match.
- HYPRE and the project interfaces build successfully from a clean isolated
  checkout. The vendored `hypre/source` Git tree remains
  `ab1f65095a820e6ecfa4473218985779df654a83`.

These checks verify structural and behavioral preservation. The cleanup does
not replace accepted timings with measurements from reorganized code.

## Reproduction boundaries

Existing prepared experiments retain strict source hashes. Changed or missing
historical execution paths are rejected; use the captured numerical source for
an exact historical rerun. A fresh experiment records the current source and
runtime environment separately. Regenerated figures capture current rendering
helpers without overwriting historical execution snapshots.

See the [reproduction index](reproduction.md) for accepted artifacts and commands.
