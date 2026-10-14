# Repository layout

The [official registry](../experiments/paper_final/reproduction/manifest.json)
selects the complete paper datasets. See [reproduction](reproduction.md) for
verification and data definitions.

| Path | Responsibility |
|---|---|
| `problems/` | PDE definitions, contexts, and input streams |
| `setup/` | LinUCB, setup spaces, candidate schedules, and checkpoints |
| `solve/` | Recursive LSTDQ, cycle execution, and recovery accounting |
| `hypre/source/` | Vendored HYPRE and upstream license notices |
| `hypre/interfaces/`, `hypre/bindings/` | Project native interface and Python bindings |
| `experiments/paper_final/online/` | Paired online execution and recorded outcomes |
| `experiments/paper_final/matched_policy/` | Fresh fixed-weight scans and frozen-policy evaluation |
| `experiments/paper_final/reproduction/` | Official compact configurations, summaries, audits, paper assets, and manifests |
| `data/official/module04/` | Complete online study, six seeds for each of six problem/grid combinations |
| `data/official/module05/` | Complete matched-policy study with six frozen checkpoints |
| `data/official/module06/` | Original six-seed Type 2 variable-flow study |
| `data/official/runtime/` | Frozen execution source and original native libraries |
| `data/archive/` | All historical or unselected experimental data and exports; excluded from Git |
| `tools/official_data.py` | Explicit official-path resolution without archive discovery |
| `tools/verify_official_data.py` | Publication and optional full local data verification |
| `tools/paper/` | Current paper figure/table readers and publishers |
| `results/` | Fresh, unselected output; excluded from Git |

The execution-source manifest identifies the numerical files used by the official
runs. Organization and publication helpers are separate from that frozen numerical
source. Raw data, runtime binaries, and archives are local-only; the publication
bundle and manifests are suitable for version control.
