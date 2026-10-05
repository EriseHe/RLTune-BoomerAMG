# Experiments

Cross-component experiment protocols live here. These workflows may import
both `setup/` and `solve/`, while generated data belongs under
`results/`.

- `paper_final/`: the two official paper experiments: Module 04 online
  autotuning and Module 05 matched-hierarchy policy evaluation.
- `joint/`: shared setup/solve composition and general experiment support.
- `diagnostics/`: component-spanning diagnostic tools and regression tests;
  these are development utilities, not additional official paper experiments.
- `archive/`: earlier protocols, development studies, and legacy workflows.

Setup-only and solve-only tests remain with their owning component.
See the [reproduction index](../docs/reproduction.md) for the accepted protocols
and required data.
