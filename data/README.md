# Experiment data

The official paper runs are `official/module04/`, `official/module05/`, and
`official/module06/`. The shared frozen runtime is `official/runtime/`.

All other historical experiment data and exports are in `archive/20261014/`.
Both directories are excluded from Git. The versioned publication bundle and
explicit selection live in `../experiments/paper_final/reproduction/`.

Run `python tools/verify_official_data.py --full` from the repository root to
verify the local official data without executing an experiment.
