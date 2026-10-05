# Archived paper development

These studies are research history outside the official SISC experiment tree.
The current submission uses only the
[official Module 04 and accepted Module 05 workflows](../../paper_final/README.md).

| Directory / workflow | Historical purpose |
|---|---|
| `01_numerics`, timing and stopping checks | Numerical and recovery diagnostics |
| `02_diagnostics` | Earlier fixed-policy and schedule screens |
| `03_activation` | Development studies of activation boundaries |
| `04_online` | Earlier configurations, cap/penalty studies, and retiming |
| `05_policy`, original/repeat/refresh/anchored runners | Matched-policy development before accepted Run 05 |
| `05_online_policies` | Separate solve-specific checkpoint preparation |
| `06_policy` | Separate frozen complete-method evaluation |

Python module names and launcher paths now use
`experiments.archive.paper_development`, for example:

```sh
python -m experiments.archive.paper_development.run_05_policy_repeat --help
python -m experiments.archive.paper_development.run_06_frozen_methods --help
```

[MOVED_FILES.json](MOVED_FILES.json) maps old source/configuration paths to their
archive locations. Result directories retain their historical names. Existing
protocol JSON, seeds, checkpoints, and recorded hashes have not been rewritten.

Prepared historical experiments retain strict source checks. An archived or
changed execution source is reported as a source difference, so resuming an old
measurement requires its captured source snapshot. Standalone release verifiers
remain available for checking completed data without rerunning a solver.

Reusable execution and accounting helpers are imported from the active
`experiments.paper_final.common` package. The old checkpoint decision-history
supplement remains a small internal helper in this archive; there is no new
checkpoint interface or framework.
