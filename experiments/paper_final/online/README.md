# Online experiment support

The public Module 04 commands are `experiments.paper_final.run_04_online` and
`experiments.paper_final.analyze_04_online`. This package implements their
configuration, execution, provenance, and reporting. It supports the official
Default, LinUCB, and LinUCB–LSTDQ methods only.

| Module | Responsibility |
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

Learning algorithms remain under `setup/` and `solve/`; this package composes
them and owns experiment policy. Imports do not run experiments or change
process settings. Run commands from the repository root with `python -m`.

Construction failure permits three learned setup attempts total, then one
default fallback. Solve nonconvergence triggers default recovery directly.
The accepted final-unrecovered protocol rolls provisional observations back.
Cleanup preserves these rules; it does not introduce a revised failure policy.

The [reproduction guide](../../../docs/reproduction.md) documents full commands,
frozen inputs, timing boundaries and output requirements.
