"""Diagnostic entry point for the frozen-setup TD experiment."""

from experiments.joint.solve_control.online_td_experiment_common import main


if __name__ == "__main__":
    main()
if __package__ in {None, ""}:
    import _project_paths  # noqa: F401
