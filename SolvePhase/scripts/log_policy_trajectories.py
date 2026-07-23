"""Compatibility entry point for :mod:`solve.scripts.log_policy_trajectories`."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("solve.scripts.log_policy_trajectories", run_name="__main__")
else:
    import sys as _sys

    from solve.scripts import log_policy_trajectories as _implementation

    _sys.modules[__name__] = _implementation
