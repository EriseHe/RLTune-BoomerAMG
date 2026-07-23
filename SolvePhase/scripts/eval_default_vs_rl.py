"""Compatibility entry point for :mod:`solve.scripts.eval_default_vs_rl`."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("solve.scripts.eval_default_vs_rl", run_name="__main__")
else:
    import sys as _sys

    from solve.scripts import eval_default_vs_rl as _implementation

    _sys.modules[__name__] = _implementation
