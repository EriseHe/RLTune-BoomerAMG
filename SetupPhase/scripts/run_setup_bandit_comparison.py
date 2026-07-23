try:
    from ._compat import delegate
except ImportError:
    from _compat import delegate

delegate("setup.scripts.run_setup_bandit_comparison", globals())
