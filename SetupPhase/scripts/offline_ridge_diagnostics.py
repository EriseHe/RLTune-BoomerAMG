try:
    from ._compat import delegate
except ImportError:
    from _compat import delegate

delegate("setup.scripts.offline_ridge_diagnostics", globals())
