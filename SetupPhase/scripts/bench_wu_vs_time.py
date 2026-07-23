try:
    from ._compat import delegate
except ImportError:
    from _compat import delegate

delegate("setup.scripts.bench_wu_vs_time", globals())
