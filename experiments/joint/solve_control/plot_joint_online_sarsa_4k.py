from __future__ import annotations

"""Compatibility entry point for the former plot-module name.

New code should import :mod:`joint_experiment_plotting`.  Keeping this thin
adapter preserves historical commands without making current experiments
depend on a SARSA-named module.
"""

import joint_experiment_plotting as _implementation


generate_plots = _implementation.generate_plots
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
