"""Compatibility entry point for :mod:`solve.scripts.gym_test`."""

import sys as _sys

from solve.scripts import gym_test as _implementation

_sys.modules[__name__] = _implementation
