"""Compatibility alias for the solve relaxation environment."""

import sys as _sys

from solve.core import amg_gym_env as _implementation

_sys.modules[__name__] = _implementation
