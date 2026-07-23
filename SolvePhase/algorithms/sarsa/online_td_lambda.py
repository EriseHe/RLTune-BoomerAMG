"""Compatibility alias for the online TD(lambda) implementation."""

import sys as _sys

from solve.controllers.sarsa import online_td_lambda as _implementation

_sys.modules[__name__] = _implementation
