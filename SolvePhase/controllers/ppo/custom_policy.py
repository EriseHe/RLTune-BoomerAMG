"""Compatibility alias for the custom PPO controller policy."""

import sys as _sys

from solve.controllers.ppo import custom_policy as _implementation

_sys.modules[__name__] = _implementation
