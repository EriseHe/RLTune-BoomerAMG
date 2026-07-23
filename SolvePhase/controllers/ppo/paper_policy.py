"""Compatibility alias for the reference-paper PPO controller policy."""

import sys as _sys

from solve.controllers.ppo import paper_policy as _implementation

_sys.modules[__name__] = _implementation
