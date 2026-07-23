"""Canonical setup-phase package."""

from .registry import *  # noqa: F401,F403
from .registry import __all__
from ._compat import install_checkpoint_aliases


install_checkpoint_aliases()

del install_checkpoint_aliases
