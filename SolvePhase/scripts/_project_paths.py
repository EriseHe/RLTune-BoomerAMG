"""Compatibility alias for solve script path setup."""

import sys as _sys

from solve.scripts import _project_paths as _implementation

_sys.modules[__name__] = _implementation
