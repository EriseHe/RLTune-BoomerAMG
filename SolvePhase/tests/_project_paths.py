"""Compatibility alias for solve test path setup."""

import sys as _sys

from solve.tests import _project_paths as _implementation

_sys.modules[__name__] = _implementation
