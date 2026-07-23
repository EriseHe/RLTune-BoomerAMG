"""Compatibility import for :mod:`setup.scripts._project_paths`."""

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from setup.scripts._project_paths import *  # noqa: F401,F403,E402
