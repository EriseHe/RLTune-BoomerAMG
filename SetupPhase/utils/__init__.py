"""Module alias for the canonical :mod:`setup.utils` package."""

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from setup._compat import install_legacy_utility_aliases


install_legacy_utility_aliases(__name__)

del Path, _REPO_ROOT, install_legacy_utility_aliases
