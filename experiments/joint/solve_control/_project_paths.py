"""Make shared setup and solve modules importable from joint entry points."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTICS_DIR = REPO_ROOT / "experiments" / "diagnostics" / "solve_control"

for path in (REPO_ROOT, DIAGNOSTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
