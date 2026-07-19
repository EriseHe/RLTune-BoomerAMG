"""Repository paths shared by solve-phase command-line entry points."""

from pathlib import Path
import sys


SOLVE_PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOLVE_PHASE_ROOT.parent
CORE_DIR = SOLVE_PHASE_ROOT / "core"

for path in (REPO_ROOT, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
