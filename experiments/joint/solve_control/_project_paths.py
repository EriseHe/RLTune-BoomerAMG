"""Make shared setup and solve modules importable from joint entry points."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_PHASE_ROOT = REPO_ROOT / "SetupPhase"
SETUP_SCRIPTS_DIR = SETUP_PHASE_ROOT / "scripts"
SOLVE_CORE_DIR = REPO_ROOT / "SolvePhase" / "core"
DIAGNOSTICS_DIR = REPO_ROOT / "experiments" / "diagnostics" / "solve_control"

for path in (REPO_ROOT, SOLVE_CORE_DIR, SETUP_PHASE_ROOT, SETUP_SCRIPTS_DIR, DIAGNOSTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
