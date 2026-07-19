"""Make shared project modules importable from solve diagnostics."""

from pathlib import Path
import sys


DIAGNOSTICS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
JOINT_SOLVE_DIR = REPO_ROOT / "experiments" / "joint" / "solve_control"
SOLVE_CORE_DIR = REPO_ROOT / "SolvePhase" / "core"
SETUP_PHASE_ROOT = REPO_ROOT / "SetupPhase"
SETUP_SCRIPTS_DIR = SETUP_PHASE_ROOT / "scripts"

for path in (
    REPO_ROOT,
    DIAGNOSTICS_DIR,
    JOINT_SOLVE_DIR,
    SOLVE_CORE_DIR,
    SETUP_PHASE_ROOT,
    SETUP_SCRIPTS_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
