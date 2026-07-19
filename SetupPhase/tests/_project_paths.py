"""Make setup-phase modules importable during direct unittest discovery."""

from pathlib import Path
import sys


SETUP_PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SETUP_PHASE_ROOT.parent

for path in (REPO_ROOT, SETUP_PHASE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
