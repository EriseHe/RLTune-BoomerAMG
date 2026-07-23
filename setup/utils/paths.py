"""Repository output paths used by setup-only entry points."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_RESULTS_ROOT = REPO_ROOT / "results" / "setup"
