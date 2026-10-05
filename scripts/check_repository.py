"""Run fatal source checks and each unittest group in a fresh Python process.

Use --static-only before building native interfaces, or --tests-only afterward.
Development diagnostics and archived paper tests are included when present.
Use --core-only to limit tests to the five required groups used by Linux CI.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path("experiments/archive/paper_development")
REQUIRED_TEST_GROUPS = (
    ("problems", Path("problems/tests")),
    ("setup", Path("setup/tests")),
    ("solve and native integration", Path("solve/tests")),
    ("experiment infrastructure", Path("experiments/tests")),
    ("official paper modules 04 and 05", Path("experiments/paper_final")),
)


def source_files(archive: Path) -> list[Path]:
    """Inspect project source, including the maintained paper-development archive."""
    files = {ROOT / "hypre/__init__.py"}
    for directory in (
        "problems",
        "setup",
        "solve",
        "experiments",
        "hypre/bindings",
        "scripts",
    ):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path.is_relative_to(ROOT / "experiments/archive"):
                continue
            files.add(path)
    if archive.is_dir():
        files.update(
            path for path in archive.rglob("*.py") if "__pycache__" not in path.parts
        )
    return sorted(path for path in files if path.is_file())


def check_source(archive: Path) -> None:
    files = source_files(archive)
    for path in files:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )
        compile(tree, str(path), "exec")
    print(f"Python 3.10 syntax: {len(files)} project files passed", flush=True)
    # Explicit files include the maintained archive despite the general archive exclusion.
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *(str(path) for path in files)],
        cwd=ROOT,
        check=True,
    )


def run_tests(archive: Path, *, core_only: bool = False) -> None:
    groups = [(name, ROOT / path, True) for name, path in REQUIRED_TEST_GROUPS]
    if not core_only:
        groups.extend(
            (
                (
                    "development diagnostics",
                    ROOT / "experiments/diagnostics/solve_control",
                    False,
                ),
                ("archived paper development", archive, False),
            )
        )
    for name, path, required in groups:
        if not path.is_dir() or not any(path.glob("test_*.py")):
            if required:
                raise RuntimeError(f"Required test group is missing: {name} ({path})")
            print(
                f"Omitting {name}: test directory is absent from this checkout",
                flush=True,
            )
            continue
        print(f"\nTesting {name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(path),
                "-p",
                "test_*.py",
            ],
            cwd=ROOT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--static-only", action="store_true")
    phase.add_argument("--tests-only", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only required core/native, infrastructure and official paper test groups",
    )
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    archive = (ROOT / args.archive_dir).resolve()
    if not args.tests_only:
        check_source(archive)
    if not args.static_only:
        run_tests(archive, core_only=args.core_only)


if __name__ == "__main__":
    main()
