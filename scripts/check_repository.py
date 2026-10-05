"""Check SISC source and run each test group in a fresh Python process."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_GROUPS = (
    ("problems", "problems/tests"),
    ("setup", "setup/tests"),
    ("solve and native integration", "solve/tests"),
    ("experiment infrastructure", "experiments/tests"),
    ("official paper modules 04 and 05", "experiments/paper_final"),
)


def check_source() -> None:
    files = {ROOT / "hypre/__init__.py"}
    for directory in (
        "problems",
        "setup",
        "solve",
        "experiments",
        "hypre/bindings",
        "scripts",
    ):
        files.update(
            path
            for path in (ROOT / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for path in sorted(files):
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )
        compile(tree, str(path), "exec")
    print(f"Python 3.10 syntax: {len(files)} project files passed", flush=True)
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *(str(path) for path in sorted(files))],
        cwd=ROOT,
        check=True,
    )


def run_tests() -> None:
    for name, directory in TEST_GROUPS:
        path = ROOT / directory
        if not path.is_dir() or not any(path.glob("test_*.py")):
            raise RuntimeError(f"Required test group is missing: {name} ({path})")
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
    args = parser.parse_args()
    if not args.tests_only:
        check_source()
    if not args.static_only:
        run_tests()


if __name__ == "__main__":
    main()
