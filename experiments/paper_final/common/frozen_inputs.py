"""Execution and verification helpers for the official studies."""
from __future__ import annotations
from .artifacts import ROOT

def execution_source_paths():
    """Capture current submission source, excluding vendor and data artifacts."""
    packages = (
        "experiments/paper_final",
        "experiments/paper_final/online",
        "problems",
        "setup",
        "solve",
        "hypre/bindings",
    )
    paths = {
        str(path.relative_to(ROOT))
        for package in packages
        for path in (ROOT / package).rglob("*.py")
    }
    paths.add("experiments/runtime.py")
    interfaces = ROOT / "hypre/interfaces"
    paths.update(
        str(p.relative_to(ROOT))
        for p in interfaces.iterdir()
        if p.is_file() and (p.suffix in (".c", ".h") or p.name == "Makefile")
    )
    paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob("requirements*.txt"))
    if (ROOT / "pyproject.toml").is_file():
        paths.add("pyproject.toml")
    return paths
