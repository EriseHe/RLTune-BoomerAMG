"""Delegate a historical command module to its canonical implementation."""

from importlib import import_module
from pathlib import Path
import runpy
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def delegate(module_name: str, namespace: dict[str, Any]) -> None:
    if namespace["__name__"] == "__main__":
        runpy.run_module(module_name, run_name="__main__")
        return
    module = import_module(module_name)
    names = getattr(
        module,
        "__all__",
        tuple(name for name in vars(module) if not name.startswith("_")),
    )
    namespace.update({name: getattr(module, name) for name in names})
