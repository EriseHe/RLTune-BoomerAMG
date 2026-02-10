"""
HYPRE library loader for BoomerAMG solver.

Handles loading the HYPRE shared library (DLL on Windows) via ctypes.

Configuration via environment variables:
  - HYPRE_LIBHYPRE: Full path to libHYPRE.* (recommended)
  - HYPRE_DIR / HYPRE_PREFIX: Directory containing HYPRE build/install
  - HYPRE_DLL_DIRS: Additional DLL search directories (Windows, ';'-separated)
"""

import os
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_DLL_DIR_HANDLES = []
_ADDED_DLL_DIRS = set()  # Track which directories have been added
_CACHED_LIBRARY = None   # Cache loaded library


def _add_dll_directory(path: str) -> None:
    """Add a directory to the DLL search path on Windows (only once per path)."""
    if not path:
        return
    if not os.path.isdir(path):
        return
    # Skip if already added
    if path in _ADDED_DLL_DIRS:
        return
    _ADDED_DLL_DIRS.add(path)
    
    if hasattr(os, "add_dll_directory"):
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
    else:
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _split_path_list(value: str) -> List[str]:
    """Split a path list string into individual paths."""
    if not value:
        return []

    parts = [p.strip() for p in value.split(os.pathsep) if p.strip()]
    if os.pathsep != ";" and ";" in value:
        parts.extend([p.strip() for p in value.split(";") if p.strip()])

    seen = set()
    unique = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


@dataclass(frozen=True)
class HypreLibrarySpec:
    """Specification for a loaded HYPRE library."""
    lib_path: str
    dll_dirs: List[str]


def _candidate_library_names() -> List[str]:
    """Get candidate library file names for the current platform."""
    if os.name == "nt":
        return ["HYPRE.dll", "libHYPRE.dll", "hypre.dll"]
    if sys.platform == "darwin":
        return ["libHYPRE.dylib"]
    return ["libHYPRE.so"]


def _find_library_under(root: Path) -> Optional[str]:
    """Search for HYPRE library under a root directory."""
    if root.is_file():
        return str(root)

    library_names = _candidate_library_names()
    search_subdirs = [
        Path("."),
        Path("bin"),
        Path("lib"),
        Path("src/hypre/bin"),
        Path("src/hypre/lib"),
        Path("build/bin"),
        Path("build/lib"),
        Path("build/src/hypre/bin"),
        Path("build/src/hypre/lib"),
    ]
    for subdir in search_subdirs:
        for name in library_names:
            candidate = root / subdir / name
            if candidate.is_file():
                return str(candidate)
    return None


def resolve_hypre_library() -> HypreLibrarySpec:
    """
    Resolve the HYPRE library path and dependent DLL directories.

    Resolution order:
    1) HYPRE_LIBHYPRE env var (absolute path to libHYPRE.*)
    2) HYPRE_DIR/HYPRE_PREFIX env var (directory to search for libHYPRE.*)

    Returns
    -------
    HypreLibrarySpec
        Contains lib_path and dll_dirs

    Raises
    ------
    RuntimeError
        If HYPRE library cannot be located
    """
    env_path = os.environ.get("HYPRE_LIBHYPRE")
    if env_path:
        lib_path = env_path
        dll_dirs = [str(Path(lib_path).parent)]
        dll_dirs.extend(_split_path_list(os.environ.get("HYPRE_DLL_DIRS", "")))
        return HypreLibrarySpec(lib_path=lib_path, dll_dirs=dll_dirs)

    for env_name in ("HYPRE_DIR", "HYPRE_PREFIX", "HYPRE_ROOT"):
        root = os.environ.get(env_name)
        if not root:
            continue
        lib_path = _find_library_under(Path(root))
        if not lib_path:
            continue
        dll_dirs = [str(Path(lib_path).parent)]
        dll_dirs.extend(_split_path_list(os.environ.get("HYPRE_DLL_DIRS", "")))
        return HypreLibrarySpec(lib_path=lib_path, dll_dirs=dll_dirs)

    raise RuntimeError(
        "Could not locate HYPRE.\n\n"
        "Set one of:\n"
        "  - HYPRE_LIBHYPRE = full path to libHYPRE.* / HYPRE.dll\n"
        "  - HYPRE_DIR / HYPRE_PREFIX = directory containing a HYPRE build/install\n\n"
        "If libHYPRE has extra DLL dependencies (Windows), also set:\n"
        "  - HYPRE_DLL_DIRS = additional DLL search dirs (separated by ';')"
    )


def load_hypre():
    """
    Load the HYPRE library via ctypes.
    
    The library is cached after first load to avoid repeated loading overhead.

    Returns
    -------
    tuple
        (ctypes.CDLL, HypreLibrarySpec)

    Raises
    ------
    RuntimeError
        If HYPRE library cannot be found or loaded
    """
    global _CACHED_LIBRARY
    
    if _CACHED_LIBRARY is not None:
        return _CACHED_LIBRARY
    
    spec = resolve_hypre_library()
    for d in spec.dll_dirs:
        _add_dll_directory(d)

    lib = ctypes.CDLL(spec.lib_path)
    _CACHED_LIBRARY = (lib, spec)
    return lib, spec
