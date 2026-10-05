"""Explicit process setup for reproducible, single-thread experiment commands.

This module uses only the standard library and has no import-time side effects.
Call ``configure_single_thread`` before importing NumPy or the native bindings
in an executable entry point. Use ``single_thread_environment`` for workers.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping


THREAD_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def configure_single_thread() -> None:
    """Pin numerical libraries for a CLI process before loading them."""
    os.environ.update({key: "1" for key in THREAD_KEYS})


def single_thread_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a worker environment without changing the caller's environment."""
    result = dict(os.environ if environment is None else environment)
    result.update({key: "1" for key in THREAD_KEYS})
    return result


def prevent_sleep() -> subprocess.Popen | None:
    """Start optional macOS sleep protection; other platforms need no helper."""
    executable = Path("/usr/bin/caffeinate")
    if sys.platform != "darwin" or not executable.is_file() or not os.access(executable, os.X_OK):
        return None
    try:
        return subprocess.Popen([str(executable), "-is", "-w", str(os.getpid())])
    except OSError:
        # Sleep protection is optional and must not prevent an experiment.
        return None


def stop_sleep_prevention(process: subprocess.Popen | None) -> None:
    """Release optional sleep protection and reap its child process."""
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
