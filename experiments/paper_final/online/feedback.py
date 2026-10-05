"""Keep native and controller costs separate in setup-learner feedback."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def as_feedback(native: Dict[str, Any], *, include_controller: bool) -> Dict[str, Any]:
    feedback = dict(native)
    controller_runtime = (
        float(native.get("infer_runtime", 0.0)) if include_controller else 0.0
    )
    native_runtime = float(native.get("native_runtime", native["runtime"]))
    reported_solve_runtime = float(native.get("solve_runtime", 0.0))
    inferred_native_solve = reported_solve_runtime
    if not np.isclose(
        float(native.get("setup_runtime", 0.0)) + reported_solve_runtime,
        native_runtime,
        rtol=1e-9,
        atol=1e-12,
    ):
        inferred_native_solve = reported_solve_runtime - controller_runtime
    native_solve_runtime = float(
        native.get("native_solve_runtime", inferred_native_solve)
    )
    feedback["native_runtime"] = native_runtime
    feedback["native_solve_runtime"] = native_solve_runtime
    feedback["infer_runtime"] = float(controller_runtime)
    feedback["runtime"] = float(native_runtime + controller_runtime)
    feedback["solve_runtime"] = float(native_solve_runtime + controller_runtime)
    return feedback


__all__ = ["as_feedback"]
