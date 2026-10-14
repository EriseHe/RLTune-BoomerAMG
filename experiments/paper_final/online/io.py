"""JSON output and runtime summaries used by the official online study."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import numpy as np


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _runtime_with_overhead(row: Dict[str, Any]) -> float:
    return float(row["runtime"]) + float(row.get("infer_runtime", 0.0))


def _solve_with_overhead(row: Dict[str, Any]) -> float:
    return float(row["solve_runtime"]) + float(row.get("infer_runtime", 0.0))


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mean_optional = lambda field: float(
        np.mean([float(row.get(field, 0.0)) for row in rows])
    )
    return {
        "cases": int(len(rows)),
        "failed_count": int(sum(bool(row.get("failed", False)) for row in rows)),
        "mean_runtime_sec": float(np.mean([float(row["runtime"]) for row in rows])),
        "mean_runtime_with_overhead_sec": float(
            np.mean([_runtime_with_overhead(row) for row in rows])
        ),
        "mean_setup_runtime_sec": float(
            np.mean([float(row["setup_runtime"]) for row in rows])
        ),
        "mean_solve_runtime_sec": float(
            np.mean([float(row["solve_runtime"]) for row in rows])
        ),
        "mean_solve_with_overhead_sec": float(
            np.mean([_solve_with_overhead(row) for row in rows])
        ),
        "mean_policy_overhead_sec": float(
            np.mean([float(row.get("infer_runtime", 0.0)) for row in rows])
        ),
        "mean_feature_runtime_sec": mean_optional("feature_runtime"),
        "mean_decision_runtime_sec": mean_optional("decision_runtime"),
        "mean_update_runtime_sec": mean_optional("update_runtime"),
        "mean_iterations": float(np.mean([int(row["iterations"]) for row in rows])),
    }
