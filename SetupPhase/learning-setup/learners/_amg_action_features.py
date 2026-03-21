from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


ACTION_FEATURE_KEYS = (
    "strong_threshold",
    "max_row_sum",
    "trunc_factor",
    "P_max_elmts",
    "agg_num_levels",
)

ACTION_FEATURE_DEFAULTS = {
    "P_max_elmts": 4.0,
    "agg_num_levels": 0.0,
}

# The original learners used scales for the first 3 continuous knobs.
# For backward compatibility, callers may still pass only those 3 values;
# the last 2 discrete knobs will use these defaults.
ACTION_FEATURE_DEFAULT_SCALES = (0.25, 0.10, 0.20, 4.0, 1.0)

ACTION_FEATURE_DIM = (
    len(ACTION_FEATURE_KEYS)  # linear
    + len(ACTION_FEATURE_KEYS)  # squares
    + (len(ACTION_FEATURE_KEYS) * (len(ACTION_FEATURE_KEYS) - 1)) // 2  # pairwise products
)


def normalize_action_scales(action_scales: Sequence[float]) -> np.ndarray:
    scales = np.asarray(list(action_scales), dtype=float).reshape(-1)
    if scales.size == 3:
        scales = np.concatenate(
            [scales, np.asarray(ACTION_FEATURE_DEFAULT_SCALES[3:], dtype=float)],
            axis=0,
        )
    if scales.size != len(ACTION_FEATURE_KEYS):
        raise ValueError(
            f"action_scales must have length 3 or {len(ACTION_FEATURE_KEYS)}"
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("action_scales must be finite and > 0")
    return scales


def action_center_from_actions(
    actions: Sequence[Dict[str, Any]],
    action_center: Optional[Dict[str, Any]],
) -> np.ndarray:
    means: Dict[str, float] = {}
    for key in ACTION_FEATURE_KEYS:
        vals = [float(a[key]) for a in actions if key in a]
        if vals:
            m = float(np.mean(np.asarray(vals, dtype=float)))
            if not np.isfinite(m):
                raise ValueError(f"non-finite mean for action key {key}")
            means[key] = m

    out = []
    for key in ACTION_FEATURE_KEYS:
        if action_center is not None and key in action_center:
            v = float(action_center[key])
        elif key in means:
            v = float(means[key])
        elif key in ACTION_FEATURE_DEFAULTS:
            v = float(ACTION_FEATURE_DEFAULTS[key])
        else:
            raise KeyError(f"Missing required action-center key: {key}")

        if not np.isfinite(v):
            raise ValueError(f"action_center[{key!r}] must be finite")
        out.append(v)

    return np.asarray(out, dtype=float)


def action_param_vector(params: Dict[str, Any], *, err_prefix: str) -> np.ndarray:
    try:
        th = float(params["strong_threshold"])
        mxrs = float(params["max_row_sum"])
        tr = float(params["trunc_factor"])
    except KeyError as e:
        raise KeyError(f"{err_prefix} action missing required key: {e}") from e

    p_max = float(params.get("P_max_elmts", ACTION_FEATURE_DEFAULTS["P_max_elmts"]))
    agg_nl = float(params.get("agg_num_levels", ACTION_FEATURE_DEFAULTS["agg_num_levels"]))
    out = np.asarray([th, mxrs, tr, p_max, agg_nl], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("action parameters must be finite")
    return out


def poly2_features(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float).reshape(-1)
    if x.size != len(ACTION_FEATURE_KEYS):
        raise ValueError(f"expected {len(ACTION_FEATURE_KEYS)} action values, got {x.size}")

    feats = [x, x * x]
    cross_terms = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            cross_terms.append(x[i] * x[j])
    feats.append(np.asarray(cross_terms, dtype=float))
    return np.concatenate(feats, axis=0)

