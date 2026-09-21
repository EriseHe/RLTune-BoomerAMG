"""Opt-in decision audit and near-tie ablation; ordinary learners are unchanged."""
from __future__ import annotations

import hashlib
import json
import numpy as np


class TimingSelection:
    """Raw tie rule, stable numerical ties, or calibrated reference retention.

    sigma is a pooled within-workload timing SD from a separate calibration.
    The propagated width is a heuristic under homoscedastic independent noise,
    NOT a confidence guarantee for the actual heteroscedastic timing process.
    """

    def __init__(self, mode="raw", sigma=0.0, multiplier=2.0, numeric_tolerance=1e-12):
        if mode not in {"raw", "stable", "near_tie"}:
            raise ValueError(mode)
        if any(not np.isfinite(x) or x < 0 for x in (sigma, multiplier, numeric_tolerance)):
            raise ValueError("Selection scales must be finite and nonnegative")
        self.mode, self.sigma = mode, float(sigma)
        self.multiplier, self.numeric_tolerance = float(multiplier), float(numeric_tolerance)
        self.last = {}

    def __call__(self, model, context, arms, scores):
        arms = np.asarray(arms, dtype=np.int64)
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Nonfinite setup scores")
        state_before = json.dumps(model.rng.bit_generator.state, sort_keys=True)
        best = float(np.min(scores))
        numerical = np.flatnonzero(scores <= best + self.numeric_tolerance)
        best_location = int(numerical[np.argmin(arms[numerical])])
        reference = int(model.history[-1].arm_index) if model.history else None
        matches = np.flatnonzero(arms == reference) if reference is not None else []
        width = 0.0
        reference_gap = None
        if len(matches):
            reference_location = int(matches[0])
            reference_gap = float(scores[reference_location] - best)
            if self.mode == "near_tie":
                difference = model._phi(context, reference) - model._phi(context, int(arms[best_location]))
                projected = model.A_inv @ difference
                # d' V^-1 (V-lambda I) V^-1 d, without another inverse.
                variance_factor = max(0.0, float(difference @ projected - model.l2_reg * (projected @ projected)))
                width = self.multiplier * self.sigma * np.sqrt(variance_factor)
        band = max(self.numeric_tolerance, width)
        eligible = np.flatnonzero(scores <= best + band)
        if self.mode == "raw":
            location = int(model.rng.choice(numerical))  # exact historical call
        elif len(matches) and reference_gap <= band:
            location = int(matches[0])
        else:
            location = int(eligible[np.argmin(arms[eligible])])
        ordered = np.sort(scores)
        self.last = {
            "mode": self.mode, "candidate_sha256": hashlib.sha256(arms.tobytes()).hexdigest(),
            "candidate_count": len(arms), "minimum_score": best,
            "top_two_gap": float(ordered[1] - ordered[0]) if len(scores) > 1 else None,
            "reference_arm": reference, "reference_gap": reference_gap,
            "band_sec": float(band), "selected_score_gap": float(scores[location] - best),
            "near_count": int(len(eligible)), "retained_reference": int(arms[location]) == reference,
            "rng_before": json.loads(state_before), "rng_after": model.rng.bit_generator.state,
        }
        return location
