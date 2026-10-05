"""Verify the free period-two weighted minimax grid; no PDE timings are read."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "docs/theory/period_two_minimax_verification_20260928.json"


def extrema(a, b):
    total, product = a + b, a * b
    discriminant = mp.sqrt(9 * total**2 - 20 * product)
    stationary = [
        (3 * total - discriminant) / (10 * product),
        (3 * total + discriminant) / (10 * product),
    ]
    interior = [t for t in stationary if 0 < t < 1]
    candidates = [mp.mpf(0), *interior, mp.mpf(1)]
    values = [mp.sqrt(t) * (1 - a * t) * (1 - b * t) for t in candidates]
    return max(v * v for v in values), candidates, values


def evaluate():
    mp.mp.dps = 80
    grid = [mp.mpf(20 + i) / 20 for i in range(41)]
    ranked = sorted(
        (extrema(a, b)[0], a, b) for i, a in enumerate(grid) for b in grid[: i + 1]
    )
    assert len(ranked) == 861
    assert ranked[0][1:] == (mp.mpf("2.85"), mp.mpf("1.10"))
    assert ranked[1][0] - ranked[0][0] > mp.mpf("0.0005")
    exact_a, exact_b = 2 + 2 / mp.sqrt(5), 2 - 2 / mp.sqrt(5)
    tolerance = mp.mpf("1e-70")
    assert abs(exact_a + exact_b - 4) < tolerance
    assert abs(exact_a * exact_b - mp.mpf(16) / 5) < tolerance
    assert abs(extrema(exact_a, exact_b)[0] - mp.mpf(1) / 25) < tolerance
    points = [(3 - mp.sqrt(5)) / 8, (3 + mp.sqrt(5)) / 8, mp.mpf(1)]
    for t, sign in zip(points, (1, -1, 1)):
        assert (
            abs(mp.sqrt(t) * (1 - exact_a * t) * (1 - exact_b * t) - mp.mpf(sign) / 5)
            < tolerance
        )

    # Exact rational checks of the two perturbation identities in the note.
    rounded_a, grid_a, low = Fraction(29, 10), Fraction(57, 20), Fraction(11, 10)
    assert rounded_a + low == 4
    assert rounded_a * low - Fraction(16, 5) == Fraction(-1, 100)
    assert rounded_a - grid_a == Fraction(1, 20)
    assert (grid_a - rounded_a) * low == Fraction(-11, 200)

    def number(value):
        return mp.nstr(value, 75)

    comparisons = []
    for label, a, b in (
        ("continuous_exact", exact_a, exact_b),
        ("componentwise_rounded", mp.mpf("2.90"), mp.mpf("1.10")),
        ("grid_minimizer", mp.mpf("2.85"), mp.mpf("1.10")),
        ("historical_anchored_grid", mp.mpf("2.60"), mp.mpf(1)),
    ):
        eta, locations, values = extrema(a, b)
        comparisons.append(
            {
                "name": label,
                "high": number(a),
                "low": number(b),
                "eta": number(eta),
                "locations": list(map(number, locations)),
                "weighted_values": list(map(number, values)),
            }
        )
    rounded_eta = extrema(mp.mpf("2.9"), mp.mpf("1.1"))[0]
    return {
        "passed": True,
        "decimal_precision": mp.mp.dps,
        "mpmath_version": mp.__version__,
        "objective": "max_{0<=t<=1} t (1-a*t)^2 (1-b*t)^2",
        "grid": "{1+j/20: j=0,...,40}",
        "unordered_pairs_checked": len(ranked),
        "grid_minimizers_ordered": [["2.85", "1.10"], ["1.10", "2.85"]],
        "method": "Endpoints plus analytic interior stationary points of sqrt(t)*p(t); polynomial zeros have objective zero",
        "selection_uses_pde_timings": False,
        "continuous_result": "Unique polynomial 1-4*t+(16/5)*t^2, eta=1/25; coefficients 2 +/- 2/sqrt(5)",
        "exact_rational_perturbation_checks": True,
        "comparisons": comparisons,
        "runner_up_gap": number(ranked[1][0] - ranked[0][0]),
        "surrogate_reduction_grid_vs_rounded_pct": number(
            100 * (1 - ranked[0][0] / rounded_eta)
        ),
        "ranked_grid": [
            {"high": number(a), "low": number(b), "eta": number(eta)}
            for eta, a, b in ranked
        ],
        "limits": "High precision finite verification, not an interval-arithmetic certificate; objective concerns normalized SPD smoothing, not multilevel completion time or phase preference",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "decimal_precision",
                    "unordered_pairs_checked",
                    "grid_minimizers_ordered",
                    "runner_up_gap",
                    "surrogate_reduction_grid_vs_rounded_pct",
                )
            },
            indent=2,
        )
    )
