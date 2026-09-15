"""Window-free RL activation from first-attempt reliability evidence.

Implements equations (1)--(4) in docs/theory/dynamic_rl_start_and_theorem_framework.md.
The gate observes only the default-solve prefix, once per problem, before any
successful recovery can conceal a first-attempt failure. Crossing after problem
t enables RL on problem t+1. No fixed deadline or minimum window is imposed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.special import betainc, betaln, logsumexp


def _log_uniform_bernoulli_evalues(
    failures: np.ndarray, successes: np.ndarray, p0: float,
) -> np.ndarray:
    """Log of the continuous uniform-q integral for each suffix.

    Evaluate the lower incomplete beta in scaled form when its CDF could
    underflow. The continued fraction cancels the likelihood denominator
    analytically; e.g. 4999 failures still gives exactly E = 1/5000.
    This integrates cumulative likelihoods, not single-step averages.
    """
    a = np.asarray(failures, dtype=float) + 1.0
    b = np.asarray(successes, dtype=float) + 1.0
    lower = p0 < (a + 1.0) / (a + b + 2.0)
    result = np.empty_like(a)
    au, bu = a[~lower], b[~lower]
    result[~lower] = (
        betaln(au, bu) + np.log(betainc(au, bu, p0))
        - au * math.log(p0) - (bu - 1.0) * math.log1p(-p0)
    )
    if not np.any(lower):
        return result
    al, bl = a[lower], b[lower]
    qab, qap, qam = al + bl, al + 1.0, al - 1.0
    c = np.ones_like(al)
    d = 1.0 / (1.0 - qab * p0 / qap)
    h = d.copy()
    tiny = 1e-300
    for m in range(1, 257):
        m2 = 2 * m
        for aa in (
            m * (bl - m) * p0 / ((qam + m2) * (al + m2)),
            -(al + m) * (qab + m) * p0 / ((al + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            d = np.where(np.abs(d) < tiny, np.copysign(tiny, d), d)
            c = np.where(np.abs(c) < tiny, np.copysign(tiny, c), c)
            d = 1.0 / d
            change = d * c
            h *= change
        if np.all(np.abs(change - 1.0) < 2e-14):
            break
    else:
        raise ArithmeticError("Continuous activation beta fraction did not converge")
    if not np.all(np.isfinite(h) & (h > 0)):
        raise ArithmeticError("Invalid continuous activation beta fraction")
    result[lower] = math.log1p(-p0) + np.log(h) - np.log(al)
    return result


@dataclass(frozen=True)
class ReliabilityActivationSpec:
    p_bad: float
    p_good: float | None
    delta: float
    kind: str = "reliability_mixture"
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "reliability_mixture":
            if self.p_good is None or not (0.0 < self.p_good < self.p_bad < 1.0):
                raise ValueError("solve_activation requires 0 < p_good < p_bad < 1")
            if self.horizon is not None:
                raise ValueError("The legacy reliability mixture has no finite horizon")
        elif self.kind == "composite_reliability_mixture":
            if not 0.0 < self.p_bad < 1.0 or self.p_good is not None:
                raise ValueError("Composite activation requires 0 < p_bad < 1 and no p_good")
            if type(self.horizon) is not int or self.horizon < 1:
                raise ValueError("Composite activation requires a positive integer horizon")
        else:
            raise ValueError(f"Unknown solve_activation.kind: {self.kind}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("solve_activation.delta must lie strictly between 0 and 1")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReliabilityActivationSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("solve_activation must be an object")
        unknown = set(raw) - {"kind", "p_bad", "p_good", "delta", "horizon"}
        if unknown:
            raise ValueError(f"Unknown solve_activation keys: {sorted(unknown)}")
        return cls(
            kind=str(raw.get("kind", "reliability_mixture")),
            p_bad=float(raw["p_bad"]),
            p_good=None if raw.get("p_good") is None else float(raw["p_good"]),
            delta=float(raw["delta"]),
            horizon=raw.get("horizon"),
        )


class ReliabilityActivationGate:
    def __init__(self, spec: ReliabilityActivationSpec) -> None:
        self.spec = spec
        self.observations = 0
        self.failures = 0
        self.log_accumulator = -math.inf
        self.log_evalue = 0.0
        self.crossing_case: int | None = None
        if spec.p_good is not None:
            self._log_success = math.log1p(-spec.p_good) - math.log1p(-spec.p_bad)
            self._log_failure = math.log(spec.p_good) - math.log(spec.p_bad)
        self._prefix_failures = [0]
        self.log_threshold = -math.log(spec.delta)

    @property
    def active(self) -> bool:
        return self.crossing_case is not None

    @property
    def can_observe(self) -> bool:
        return not self.active and (
            self.spec.horizon is None or self.observations < self.spec.horizon
        )

    def observe(self, *, first_attempt_failed: bool) -> None:
        if not self.can_observe:
            raise RuntimeError("The activation gate must stop observing after crossing or horizon")
        self.observations += 1
        self.failures += int(first_attempt_failed)
        t = self.observations
        if self.spec.kind == "composite_reliability_mixture":
            horizon = self.spec.horizon
            failures = self.failures - np.asarray(self._prefix_failures)
            successes = np.arange(t, 0, -1) - failures
            log_suffixes = _log_uniform_bernoulli_evalues(
                failures, successes, self.spec.p_bad,
            )
            self._prefix_failures.append(self.failures)
            self.log_accumulator = float(logsumexp(log_suffixes)) - math.log(horizon)
            log_dormant = math.log1p(-t / horizon) if t < horizon else -math.inf
            self.log_evalue = float(np.logaddexp(self.log_accumulator, log_dormant))
        else:
            log_weight = -math.log(t) - math.log(t + 1)
            self.log_accumulator = float(np.logaddexp(self.log_accumulator, log_weight)) + (
                self._log_failure if first_attempt_failed else self._log_success
            )
            # Include the dormant mass sum_{k>t} omega_k=1/(t+1).
            self.log_evalue = float(np.logaddexp(self.log_accumulator, -math.log(t + 1)))
        if self.log_evalue >= self.log_threshold:
            self.crossing_case = t

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self.spec),
            "observations": self.observations,
            "first_attempt_failures": self.failures,
            "log_accumulator": (
                self.log_accumulator if self.observations else None
            ),
            "log_evalue": self.log_evalue,
            "log_threshold": self.log_threshold,
            "crossing_case": self.crossing_case,
            "first_rl_case": (
                None if self.crossing_case is None else self.crossing_case + 1
            ),
        }
