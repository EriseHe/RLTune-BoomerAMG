from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping
from problems.registry import DEFAULT_SETUP_CONTEXT, normalize_setup_context_mode
from setup.registry import COMPOSABLE_SETUP_KINDS, ONLINE_SETUP_KINDS
from solve.controllers.common import CANONICAL_PROBLEM_CONTEXT, PROBLEM_CONTEXT_MODES
from solve.registry import COMPOSABLE_SOLVE_KINDS, ONLINE_SOLVE_KINDS, normalize_solve_kind

METHOD_KEYS = {
    "id",
    "setup",
    "setup_space",
    "setup_context",
    "candidate_sampling",
    "solve",
    "fixed_weight",
    "seed_offset",
    "setup_warmup_cases",
    "solve_activation_case",
    "solve_activation",
    "solve_tolerance",
    "solve_context",
}
CANDIDATE_SAMPLING_METHODS = ("uniform512", "structured512")
CONTEXT_DISPLAY_LABELS = {
    "default": "8D",
    "canonical": "8D",
    "canonical_no_c_mean": "7D",
    "diffusion3d": "4D",
}


@dataclass(frozen=True)
class ComposableMethodSpec:
    """One setup/solve branch selected by a high-level experiment config."""

    name: str
    setup_kind: str
    solve_kind: str
    setup_space: str | None = None
    setup_context: str = DEFAULT_SETUP_CONTEXT
    candidate_sampling: str = "uniform512"
    fixed_weight: float | None = None
    seed_offset: int = 0
    setup_warmup_cases: int = 0
    solve_activation_case: int = 0
    solve_tolerance: float | None = None
    solve_context: str = CANONICAL_PROBLEM_CONTEXT
    solve_activation: None = None

    @property
    def family(self) -> str:
        if self.setup_kind == "default":
            return "default_setup"
        if self.solve_kind == "default":
            return "default"
        return "recursive_lstdq_lcb"

    @property
    def label(self) -> str:
        setup = {"default": "Default setup", "linucb": "LinUCB"}[self.setup_kind]
        if self.setup_context != DEFAULT_SETUP_CONTEXT:
            setup = f"{setup} ({CONTEXT_DISPLAY_LABELS.get(self.setup_context, self.setup_context)})"
        solve = {
            "default": "default solve",
            "recursive_lstdq": "Recursive LSTDQ",
        }[self.solve_kind]
        if self.solve_activation_case:
            solve = f"{solve}; activate={self.solve_activation_case}"
        if self.solve_tolerance is not None:
            solve = f"{solve}; tol={float(self.solve_tolerance):g}"
        if self.solve_context not in {CANONICAL_PROBLEM_CONTEXT, self.setup_context}:
            solve = f"{solve}; context={CONTEXT_DISPLAY_LABELS.get(self.solve_context, self.solve_context)}"
        return f"{setup} + {solve}"

    def resolve_solve_tolerance(self, default: float) -> float:
        """Return this branch's stopping tolerance or the experiment default."""
        tolerance = (
            float(default)
            if self.solve_tolerance is None
            else float(self.solve_tolerance)
        )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("solve tolerance must be finite and positive")
        return tolerance

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ComposableMethodSpec":
        solve_kind = normalize_solve_kind(raw.get("solve"))
        if raw.get("setup") == "default" and solve_kind != "default":
            raise ValueError("The official Default branch uses default setup and solve")
        if raw.get("setup") not in {"default", "linucb"} or solve_kind not in {
            "default",
            "recursive_lstdq",
        }:
            raise ValueError(
                "The submission supports only Default, LinUCB, and LinUCB–LSTDQ"
            )
        unknown = set(raw) - METHOD_KEYS
        if unknown:
            raise ValueError(
                f"method {raw.get('id', '<unknown>')!r} has unknown keys: {sorted(unknown)}"
            )
        name = str(raw["id"])
        if not name.strip():
            raise ValueError("method id cannot be empty")
        if any((character in name for character in "/\\")):
            raise ValueError("method ids cannot contain path separators")
        setup_kind = str(raw["setup"])
        if setup_kind not in COMPOSABLE_SETUP_KINDS:
            raise ValueError(f"Unsupported setup kind: {setup_kind}")
        if solve_kind not in COMPOSABLE_SOLVE_KINDS:
            raise ValueError(f"Unsupported solve kind: {solve_kind}")
        setup_space_raw = raw.get("setup_space")
        setup_space = None if setup_space_raw is None else str(setup_space_raw).strip()
        setup_context = normalize_setup_context_mode(
            raw.get("setup_context", DEFAULT_SETUP_CONTEXT)
        )
        candidate_sampling = (
            str(raw.get("candidate_sampling", "uniform512"))
            .strip()
            .lower()
            .replace("-", "")
        )
        if setup_space is not None:
            if not setup_space:
                raise ValueError(f"method {name!r} setup_space cannot be empty")
            if setup_kind not in ONLINE_SETUP_KINDS:
                raise ValueError(
                    f"method {name!r} can only set setup_space for a setup bandit"
                )
            if candidate_sampling not in CANDIDATE_SAMPLING_METHODS:
                raise ValueError(
                    f"method {name!r} candidate_sampling must be uniform512 or structured512"
                )
        elif "candidate_sampling" in raw:
            raise ValueError(
                f"method {name!r} requires setup_space when setting candidate_sampling"
            )
        if "setup_context" in raw and setup_kind not in ONLINE_SETUP_KINDS:
            raise ValueError(
                f"method {name!r} can only set setup_context for a setup bandit"
            )
        if setup_context != DEFAULT_SETUP_CONTEXT and setup_space is None:
            raise ValueError(
                f"method {name!r} requires setup_space when setting a non-default setup_context"
            )
        fixed_weight = None
        if solve_kind == "fixed":
            if "fixed_weight" not in raw:
                raise ValueError(f"method {name!r} requires fixed_weight")
            fixed_weight = float(raw["fixed_weight"])
            if not math.isfinite(fixed_weight) or fixed_weight <= 0.0:
                raise ValueError(
                    f"method {name!r} fixed_weight must be finite and positive"
                )
        seed_offset = int(raw.get("seed_offset", 0))
        setup_warmup_cases = int(raw.get("setup_warmup_cases", 0))
        if setup_warmup_cases < 0:
            raise ValueError(f"method {name!r} setup_warmup_cases must be non-negative")
        if setup_warmup_cases and setup_kind not in ONLINE_SETUP_KINDS:
            raise ValueError(
                f"method {name!r} can only warm up an online setup learner"
            )
        solve_activation_case = int(raw.get("solve_activation_case", 0))
        if solve_activation_case < 0:
            raise ValueError(
                f"method {name!r} solve_activation_case must be non-negative"
            )
        if solve_activation_case and solve_kind in {"default", "fixed"}:
            raise ValueError(
                f"method {name!r} cannot delay a non-learning solve policy"
            )
        if raw.get("solve_activation") is not None:
            raise ValueError(
                "The official study uses a fixed solve activation boundary"
            )
        solve_activation = None
        solve_tolerance_raw = raw.get("solve_tolerance")
        solve_tolerance = (
            None if solve_tolerance_raw is None else float(solve_tolerance_raw)
        )
        if solve_tolerance is not None and (
            not math.isfinite(solve_tolerance) or solve_tolerance <= 0.0
        ):
            raise ValueError(
                f"method {name!r} solve_tolerance must be finite and positive"
            )
        solve_context = (
            str(raw.get("solve_context", CANONICAL_PROBLEM_CONTEXT)).strip().lower()
        )
        if solve_context not in PROBLEM_CONTEXT_MODES:
            raise ValueError(
                f"method {name!r} solve_context must be one of {PROBLEM_CONTEXT_MODES}"
            )
        if "solve_context" in raw and solve_kind not in ONLINE_SOLVE_KINDS:
            raise ValueError(
                f"method {name!r} can only set solve_context for an online solve controller"
            )
        return cls(
            name=name,
            setup_kind=setup_kind,
            solve_kind=solve_kind,
            setup_space=setup_space,
            setup_context=setup_context,
            candidate_sampling=candidate_sampling,
            fixed_weight=fixed_weight,
            seed_offset=seed_offset,
            setup_warmup_cases=setup_warmup_cases,
            solve_activation_case=solve_activation_case,
            solve_tolerance=solve_tolerance,
            solve_context=solve_context,
            solve_activation=solve_activation,
        )


__all__ = ["CANDIDATE_SAMPLING_METHODS", "ComposableMethodSpec", "METHOD_KEYS"]
