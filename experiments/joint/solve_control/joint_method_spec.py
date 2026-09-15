from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from joint_rl_activation import ReliabilityActivationSpec
from problems.registry import (
    DEFAULT_SETUP_CONTEXT,
    normalize_setup_context_mode,
)
from setup.registry import COMPOSABLE_SETUP_KINDS, ONLINE_SETUP_KINDS
from solve.controllers.common import (
    CANONICAL_PROBLEM_CONTEXT,
    PROBLEM_CONTEXT_MODES,
)
from solve.registry import COMPOSABLE_SOLVE_KINDS, ONLINE_SOLVE_KINDS


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
    solve_activation: ReliabilityActivationSpec | None = None

    @property
    def family(self) -> str:
        if self.setup_kind == "default" and self.solve_kind == "default":
            return "default_setup"
        if (
            self.setup_kind in ONLINE_SETUP_KINDS
            and self.solve_kind == "default"
        ):
            return "default"
        if (
            self.setup_kind in ONLINE_SETUP_KINDS
            and self.solve_kind == "fixed"
        ):
            if math.isclose(
                float(self.fixed_weight),
                1.6,
                rel_tol=1.0e-5,
                abs_tol=1.0e-8,
            ):
                return "fixed_w1.6"
            return f"fixed_w{float(self.fixed_weight):g}"
        family = {
            "ppo": "ppo",
            "recursive_mc": "recursive_mc_lcb",
            "recursive_lstdq_v1": "recursive_lstdq_lcb",
            "recursive_lstdq_v2": "recursive_lstdq_v2_lcb",
            "recursive_lstdq_v3": "recursive_lstdq_v3_lcb",
            "rblspi": "rblspi",
            "stagewise_lsvi": "stagewise_lsvi_lcb",
            "structured_model_based": "structured_model_based",
            "recalibrated_lsvi": "recalibrated_lsvi_lcb",
        }[self.solve_kind]
        if self.setup_kind == "default":
            return f"default_setup_{family}"
        return family

    @property
    def label(self) -> str:
        setup = {
            "default": "Default setup",
            "linucb": "Online LinUCB",
            "linucb_v5": "Online LinUCB v5",
            "linucb_v5_rbf": "Online LinUCB v5 RBF",
            "linucb_v6": "Online LinUCB v6",
            "lints": "Online LinTS v2",
        }[self.setup_kind]
        if self.setup_space is not None:
            qualifiers = []
            if self.setup_space != "recommended":
                qualifiers.append(self.setup_space)
            if self.candidate_sampling != "structured512":
                qualifiers.append(
                    self.candidate_sampling.replace("512", "-512")
                )
            if self.setup_context != DEFAULT_SETUP_CONTEXT:
                qualifiers.append(
                    f"context={self.setup_context.replace('_', '-')}"
                )
            if qualifiers:
                setup = f"{setup} [{'; '.join(qualifiers)}]"
        if self.setup_warmup_cases:
            setup = f"{setup}; warmup={self.setup_warmup_cases}"
        if self.solve_kind == "fixed":
            solve = f"fixed w={float(self.fixed_weight):g}"
        else:
            solve = {
                "default": "default solve",
                "ppo": "PPO",
                "recursive_mc": "Recursive MC-LCB",
                "recursive_lstdq_v1": "Recursive LSTDQ-LCB",
                "recursive_lstdq_v2": "Recursive LSTDQ v2-LCB",
                "recursive_lstdq_v3": "Recursive LSTDQ v3-LCB",
                "rblspi": "Recursive BLSTDQ / RBLSPI",
                "stagewise_lsvi": "Stagewise LSVI-LCB",
                "structured_model_based": (
                    "structured model-based control"
                ),
                "recalibrated_lsvi": "Recalibrated LSVI-LCB",
            }[self.solve_kind]
        if self.solve_activation_case:
            solve = f"{solve}; activate={self.solve_activation_case}"
        if self.solve_activation is not None:
            solve = f"{solve}; activate=reliability mixture"
        if self.solve_tolerance is not None:
            solve = f"{solve}; tol={float(self.solve_tolerance):g}"
        if self.solve_context != CANONICAL_PROBLEM_CONTEXT:
            solve = f"{solve}; context={self.solve_context}"
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
        unknown = set(raw) - METHOD_KEYS
        if unknown:
            raise ValueError(
                f"method {raw.get('id', '<unknown>')!r} has unknown keys: "
                f"{sorted(unknown)}"
            )

        name = str(raw["id"])
        if not name.strip():
            raise ValueError("method id cannot be empty")
        if any(character in name for character in "/\\"):
            raise ValueError("method ids cannot contain path separators")

        setup_kind = str(raw["setup"])
        solve_kind = str(raw["solve"])
        if setup_kind not in COMPOSABLE_SETUP_KINDS:
            raise ValueError(f"Unsupported setup kind: {setup_kind}")
        if solve_kind not in COMPOSABLE_SOLVE_KINDS:
            raise ValueError(f"Unsupported solve kind: {solve_kind}")

        setup_space_raw = raw.get("setup_space")
        setup_space = (
            None if setup_space_raw is None else str(setup_space_raw).strip()
        )
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
                raise ValueError(
                    f"method {name!r} setup_space cannot be empty"
                )
            if setup_kind not in ONLINE_SETUP_KINDS:
                raise ValueError(
                    f"method {name!r} can only set setup_space for a setup bandit"
                )
            if candidate_sampling not in CANDIDATE_SAMPLING_METHODS:
                raise ValueError(
                    f"method {name!r} candidate_sampling must be "
                    "uniform512 or structured512"
                )
        elif "candidate_sampling" in raw:
            raise ValueError(
                f"method {name!r} requires setup_space when setting "
                "candidate_sampling"
            )
        if "setup_context" in raw and setup_kind not in ONLINE_SETUP_KINDS:
            raise ValueError(
                f"method {name!r} can only set setup_context for a setup bandit"
            )
        if (
            setup_context != DEFAULT_SETUP_CONTEXT
            and setup_space is None
        ):
            raise ValueError(
                f"method {name!r} requires setup_space when setting "
                "a non-default setup_context"
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
            raise ValueError(
                f"method {name!r} setup_warmup_cases must be non-negative"
            )
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
        activation_raw = raw.get("solve_activation")
        solve_activation = (
            None if activation_raw is None
            else ReliabilityActivationSpec.from_mapping(activation_raw)
        )
        if solve_activation is not None:
            if solve_kind not in ONLINE_SOLVE_KINDS:
                raise ValueError(
                    "Dynamic solve_activation requires an online solve controller"
                )
            if setup_kind not in ONLINE_SETUP_KINDS:
                raise ValueError(
                    "Dynamic solve_activation requires an online setup learner"
                )
            if solve_activation_case or setup_warmup_cases:
                raise ValueError(
                    "Dynamic solve_activation cannot also use a fixed activation or warmup"
                )
        solve_tolerance_raw = raw.get("solve_tolerance")
        solve_tolerance = (
            None
            if solve_tolerance_raw is None
            else float(solve_tolerance_raw)
        )
        if solve_tolerance is not None and (
            not math.isfinite(solve_tolerance) or solve_tolerance <= 0.0
        ):
            raise ValueError(
                f"method {name!r} solve_tolerance must be finite and positive"
            )
        solve_context = str(
            raw.get("solve_context", CANONICAL_PROBLEM_CONTEXT)
        ).strip().lower()
        if solve_context not in PROBLEM_CONTEXT_MODES:
            raise ValueError(
                f"method {name!r} solve_context must be one of "
                f"{PROBLEM_CONTEXT_MODES}"
            )
        if (
            "solve_context" in raw
            and solve_kind not in ONLINE_SOLVE_KINDS
        ):
            raise ValueError(
                f"method {name!r} can only set solve_context for an "
                "online solve controller"
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

    def to_runner_token(self) -> str:
        """Serialize only at the temporary legacy-runner boundary."""

        setup = self.setup_kind
        if self.setup_space is not None:
            setup_parts = [
                setup,
                self.setup_space,
                self.candidate_sampling,
            ]
            if self.setup_context != DEFAULT_SETUP_CONTEXT:
                setup_parts.append(self.setup_context)
            setup = "@".join(setup_parts)
        solve = self.solve_kind
        if solve == "fixed":
            solve = f"fixed@{float(self.fixed_weight):g}"
        elif self.solve_context != CANONICAL_PROBLEM_CONTEXT:
            solve = f"{solve}@{self.solve_context}"
        token = f"{self.name}:{setup}:{solve}"
        if (
            self.seed_offset
            or self.setup_warmup_cases
            or self.solve_activation_case
            or self.solve_activation is not None
            or self.solve_tolerance is not None
        ):
            token = f"{token}:{self.seed_offset}"
        if (
            self.setup_warmup_cases
            or self.solve_activation_case
            or self.solve_activation is not None
            or self.solve_tolerance is not None
        ):
            token = f"{token}:{self.setup_warmup_cases}"
        if (
            self.solve_activation_case
            or self.solve_tolerance is not None
            or self.solve_activation is not None
        ):
            activation = str(self.solve_activation_case)
            if self.solve_activation is not None:
                rule = self.solve_activation
                if rule.kind == "composite_reliability_mixture":
                    activation = f"composite@{rule.p_bad!r}@{rule.delta!r}@{rule.horizon}"
                else:
                    activation = f"mixture@{rule.p_bad!r}@{rule.p_good!r}@{rule.delta!r}"
            token = f"{token}:{activation}"
        if self.solve_tolerance is not None:
            token = f"{token}:{float(self.solve_tolerance):g}"
        return token

    @classmethod
    def from_runner_token(cls, raw: str) -> "ComposableMethodSpec":
        """Parse the temporary ``name:setup:solve`` CLI boundary."""

        parts = str(raw).split(":")
        if len(parts) not in {3, 4, 5, 6, 7} or not all(
            part.strip() for part in parts
        ):
            raise ValueError(
                "Composable methods must use "
                "name:setup:solve[:seed_offset[:setup_warmup_cases"
                "[:solve_activation_case[:solve_tolerance]]]] syntax"
            )
        name, setup_token, solve_token = (
            part.strip() for part in parts[:3]
        )
        setup_parts = setup_token.split("@")
        if len(setup_parts) > 4:
            raise ValueError(
                "Setup tokens use "
                "setup[@space[@candidate_sampling[@context]]] syntax"
            )
        method: dict[str, Any] = {
            "id": name,
            "setup": setup_parts[0],
            "solve": solve_token,
        }
        if len(setup_parts) >= 2:
            method["setup_space"] = setup_parts[1].strip()
        if len(setup_parts) == 3:
            method["candidate_sampling"] = setup_parts[2]
        if len(setup_parts) == 4:
            method["candidate_sampling"] = setup_parts[2]
            method["setup_context"] = setup_parts[3]
        if solve_token.startswith("fixed@"):
            method["solve"] = "fixed"
            method["fixed_weight"] = float(
                solve_token.split("@", 1)[1]
            )
        elif "@" in solve_token:
            solve_parts = solve_token.split("@")
            if len(solve_parts) != 2:
                raise ValueError(
                    "Solve-controller tokens use solve[@context] syntax"
                )
            method["solve"] = solve_parts[0]
            method["solve_context"] = solve_parts[1]
        if len(parts) >= 4:
            method["seed_offset"] = int(parts[3])
        if len(parts) >= 5:
            method["setup_warmup_cases"] = int(parts[4])
        if len(parts) >= 6:
            if parts[5].startswith("composite@"):
                activation = parts[5].split("@")
                if len(activation) != 4:
                    raise ValueError("Composite activation token uses composite@p_bad@delta@horizon")
                method["solve_activation"] = {
                    "kind": "composite_reliability_mixture",
                    "p_bad": float(activation[1]),
                    "delta": float(activation[2]),
                    "horizon": int(activation[3]),
                }
            elif parts[5].startswith("mixture@"):
                activation = parts[5].split("@")
                if len(activation) != 4:
                    raise ValueError(
                        "Dynamic activation token uses mixture@p_bad@p_good@delta"
                    )
                method["solve_activation"] = {
                    "p_bad": float(activation[1]),
                    "p_good": float(activation[2]),
                    "delta": float(activation[3]),
                }
            else:
                method["solve_activation_case"] = int(parts[5])
        if len(parts) == 7:
            method["solve_tolerance"] = float(parts[6])
        return cls.from_mapping(method)


__all__ = [
    "CANDIDATE_SAMPLING_METHODS",
    "ComposableMethodSpec",
    "METHOD_KEYS",
]
