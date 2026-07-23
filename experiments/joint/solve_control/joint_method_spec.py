from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from setup.registry import COMPOSABLE_SETUP_KINDS, ONLINE_SETUP_KINDS
from solve.registry import COMPOSABLE_SOLVE_KINDS


METHOD_KEYS = {
    "id",
    "setup",
    "setup_space",
    "candidate_sampling",
    "solve",
    "fixed_weight",
}
CANDIDATE_SAMPLING_METHODS = ("uniform512", "structured512")


@dataclass(frozen=True)
class ComposableMethodSpec:
    """One setup/solve branch selected by a high-level experiment config."""

    name: str
    setup_kind: str
    solve_kind: str
    setup_space: str | None = None
    candidate_sampling: str = "uniform512"
    fixed_weight: float | None = None

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
            "lints": "Online LinTS v2",
        }[self.setup_kind]
        if self.setup_space is not None:
            candidate = self.candidate_sampling.replace("512", "-512")
            setup = f"{setup} [{self.setup_space}; {candidate}]"
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
        return f"{setup} + {solve}"

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
            if setup_kind not in {"linucb", "lints"}:
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

        fixed_weight = None
        if solve_kind == "fixed":
            if "fixed_weight" not in raw:
                raise ValueError(f"method {name!r} requires fixed_weight")
            fixed_weight = float(raw["fixed_weight"])
            if not math.isfinite(fixed_weight) or fixed_weight <= 0.0:
                raise ValueError(
                    f"method {name!r} fixed_weight must be finite and positive"
                )

        return cls(
            name=name,
            setup_kind=setup_kind,
            solve_kind=solve_kind,
            setup_space=setup_space,
            candidate_sampling=candidate_sampling,
            fixed_weight=fixed_weight,
        )

    def to_runner_token(self) -> str:
        """Serialize only at the temporary legacy-runner boundary."""

        setup = self.setup_kind
        if self.setup_space is not None:
            setup = (
                f"{setup}@{self.setup_space}@{self.candidate_sampling}"
            )
        solve = self.solve_kind
        if solve == "fixed":
            solve = f"fixed@{float(self.fixed_weight):g}"
        return f"{self.name}:{setup}:{solve}"

    @classmethod
    def from_runner_token(cls, raw: str) -> "ComposableMethodSpec":
        """Parse the temporary ``name:setup:solve`` CLI boundary."""

        parts = str(raw).split(":", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                "Composable methods must use name:setup:solve syntax"
            )
        name, setup_token, solve_token = (
            part.strip() for part in parts
        )
        setup_parts = setup_token.split("@")
        if len(setup_parts) > 3:
            raise ValueError(
                "Setup tokens use "
                "setup[@space[@candidate_sampling]] syntax"
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
        if solve_token.startswith("fixed@"):
            method["solve"] = "fixed"
            method["fixed_weight"] = float(
                solve_token.split("@", 1)[1]
            )
        return cls.from_mapping(method)


__all__ = [
    "CANDIDATE_SAMPLING_METHODS",
    "ComposableMethodSpec",
    "METHOD_KEYS",
]
