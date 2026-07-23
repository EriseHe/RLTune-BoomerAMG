from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from solve.controllers.common.config_decode import (
    config_value,
    reject_unknown_config_keys,
)


_JSON_KEYS = frozenset(
    {
        "prior_precision",
        "noise_precision",
        "gram_ridge",
    }
)


@dataclass(frozen=True)
class RecursiveBlstdqSpec:
    """Bayesian LSTDQ posterior controls for RBLSPI-style exploration."""

    prior_precision: float = 1.0e4
    noise_precision: float = 1.0e6
    gram_ridge: float = 1.0e-6

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        name: str = "solve.rblspi",
    ) -> "RecursiveBlstdqSpec":
        """Decode the RBLSPI section of a joint experiment config."""

        reject_unknown_config_keys(raw, _JSON_KEYS, name=name)
        return cls(
            prior_precision=float(
                config_value(
                    raw,
                    "prior_precision",
                    cls.prior_precision,
                )
            ),
            noise_precision=float(
                config_value(
                    raw,
                    "noise_precision",
                    cls.noise_precision,
                )
            ),
            gram_ridge=float(
                config_value(raw, "gram_ridge", cls.gram_ridge)
            ),
        )


__all__ = ["RecursiveBlstdqSpec"]
