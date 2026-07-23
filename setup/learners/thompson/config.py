from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinTSV2Spec:
    """Algorithm-specific configuration for shared linear TS v2."""

    relative_sampling_scale: float = 0.15
    loss_scale_prior: float = 0.1
    posterior_seed: int | None = None

    def learner_kwargs(self) -> dict[str, Any]:
        return {
            "relative_sampling_scale": float(
                self.relative_sampling_scale
            ),
            "loss_scale_prior": float(self.loss_scale_prior),
            "posterior_seed": self.posterior_seed,
        }


__all__ = ["LinTSV2Spec"]
