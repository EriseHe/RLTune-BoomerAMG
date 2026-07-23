from __future__ import annotations

from typing import Any, Mapping


def reject_unknown_config_keys(
    raw: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    name: str,
) -> None:
    """Reject experiment fields that are not owned by a controller family."""

    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def config_value(
    raw: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Return a configured value, treating JSON null as the family default."""

    value = raw.get(key)
    return default if value is None else value


__all__ = ["config_value", "reject_unknown_config_keys"]
