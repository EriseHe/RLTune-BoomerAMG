"""Durable trajectory records for the official online study."""

from __future__ import annotations
import json
from typing import Any, Dict
from .io import _json_ready


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")
