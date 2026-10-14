"""Durable experiment files and hashes; importing this module loads no solver."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def ready(value):
    value = _json_ready(value)
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(ready(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def emit(output, message, **fields):
    event = {"at": now(), "message": message, **fields}
    with (output / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(ready(event), sort_keys=True) + "\n")
    print(
        f"[{event['at']}] {message}"
        + (" " + json.dumps(ready(fields)) if fields else ""),
        flush=True,
    )


def checkpoint_payload_hash(path, *, exclude=()):
    import numpy as np

    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        for key in sorted(set(payload.files) - set(exclude)):
            a = np.asarray(payload[key])
            h.update(key.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def read_records(path, *, repair_tail=False):
    """Stream durable rows; only a torn final append may be repaired on resume."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b" if repair_tail else "rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not repair_tail or handle.read(1):
                    raise ValueError(
                        f"Corrupt experiment record in {path} at byte {start}"
                    )
                path.with_suffix(f".torn.{time.time_ns()}").write_bytes(line)
                handle.seek(start)
                handle.truncate()
                return
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Unterminated record in {path}")
                handle.seek(0, 2)
                handle.write(b"\n")
            yield row


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def support_source_paths():
    """Current shared implementations, including newly added untracked source."""
    directories = (
        Path(__file__).parent,
        ROOT / "experiments/paper_final/online",
        ROOT / "solve/core",
        ROOT / "solve/controllers/common",
    )
    paths = {
        str(path.relative_to(ROOT))
        for directory in directories
        for path in directory.glob("*.py")
    }
    paths.add("experiments/runtime.py")
    return paths


def source_differences(manifest):
    """Report missing and changed source; an archived path is still a change."""
    return [
        name
        for name, expected in manifest.items()
        if not (ROOT / name).is_file() or file_hash(ROOT / name) != expected
    ]


def presentation_source_paths():
    """Current rendering sources, separate from frozen numerical provenance."""
    return {
        "experiments/paper_final/plot_05_policy_minimax.py",
        "experiments/paper_final/common/__init__.py",
        "experiments/paper_final/common/artifacts.py",
        "experiments/paper_final/common/figure_style.py",
        "experiments/paper_final/common/scientific_transforms.py",
    }
