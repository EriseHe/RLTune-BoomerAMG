"""Read the exact accepted Module 04 inputs without recomputing logarithms."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STREAM_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "reproduction" / "online" / "streams"
)


def sampling_signature(
    args: Any,
    *,
    problem: str,
    grid: tuple[int, int, int],
    advection: tuple[float, float, float],
    sampled_advection_range: tuple[float, float] | None,
) -> dict[str, Any]:
    """Bind an input capture to every setting used by the original sampler."""
    groups = str(args.train_seed_groups).strip()
    return {
        "problem": str(problem),
        "grid": [int(value) for value in grid],
        "diffusion_range": [float(args.c_min), float(args.c_max)],
        "advection": [float(value) for value in advection],
        "sampled_advection_range": None
        if sampled_advection_range is None
        else [float(value) for value in sampled_advection_range],
        "seed": int(args.seed),
        "seed_groups": [
            [int(value.strip()) for value in group.split(",") if value.strip()]
            for group in groups.split(";")
            if group.strip()
        ],
        "shuffle_seeds": [
            int(value.strip())
            for value in str(args.train_shuffle_seeds).split(",")
            if value.strip()
        ],
        "instance_offset": int(args.instance_offset),
        "train_cases": int(args.train_cases),
        "cases_per_seed": int(args.train_cases_per_seed),
        "group_take": int(args.train_group_take),
    }


def canonical_stream_bytes(stream: Any) -> bytes:
    """Serialize inputs exactly as the historical stream SHA-256 calculation."""
    return b"".join(
        json.dumps(
            {
                "mkw": dict(matrix_kwargs),
                "context": np.asarray(context, dtype=float).tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for matrix_kwargs, context in stream
    )


def load_recorded_online_stream(
    args: Any,
    *,
    problem: str,
    grid: tuple[int, int, int],
    advection: tuple[float, float, float],
    sampled_advection_range: tuple[float, float] | None,
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], dict[str, Any]] | None:
    """Verify and replay a recognized pinned stream, or leave sampling unchanged."""
    expected_hash = str(getattr(args, "expected_stream_hash", "") or "").strip()
    if not expected_hash:
        return None
    manifest = json.loads(
        (STREAM_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported recorded online input format")
    entry: Mapping[str, Any] | None = manifest["streams"].get(expected_hash)
    if entry is None:
        return None
    signature = sampling_signature(
        args,
        problem=problem,
        grid=grid,
        advection=advection,
        sampled_advection_range=sampled_advection_range,
    )
    if signature != entry["sampling_signature"]:
        raise ValueError("Recorded stream hash is bound to different sampling settings")
    if entry["canonical_sha256"] != expected_hash:
        raise ValueError("Recorded stream identity does not match the requested hash")
    filename = f"{expected_hash}.jsonl.gz"
    if entry["file"] != filename:
        raise ValueError("Recorded stream filename does not match its identity")
    compressed = (STREAM_DIRECTORY / filename).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != entry["compressed_sha256"]:
        raise ValueError("Recorded stream compressed file hash mismatch")
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Recorded stream canonical input hash mismatch")
    stream = []
    for line in payload.splitlines():
        row = json.loads(line)
        if set(row) != {"mkw", "context"}:
            raise ValueError("Invalid recorded stream row")
        context = np.asarray(row["context"], dtype=float)
        if context.shape != (8,) or not np.all(np.isfinite(context)):
            raise ValueError("Invalid recorded PDE context")
        stream.append((dict(row["mkw"]), context))
    if len(stream) != int(args.train_cases) or len(stream) != entry["cases"]:
        raise ValueError("Recorded stream length does not match the sampling settings")
    if canonical_stream_bytes(stream) != payload:
        raise ValueError("Recorded inputs do not round-trip to the canonical stream")
    stream_manifest = dict(entry["stream_manifest"])
    if stream_manifest["sha256"] != expected_hash:
        raise ValueError("Recorded stream manifest identity mismatch")
    return stream, stream_manifest
