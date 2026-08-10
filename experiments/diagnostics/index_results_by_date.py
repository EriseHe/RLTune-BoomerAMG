#!/usr/bin/env python3
"""Build a non-destructive, date-oriented index of experiment outputs.

The executable configs in this repository intentionally write to stable semantic
paths such as ``results/joint/...``.  Moving those directories breaks saved
commands and can split an experiment that is still running.  This utility keeps
the producer paths unchanged and creates relative symlinks under
``results/by_date/YYYY-MM-DD`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DATE_TOKEN = re.compile(r"(?<!\d)(20\d{6})(?:[_-]\d{6})?(?!\d)")
RUN_MARKERS = {
    "config.json",
    "pipeline_summary.json",
    "progress.json",
    "result.json",
    "run_config.json",
    "stream_manifest.json",
    "tuning_summary.json",
}
IGNORED_RESULT_ROOTS = {"archive", "by_date"}
GENERIC_PARENT_NAMES = {"diagnosis", "run_logs", "solve_control"}


@dataclass(frozen=True)
class ResultEntry:
    date: str
    date_basis: str
    category: str
    alias: str
    source: str
    kind: str
    status: str


def _date_token(path: Path) -> str | None:
    match = DATE_TOKEN.search(path.name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")


def _direct_file_names(path: Path) -> set[str]:
    return {child.name for child in path.iterdir() if child.is_file()}


def _contains_direct_run_marker(path: Path) -> bool:
    return bool(_direct_file_names(path) & RUN_MARKERS)


def _under_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(root == path or root in path.parents for root in roots)


def _candidate_directories(results_dir: Path) -> list[Path]:
    directories = [
        path
        for path in results_dir.rglob("*")
        if path.is_dir()
        and not path.is_symlink()
        and not any(
            results_dir / name == path or results_dir / name in path.parents
            for name in IGNORED_RESULT_ROOTS
        )
    ]

    dated = {path for path in directories if _date_token(path) is not None}
    eligible_dated: list[Path] = []
    for path in dated:
        has_dated_descendant = any(path in other.parents for other in dated)
        # Dated containers such as an invalidation bundle may hold runs from
        # several earlier days.  Keep the individual dated descendants instead.
        if has_dated_descendant and not _contains_direct_run_marker(path):
            continue
        eligible_dated.append(path)

    selected: list[Path] = []
    for path in sorted(eligible_dated, key=lambda item: (len(item.parts), str(item))):
        if not _under_any(path, selected):
            selected.append(path)

    marker_directories = [
        path for path in directories if _contains_direct_run_marker(path)
    ]
    for path in sorted(marker_directories, key=lambda item: (len(item.parts), str(item))):
        if not _under_any(path, selected):
            selected.append(path)

    setup_dir = results_dir / "setup"
    if setup_dir.is_dir():
        for path in sorted(setup_dir.iterdir()):
            if path.is_dir() and not path.is_symlink() and not _under_any(path, selected):
                selected.append(path)

    return sorted(selected)


def _candidate_files(results_dir: Path, directory_roots: Iterable[Path]) -> list[Path]:
    roots = tuple(directory_roots)
    return sorted(
        path
        for path in results_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {".DS_Store", "README.md"}
        and _date_token(path) is not None
        and not any(
            results_dir / name == path or results_dir / name in path.parents
            for name in IGNORED_RESULT_ROOTS
        )
        and not _under_any(path, roots)
    )


def _birth_date(path: Path) -> str:
    files = (
        [path]
        if path.is_file()
        else [
            child
            for child in path.rglob("*")
            if child.is_file() and child.name != ".DS_Store"
        ]
    )
    timestamps = []
    for file_path in files:
        stat = file_path.stat()
        timestamps.append(float(getattr(stat, "st_birthtime", stat.st_mtime)))
    timestamp = min(timestamps) if timestamps else path.stat().st_mtime
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def _run_status(path: Path, active_outputs: set[Path]) -> str:
    resolved = path.resolve()
    if any(resolved == active or resolved in active.parents for active in active_outputs):
        return "active"
    if path.is_file():
        return "artifact"
    if (path / "result.json").is_file():
        return "complete"
    if (path / "progress.json").is_file() or (path / "stream_manifest.json").is_file():
        return "partial"
    if any(path.rglob("result.json")):
        return "complete"
    return "artifact"


def _category_and_alias(path: Path, results_dir: Path) -> tuple[str, str]:
    relative = path.relative_to(results_dir)
    category = relative.parts[0] if len(relative.parts) > 1 else "other"
    parent_parts = [
        part
        for part in relative.parts[1:-1]
        if part not in GENERIC_PARENT_NAMES
    ]
    context = parent_parts[-2:]
    alias = "__".join((*context, path.name)) if context else path.name
    return category, alias


def discover_entries(results_dir: Path, active_outputs: set[Path]) -> list[ResultEntry]:
    directory_roots = _candidate_directories(results_dir)
    sources = [*directory_roots, *_candidate_files(results_dir, directory_roots)]
    entries: list[ResultEntry] = []
    used_aliases: set[tuple[str, str, str]] = set()

    for source in sources:
        run_date = _date_token(source)
        date_basis = "name" if run_date is not None else "filesystem birth time"
        run_date = run_date or _birth_date(source)
        category, alias = _category_and_alias(source, results_dir)
        alias_key = (run_date, category, alias)
        if alias_key in used_aliases:
            digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
            alias = f"{alias}__{digest}"
            alias_key = (run_date, category, alias)
        used_aliases.add(alias_key)
        entries.append(
            ResultEntry(
                date=run_date,
                date_basis=date_basis,
                category=category,
                alias=alias,
                source=str(source.relative_to(results_dir.parent)),
                kind="directory" if source.is_dir() else "file",
                status=_run_status(source, active_outputs),
            )
        )

    return sorted(
        entries,
        key=lambda entry: (entry.date, entry.category, entry.alias, entry.source),
    )


def _write_symlink(link_path: Path, source_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(source_path, start=link_path.parent)
    if link_path.is_symlink():
        if link_path.resolve() == source_path.resolve():
            return
        raise FileExistsError(f"Refusing to replace unrelated link: {link_path}")
    if link_path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(relative_target, target_is_directory=source_path.is_dir())


def _write_indexes(results_dir: Path, entries: list[ResultEntry]) -> None:
    index_root = results_dir / "by_date"
    index_root.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[ResultEntry]] = {}
    for entry in entries:
        by_date.setdefault(entry.date, []).append(entry)
        source_path = results_dir.parent / entry.source
        link_path = index_root / entry.date / entry.category / entry.alias
        _write_symlink(link_path, source_path)

    for run_date, date_entries in sorted(by_date.items()):
        lines = [
            f"# Experiment outputs: {run_date}",
            "",
            "This directory is a zero-copy date view. Links point to the stable",
            "producer paths elsewhere under `results/`.",
            "",
            "| Status | Category | Output | Original path | Date basis |",
            "|---|---|---|---|---|",
        ]
        for entry in date_entries:
            link = f"{entry.category}/{entry.alias}"
            lines.append(
                f"| {entry.status} | {entry.category} | "
                f"[{entry.alias}]({link}) | `{entry.source}` | {entry.date_basis} |"
            )
        lines.append("")
        (index_root / run_date / "README.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    overview = [
        "# Experiment outputs by date",
        "",
        "This is the canonical browsing view for generated experiment outputs.",
        "The entries are relative symlinks, so no output data is duplicated and",
        "existing configs, reports, and running experiments keep their original paths.",
        "",
        "| Date | Outputs | Complete | Partial | Active |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_date, date_entries in sorted(by_date.items(), reverse=True):
        counts = {
            status: sum(entry.status == status for entry in date_entries)
            for status in ("complete", "partial", "active")
        }
        overview.append(
            f"| [{run_date}]({run_date}/README.md) | {len(date_entries)} | "
            f"{counts['complete']} | {counts['partial']} | {counts['active']} |"
        )
    overview.extend(
        (
            "",
            "Regenerate from the repository root:",
            "",
            "```bash",
            "python experiments/diagnostics/index_results_by_date.py",
            "```",
            "",
        )
    )
    (index_root / "README.md").write_text("\n".join(overview), encoding="utf-8")
    (index_root / "manifest.json").write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="results directory relative to the current working directory",
    )
    parser.add_argument(
        "--active-output",
        action="append",
        default=[],
        type=Path,
        help="output path currently being written; may be repeated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the discovered manifest without creating the date view",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    active_outputs = {path.resolve() for path in args.active_output}
    entries = discover_entries(results_dir, active_outputs)
    if args.dry_run:
        print(json.dumps([asdict(entry) for entry in entries], indent=2))
        return
    _write_indexes(results_dir, entries)
    print(f"Indexed {len(entries)} outputs across {len({e.date for e in entries})} dates")


if __name__ == "__main__":
    main()
