"""Verify the small accepted Run 05 input bundle without historical run folders."""

from __future__ import annotations

from pathlib import Path

from experiments.paper_final.verify_05_policy_minimax import verify_fixed_choices
from .artifacts import ROOT, file_hash, read

DEFAULT_BUNDLE = (
    ROOT / "experiments/paper_final/reproduction/matched_policy/frozen_inputs"
)
INPUT_FILES = (
    "protocol.json",
    "inputs.json",
    "selections.json",
    "jobs_base_test.json",
    "jobs_preflight.json",
    "jobs_test.json",
)


def verify_bundle(directory):
    """Check immutable accepted bytes and independently reconstruct choices."""
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if manifest["format"] != "sisc-run05-frozen-inputs-v1":
        raise ValueError("Unsupported frozen-input bundle")
    expected = manifest["files_sha256"]
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if actual != set(expected):
        raise ValueError("Frozen-input bundle has missing or extra files")
    for name, digest in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe frozen-input path")
        if file_hash(directory / name) != digest:
            raise ValueError(f"Frozen-input bundle hash mismatch: {name}")
    saved = read(directory / "accepted_prepared.json")
    accepted = {
        "protocol.json": saved["protocol_sha256"],
        "inputs.json": saved["inputs_sha256"],
        "selections.json": saved["selections_sha256"],
        **{f"jobs_{p}.json": h for p, h in saved["jobs_sha256"].items()},
    }
    for name in expected:
        if name.startswith("checkpoints/"):
            accepted[name] = saved["checkpoint_sha256"][name]
    for name, digest in accepted.items():
        if expected.get(name) != digest:
            raise ValueError(f"Accepted input identity changed: {name}")
    protocol = read(directory / "protocol.json")
    if (
        protocol["run_number"] != 5
        or protocol["training_seeds"] != list(range(1, 7))
        or protocol["test_cases"] != 100
        or protocol["repetitions_per_case"] != 3
        or protocol["execution_count"] != 8460
    ):
        raise ValueError("Incorrect accepted Run 05 protocol")
    jobs = read(directory / "jobs_test.json")
    evidence = read(directory / "fixed_scan_evidence.json")
    scan_protocol = dict(
        protocol, scan_extra_repeat_case_ids=evidence["scan_extra_repeat_case_ids"]
    )
    selections = read(directory / "selections.json")["chosen_policies"]
    counts = {}
    for seed in protocol["training_seeds"]:
        jobmap = {(j["case_id"], j["repeat"]): j for j in jobs if j["seed"] == seed}
        if set(jobmap) != {(case, rep) for case in range(100) for rep in range(3)}:
            raise ValueError("Incorrect accepted job coverage")
        for job in jobmap.values():
            if set(job["method_policies"]) != set(protocol["methods"]):
                raise ValueError("Incorrect accepted method roles")
            if set(job["policy_order"]) != set(job["method_policies"].values()):
                raise ValueError("Incorrect accepted policy order")
            for method, selected in job["method_policies"].items():
                chosen = selections[f"{seed}/bandit_lstdq/{method}"]
                if isinstance(chosen, dict):
                    chosen = chosen[str(job["case_id"])]
                if selected != chosen:
                    raise ValueError("Frozen selection differs from evaluation jobs")
        origin = evidence["sources"][f"seed_{seed}"]
        if expected[origin["derived_path"]] != origin["derived_sha256"]:
            raise ValueError("Derived fixed-scan hash changed")
        counts[seed] = verify_fixed_choices(
            seed, jobmap, scan_protocol, directory / origin["derived_path"]
        )
        if counts[seed] != origin["records"] or counts[seed] != 4920:
            raise ValueError("Incorrect retained fixed-scan coverage")
    return {
        "bundle_sha256": file_hash(directory / "manifest.json"),
        "accepted_run": manifest["accepted_run"],
        "exact_accepted_input_hashes": True,
        "fixed_choices_reconstructed": True,
        "fixed_scan_records": sum(counts.values()),
    }


def execution_source_paths():
    """Capture current submission source, excluding vendor and data artifacts."""
    packages = (
        "experiments/paper_final",
        "experiments/paper_final/online",
        "problems",
        "setup",
        "solve",
        "hypre/bindings",
    )
    paths = {
        str(path.relative_to(ROOT))
        for package in packages
        for path in (ROOT / package).rglob("*.py")
    }
    paths.add("experiments/runtime.py")
    interfaces = ROOT / "hypre/interfaces"
    paths.update(
        str(p.relative_to(ROOT))
        for p in interfaces.iterdir()
        if p.is_file() and (p.suffix in (".c", ".h") or p.name == "Makefile")
    )
    paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob("requirements*.txt"))
    if (ROOT / "pyproject.toml").is_file():
        paths.add("pyproject.toml")
    return paths
