"""Explicit inputs for fresh Module 2 studies; no historical run discovery."""

from __future__ import annotations

import copy
from decimal import Decimal
import math
from pathlib import Path

from experiments.paper_final.common.artifacts import read

METHODS = ("reference", "fixed", "oracle", "periodic", "rl")
LABELS = {
    "reference": "Fixed w=1",
    "fixed": "Best global fixed",
    "oracle": "Best per-instance fixed",
    "periodic": "Periodic schedule",
    "rl": "Frozen LSTDQ",
}
DEFAULTS = {
    "schema_version": 1,
    "name": "matched_policy",
    "cases": 100,
    "scan_repetitions": 1,
    "scan_extra_repetitions": 2,
    "scan_repeat_fraction": 0.1,
    "evaluation_repetitions": 3,
    "weights": {"start": 1, "stop": 3, "step": 0.05},
    "periodic_pattern": [2.85, 1.10],
    "workers": 3,
    "rl_beta": None,
}
SEED_KEYS = {"inputs", "candidates", "order", "repeats"}
SOURCE_KEYS = {"seed", "run_dir", "method", "config", "controller_checkpoint", "setup_checkpoint"}


def positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def weight_grid(raw):
    if isinstance(raw, dict):
        if set(raw) != {"start", "stop", "step"}:
            raise ValueError("weights requires start, stop, step")
        start, stop, step = (Decimal(str(raw[k])) for k in ("start", "stop", "step"))
        if not all(v.is_finite() for v in (start, stop, step)) or step <= 0 or stop < start:
            raise ValueError("Invalid weight grid")
        count = (stop - start) / step
        if count != count.to_integral_value():
            raise ValueError("Weight-grid stop must lie on the grid")
        values = [float(start + i * step) for i in range(int(count) + 1)]
    elif isinstance(raw, list):
        values = [float(v) for v in raw]
    else:
        raise ValueError("weights must be a list or start/stop/step object")
    if not values or len(values) != len(set(values)) or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Weights must be unique finite positive numbers")
    return values


def fixed_name(weight):
    return "fixed_" + format(float(weight), ".12g")


def load_config(path):
    """Resolve paths relative to the configuration, validate all sources first."""
    path = Path(path).resolve()
    raw = read(path)
    allowed = set(DEFAULTS) | {"training_runs", "seeds", "problem", "solver", "inputs_file"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(raw) - allowed)}")
    cfg = {**copy.deepcopy(DEFAULTS), **raw}
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported matched-policy schema_version")
    for name in ("cases", "scan_repetitions", "evaluation_repetitions", "workers"):
        positive_int(cfg[name], name)
    positive_int(cfg["scan_extra_repetitions"], "scan_extra_repetitions", minimum=0)
    if not 0 <= float(cfg["scan_repeat_fraction"]) <= 1:
        raise ValueError("scan_repeat_fraction must be between 0 and 1")
    cfg["weights"] = weight_grid(cfg["weights"])
    if len({fixed_name(w) for w in cfg["weights"]}) != len(cfg["weights"]):
        raise ValueError("Weight names collide at supported precision")
    if not isinstance(cfg["periodic_pattern"], list) or len(cfg["periodic_pattern"]) < 2:
        raise ValueError("periodic_pattern needs at least two weights")
    cfg["periodic_pattern"] = [float(v) for v in cfg["periodic_pattern"]]
    if not all(math.isfinite(v) and v > 0 for v in cfg["periodic_pattern"]):
        raise ValueError("Invalid periodic weights")
    if cfg["rl_beta"] is not None:
        cfg["rl_beta"] = float(cfg["rl_beta"])
        if not math.isfinite(cfg["rl_beta"]) or cfg["rl_beta"] < 0:
            raise ValueError("rl_beta must be finite and nonnegative")
    if set(cfg.get("seeds", {})) != SEED_KEYS:
        raise ValueError(f"Specify all random seeds: {sorted(SEED_KEYS)}")
    for name, value in cfg["seeds"].items():
        positive_int(value, f"seeds.{name}", minimum=0)
    if not isinstance(cfg.get("training_runs"), list) or not cfg["training_runs"]:
        raise ValueError("At least one explicit training run is required")

    def resolve(value):
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

    sources = []
    labels = set()
    for source in cfg["training_runs"]:
        if set(source) - SOURCE_KEYS:
            raise ValueError(f"Unknown training-run keys: {sorted(set(source) - SOURCE_KEYS)}")
        seed = positive_int(source["seed"], "training seed label", minimum=0)
        if seed in labels:
            raise ValueError("Duplicate training seed label")
        labels.add(seed)
        method = source.get("method", "bandit_lstdq")
        if not isinstance(method, str) or not method or any(c in method for c in "/\\"):
            raise ValueError("Invalid source method")
        directory = resolve(source["run_dir"]) if "run_dir" in source else None
        defaults = {} if directory is None else {
            "config": directory / "experiment_config.json",
            "controller_checkpoint": directory / "checkpoints" / f"{method}_final.npz",
            "setup_checkpoint": directory / "final_bandit_states" / f"{method}.npz",
        }
        item = {"seed": seed, "method": method}
        for name in ("config", "controller_checkpoint", "setup_checkpoint"):
            value = resolve(source[name]) if name in source else defaults.get(name)
            if value is None or not value.is_file():
                raise ValueError(f"Missing {name} for seed {seed}: {value}")
            item[name] = str(value)
        encoder = Path(item["controller_checkpoint"] + ".encoder.json")
        if not encoder.is_file():
            raise ValueError(f"Missing controller encoder: {encoder}")
        item["encoder"] = str(encoder)
        training = read(item["config"])
        methods = [m for m in training["methods"] if m["id"] == method]
        if len(methods) != 1 or methods[0]["setup"] != "linucb" or not str(methods[0]["solve"]).startswith("recursive_lstdq"):
            raise ValueError("Module 2 requires a named LinUCB-LSTDQ training source")
        item["training_config"] = {**training, "methods": methods}
        sources.append(item)
    cfg["training_runs"] = sources
    inherited = sources[0]["training_config"]
    cfg["problem"] = cfg.get("problem", inherited["problem"])
    if "problem" not in raw and any(s["training_config"]["problem"] != cfg["problem"] for s in sources):
        raise ValueError("Training problems differ; specify the evaluation problem explicitly")
    solver = raw.get("solver", {})
    if set(solver) - {"tolerance", "max_cycles", "profile"}:
        raise ValueError("Unknown solver setting")
    cfg["solver"] = {
        "tolerance": float(solver.get("tolerance", inherited["solve"]["tolerance"])),
        "max_cycles": solver.get("max_cycles", inherited["solve"]["max_cycles"]),
        "profile": solver.get("profile", inherited["solve"]["smoother_profile"]),
    }
    for setting, source_key in (("tolerance", "tolerance"), ("max_cycles", "max_cycles"),
                                ("profile", "smoother_profile")):
        if setting not in solver and any(
            s["training_config"]["solve"][source_key] != cfg["solver"][setting]
            for s in sources
        ):
            raise ValueError(f"Training solver settings differ; specify solver.{setting}")
    positive_int(cfg["solver"]["max_cycles"], "solver.max_cycles")
    if not math.isfinite(cfg["solver"]["tolerance"]) or cfg["solver"]["tolerance"] <= 0:
        raise ValueError("Solver tolerance must be finite and positive")
    if "inputs_file" in cfg:
        cfg["inputs_file"] = str(resolve(cfg["inputs_file"]))
        if not Path(cfg["inputs_file"]).is_file():
            raise ValueError("inputs_file does not exist")
    cfg["config_path"] = str(path)
    return cfg


def policy_library(cfg):
    policies = {fixed_name(w): {"kind": "fixed", "pattern": [w]} for w in cfg["weights"]}
    policies[fixed_name(1)] = {"kind": "fixed", "pattern": [1.0]}
    policies["periodic"] = {"kind": "periodic", "pattern": cfg["periodic_pattern"]}
    policies["rl_frozen"] = {"kind": "rl", "learn": False, "explore": False, "beta": cfg["rl_beta"]}
    return policies
