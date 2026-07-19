from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


METHOD_ORDER = (
    "default_setup_default_solve",
    "bandit_only",
    "fixed_w_1.60",
    "ppo_best",
)
TOTAL_FIELDS = (
    "total_runtime",
    "total_setup_runtime",
    "total_solve_runtime",
    "total_infer_runtime",
    "total_runtime_with_controller",
)


def _repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


REPO = _repo_root()
OUT_DIR = Path(
    os.environ.get("EXP44_RESULTS_ROOT", str(REPO / "results/joint/exp44"))
)


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def env_seeds(name: str, default: str) -> list[int]:
    return [
        int(item.strip())
        for item in env_str(name, default).split(",")
        if item.strip()
    ]


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _result_path(per_seed_dir: Path, run_dir: Path, seed: int) -> Path:
    candidates = (
        per_seed_dir / f"forward_seed{seed}.json",
        run_dir / f"forward_dual_seed{seed}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"missing result file for seed {seed}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_window_result(
    run_dir: Path,
    per_seed_dir: Path,
    seed: int,
    window_name: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    result_path = _result_path(per_seed_dir, run_dir, seed)
    row = json.loads(result_path.read_text(encoding="utf-8"))
    return row["windows"][window_name], row["protocol"], result_path


def _total(method: dict[str, Any], total_key: str, cases: int) -> float:
    if total_key in method:
        return float(method[total_key])
    mean_key = total_key.replace("total_", "mean_", 1)
    if mean_key == "mean_infer_runtime":
        return float(method.get(mean_key, 0.0)) * cases
    if mean_key == "mean_runtime_with_controller":
        mean_value = float(method["mean_runtime"]) + float(
            method.get("mean_infer_runtime", 0.0)
        )
        return mean_value * cases
    return float(method[mean_key]) * cases


def _normalized_method(method: dict[str, Any], fallback_cases: int) -> dict[str, Any]:
    cases = int(method.get("cases", fallback_cases))
    totals = {key: _total(method, key, cases) for key in TOTAL_FIELDS}
    return {
        "cases": cases,
        "mean_runtime": totals["total_runtime"] / cases,
        "total_runtime": totals["total_runtime"],
        "mean_setup_runtime": totals["total_setup_runtime"] / cases,
        "total_setup_runtime": totals["total_setup_runtime"],
        "mean_solve_runtime": totals["total_solve_runtime"] / cases,
        "total_solve_runtime": totals["total_solve_runtime"],
        "mean_infer_runtime": totals["total_infer_runtime"] / cases,
        "total_infer_runtime": totals["total_infer_runtime"],
        "mean_runtime_with_controller": totals[
            "total_runtime_with_controller"
        ]
        / cases,
        "total_runtime_with_controller": totals["total_runtime_with_controller"],
        "failed_count": int(method.get("failed_count", 0)),
    }


def _improvement(reference: dict[str, Any], candidate: dict[str, Any], key: str) -> float:
    return 100.0 * (float(reference[key]) - float(candidate[key])) / float(
        reference[key]
    )


def _relative_metrics(combined: dict[str, dict[str, Any]]) -> dict[str, float]:
    relative: dict[str, float] = {}
    default = combined.get("default_setup_default_solve")
    bandit = combined.get("bandit_only")
    fixed = combined.get("fixed_w_1.60")
    ppo = combined.get("ppo_best")
    if default and bandit:
        relative["bandit_vs_default_setup"] = _improvement(
            default, bandit, "mean_runtime"
        )
    if default and fixed:
        relative["fixed_w_1.60_vs_default_setup"] = _improvement(
            default, fixed, "mean_runtime"
        )
    if default and ppo:
        relative["ppo_best_vs_default_setup"] = _improvement(
            default, ppo, "mean_runtime"
        )
    if bandit and fixed:
        relative["fixed_w_1.60_vs_bandit"] = _improvement(
            bandit, fixed, "mean_runtime"
        )
    if bandit and ppo:
        relative["ppo_best_vs_bandit"] = _improvement(
            bandit, ppo, "mean_runtime"
        )
        relative["ppo_best_with_controller_vs_bandit"] = _improvement(
            bandit, ppo, "mean_runtime_with_controller"
        )
    if fixed and ppo:
        relative["ppo_best_vs_fixed_w_1.60"] = _improvement(
            fixed, ppo, "mean_runtime"
        )
        relative["ppo_best_with_controller_vs_fixed_w_1.60"] = _improvement(
            fixed, ppo, "mean_runtime_with_controller"
        )
    return relative


def aggregate_window(
    run_dir: Path,
    per_seed_dir: Path,
    window_name: str,
    seeds: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    combined_totals: dict[str, dict[str, float | int]] = {
        method: {**{key: 0.0 for key in TOTAL_FIELDS}, "cases": 0, "failed_count": 0}
        for method in METHOD_ORDER
    }

    for seed in seeds:
        window, protocol, source_path = load_window_result(
            run_dir, per_seed_dir, seed, window_name
        )
        fallback_cases = int(
            protocol.get("eval_cases", protocol.get("cases_per_seed", 0))
        )
        row_methods: dict[str, Any] = {}
        for method_name in METHOD_ORDER:
            if method_name not in window["methods"]:
                continue
            method = _normalized_method(window["methods"][method_name], fallback_cases)
            row_methods[method_name] = method
            acc = combined_totals[method_name]
            acc["cases"] = int(acc["cases"]) + int(method["cases"])
            acc["failed_count"] = int(acc["failed_count"]) + int(
                method["failed_count"]
            )
            for total_key in TOTAL_FIELDS:
                acc[total_key] = float(acc[total_key]) + float(method[total_key])
        rows.append(
            {
                "seed": int(seed),
                "source_path": str(source_path),
                "methods": row_methods,
                "relative_pct": dict(window.get("relative_pct", {})),
            }
        )

    combined: dict[str, Any] = {}
    for method_name in METHOD_ORDER:
        acc = combined_totals[method_name]
        cases = int(acc["cases"])
        if cases <= 0:
            continue
        combined[method_name] = {
            "cases": cases,
            "mean_runtime": float(acc["total_runtime"]) / cases,
            "total_runtime": float(acc["total_runtime"]),
            "mean_setup_runtime": float(acc["total_setup_runtime"]) / cases,
            "total_setup_runtime": float(acc["total_setup_runtime"]),
            "mean_solve_runtime": float(acc["total_solve_runtime"]) / cases,
            "total_solve_runtime": float(acc["total_solve_runtime"]),
            "mean_infer_runtime": float(acc["total_infer_runtime"]) / cases,
            "total_infer_runtime": float(acc["total_infer_runtime"]),
            "mean_runtime_with_controller": float(
                acc["total_runtime_with_controller"]
            )
            / cases,
            "total_runtime_with_controller": float(
                acc["total_runtime_with_controller"]
            ),
            "failed_count": int(acc["failed_count"]),
        }

    cases_per_seed = int(rows[0]["methods"][next(iter(rows[0]["methods"]))]["cases"])
    return {
        "cases_per_seed": cases_per_seed,
        "rows": rows,
        "combined": combined,
        "relative_pct": _relative_metrics(combined),
    }


def main() -> None:
    run_dir = env_path("RUN_DIR", OUT_DIR)
    per_seed_dir = env_path("PER_SEED_DIR", run_dir / "per_seed")
    seeds = env_seeds(
        "FORWARD_SEEDS", "39393939,39394939,39400939,39406939,39412939"
    )
    eval_a_name = env_str("EVAL_A_NAME", "eval_1000")
    eval_b_name = env_str("EVAL_B_NAME", "eval_1000_dup")
    window_names = list(dict.fromkeys((eval_a_name, eval_b_name)))
    result_path = env_path(
        "SUMMARY_PATH", run_dir / "forward_continuation_summary.json"
    )
    result = {
        "stage": "summary",
        "protocol": "forward continuation from archived model snapshot",
        "run_dir": str(run_dir),
        "per_seed_dir": str(per_seed_dir),
        "seeds": seeds,
        "primary_window": env_str("PRIMARY_WINDOW", eval_a_name),
        "windows": {
            name: aggregate_window(run_dir, per_seed_dir, name, seeds)
            for name in window_names
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "result_write", "path": str(result_path)}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
