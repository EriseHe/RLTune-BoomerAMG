"""Minimal high-level BoomerAMG bandit runner."""

from __future__ import annotations

import csv
import itertools
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_AMG_bandit import _ensure_hypre_env, _ensure_mpi_env
from solvers.BoomerAMG import boomeramg


LossFn = Callable[[float], float]
SolverFn = Callable[[Any, np.ndarray, np.ndarray, Mapping[str, float], float], tuple[Any, ...]]
ProblemLike = Mapping[str, Any] | tuple[Any, Any] | list[Any]


def loss_raw(wu: float) -> float:
    return 1.0 + float(wu)


def loss_linear(knee: float = 40.0) -> LossFn:
    if knee <= 0:
        raise ValueError("knee must be positive")
    return lambda wu: 1.0 + float(wu / knee)


def loss_log(knee: float = 40.0) -> LossFn:
    if knee <= 0:
        raise ValueError("knee must be positive")
    return lambda wu: 1.0 + float(np.log1p(wu / knee))


def _normalize_problem(problem: ProblemLike) -> tuple[Any, np.ndarray, str]:
    if isinstance(problem, Mapping):
        if "A" not in problem or "b" not in problem:
            raise ValueError("Problem mapping must include 'A' and 'b'")
        return problem["A"], np.asarray(problem["b"], dtype=np.float64).reshape(-1), str(problem.get("id", ""))
    if isinstance(problem, (tuple, list)) and len(problem) >= 2:
        pid = str(problem[2]) if len(problem) > 2 else ""
        return problem[0], np.asarray(problem[1], dtype=np.float64).reshape(-1), pid
    raise TypeError("Each problem must be {'A':..., 'b':...} or (A, b[, id])")


def _as_config(action: Any, param_names: Sequence[str]) -> Dict[str, float]:
    if isinstance(action, Mapping):
        return {name: float(action[name]) for name in param_names}
    if len(param_names) == 1:
        return {param_names[0]: float(action)}
    if isinstance(action, (tuple, list, np.ndarray)) and len(action) == len(param_names):
        return {param_names[i]: float(action[i]) for i in range(len(param_names))}
    raise TypeError("Action format incompatible with tune_space parameter names")


def _build_tune_space(tune_space: Any) -> tuple[list[str], np.ndarray]:
    if isinstance(tune_space, Mapping):
        if "actions" in tune_space:
            actions = list(tune_space["actions"])
            if not actions:
                raise ValueError("tune_space['actions'] cannot be empty")
            if "params" in tune_space:
                param_names = [str(x) for x in tune_space["params"]]
            elif isinstance(actions[0], Mapping):
                param_names = [str(k) for k in actions[0].keys()]
            else:
                raise ValueError("Provide tune_space['params'] when actions are not mappings")
            normalized = [_as_config(a, param_names) for a in actions]
            return param_names, np.asarray(normalized, dtype=object)

        param_names = [str(k) for k in tune_space.keys()]
        if not param_names:
            raise ValueError("tune_space mapping cannot be empty")
        values = [list(v) for v in tune_space.values()]
        if any(len(v) == 0 for v in values):
            raise ValueError("Each tune_space parameter must have at least one candidate")
        actions = []
        for combo in itertools.product(*values):
            actions.append({param_names[i]: float(combo[i]) for i in range(len(param_names))})
        return param_names, np.asarray(actions, dtype=object)

    arr = np.asarray(tune_space, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("tune_space cannot be empty")
    return ["strong_threshold"], arr


def _resolve_bandit_factory(spec: Any) -> Callable[[np.ndarray, int], Any]:
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"tsallis_amg", "amg", "tsallisinf_amg"}:
            from learners.TsallisINF_AMG import TsallisINF_AMG

            return lambda grid, horizon: TsallisINF_AMG(grid, horizon)
        if key in {"tsallis_sor", "sor", "tsallisinf", "tsallisinf_sor"}:
            from learners.TsallisINF_SOR import TsallisINF

            return lambda grid, horizon: TsallisINF(grid, horizon)
        raise ValueError(f"Unknown bandit string: {spec}")

    if callable(spec):
        def _factory(grid: np.ndarray, horizon: int):
            try:
                model = spec(grid, horizon)
            except TypeError:
                model = spec(grid)
            return model

        return _factory

    if hasattr(spec, "predict") and hasattr(spec, "update"):
        return lambda _grid, _horizon: spec

    raise TypeError("bandit must be a string, callable factory, or object with predict/update")


def _resolve_bandits(bandit: Any, action_grid: np.ndarray, horizon: int) -> Dict[str, Any]:
    if isinstance(bandit, Mapping):
        specs = bandit
    elif isinstance(bandit, (list, tuple)) and not isinstance(bandit, str):
        specs = {f"bandit_{i}": b for i, b in enumerate(bandit)}
    else:
        name = bandit if isinstance(bandit, str) else "bandit"
        specs = {str(name): bandit}

    out: Dict[str, Any] = {}
    for name, spec in specs.items():
        model = _resolve_bandit_factory(spec)(action_grid, horizon)
        if not (hasattr(model, "predict") and hasattr(model, "update")):
            raise TypeError(f"Bandit '{name}' does not provide predict()/update()")
        out[str(name)] = model
    return out


def _solver_call(
    *,
    solver_fn: SolverFn | None,
    A: Any,
    b: np.ndarray,
    x0: np.ndarray,
    config: Mapping[str, float],
    epsilon: float,
    param_names: Sequence[str],
) -> tuple[float, float]:
    if solver_fn is None:
        if param_names != ["strong_threshold"]:
            raise ValueError(
                "Default solver only supports {'strong_threshold': ...}. "
                "Pass solver_fn for multi-parameter tuning."
            )
        k, comp, _ = boomeramg(A, b, x0, float(config["strong_threshold"]), epsilon)
    else:
        result = solver_fn(A, b, x0, config, epsilon)
        if len(result) < 2:
            raise ValueError("solver_fn must return at least (iterations, complexity, ...)")
        k, comp = result[0], result[1]
    return float(k), float(comp)


def _plot_learning(
    *,
    baseline_wu: np.ndarray,
    baseline_labels: Sequence[str],
    bandit_wu: Mapping[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    T = baseline_wu.shape[0]
    y = T - np.arange(1, T + 1)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_labels)))
    for i, label in enumerate(baseline_labels):
        plt.plot(np.cumsum(baseline_wu[:, i]), y, "--", linewidth=2, color=colors[i], label=label)
    for name, wu in bandit_wu.items():
        plt.plot(np.cumsum(wu), y, linewidth=2.7, label=name)
    plt.xlabel("Total Work Units (WU)")
    plt.ylabel("Instances Remaining")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=256)
    plt.close()


def run_boomeramg(
    problem_set: Sequence[ProblemLike],
    bandit: Any,
    loss_fn: LossFn,
    tune_space: Any,
    enable_logs: bool = False,
    *,
    epsilon: float = 1e-8,
    solver_fn: SolverFn | None = None,
    baseline_space: Any | None = None,
    out_dir: str | Path | None = None,
    run_tag: str = "run",
    progress_every: int = 100,
    hypre_lib: str | None = None,
) -> Dict[str, Any]:
    """
    High-level BoomerAMG setup-bandit runner.

    Required arguments match your requested interface:
      run_boomeramg(problem_set, bandit, loss_fn, tune_space, enable_logs=False)

    tune_space forms:
      - single parameter: [0.1, 0.2, 0.3]  -> strong_threshold
      - multi parameter mapping: {"p1": [...], "p2": [...]} (cartesian product)
      - explicit actions: {"params": [...], "actions": [dict|tuple, ...]}

    baseline_space defaults to tune_space.
    """
    if not problem_set:
        raise ValueError("problem_set cannot be empty")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    _ensure_hypre_env(hypre_lib)
    _ensure_mpi_env()

    param_names, action_grid = _build_tune_space(tune_space)
    _, baseline_grid = _build_tune_space(baseline_space if baseline_space is not None else tune_space)

    models = _resolve_bandits(bandit, action_grid, len(problem_set))

    T = len(problem_set)
    baseline_wu = np.zeros((T, len(baseline_grid)), dtype=np.float64)
    bandit_wu = {name: np.zeros(T, dtype=np.float64) for name in models}
    bandit_loss = {name: np.zeros(T, dtype=np.float64) for name in models}
    bandit_action = {name: [None] * T for name in models}
    problem_ids: list[str] = []

    for t, problem in enumerate(problem_set):
        A, b, pid = _normalize_problem(problem)
        problem_ids.append(pid)
        x0 = np.zeros(A.shape[0], dtype=np.float64)

        for i, base_action in enumerate(baseline_grid):
            base_cfg = _as_config(base_action, param_names)
            k, comp = _solver_call(
                solver_fn=solver_fn,
                A=A,
                b=b,
                x0=x0,
                config=base_cfg,
                epsilon=epsilon,
                param_names=param_names,
            )
            baseline_wu[t, i] = k * comp

        for name, model in models.items():
            action = model.predict()
            cfg = _as_config(action, param_names)
            k, comp = _solver_call(
                solver_fn=solver_fn,
                A=A,
                b=b,
                x0=x0,
                config=cfg,
                epsilon=epsilon,
                param_names=param_names,
            )
            wu = k * comp
            loss = float(loss_fn(wu))
            model.update(loss)

            bandit_wu[name][t] = wu
            bandit_loss[name][t] = loss
            bandit_action[name][t] = cfg

        if progress_every > 0 and ((t + 1) % progress_every == 0 or (t + 1) == T):
            print(f"{run_tag}: {t + 1}/{T}")

    out_root = Path(out_dir) if out_dir else (Path(__file__).resolve().parents[1] / "plots" / "BoomerAMG")
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_labels = []
    for action in baseline_grid:
        cfg = _as_config(action, param_names)
        baseline_labels.append(", ".join([f"{k}={cfg[k]:.3g}" for k in param_names]))

    plot_path = out_root / f"learning_{run_tag}_T{T}.png"
    _plot_learning(
        baseline_wu=baseline_wu,
        baseline_labels=baseline_labels,
        bandit_wu=bandit_wu,
        out_path=plot_path,
        title=f"BoomerAMG learning ({run_tag})",
    )

    best_fixed_idx = int(np.argmin(np.mean(baseline_wu, axis=0)))
    best_fixed_series = baseline_wu[:, best_fixed_idx]
    best_fixed_config = _as_config(baseline_grid[best_fixed_idx], param_names)

    summary: Dict[str, Dict[str, float]] = {}
    for name, wu in bandit_wu.items():
        summary[name] = {
            "avg_wu": float(np.mean(wu)),
            "cumulative_wu": float(np.sum(wu)),
            "avg_regret_vs_best_fixed": float(np.mean(wu - best_fixed_series)),
        }

    out: Dict[str, Any] = {
        "summary": summary,
        "param_names": param_names,
        "best_fixed_config": best_fixed_config,
        "best_fixed_avg_wu": float(np.mean(best_fixed_series)),
        "plot_path": str(plot_path),
        "baseline_wu": baseline_wu,
        "bandit_wu": bandit_wu,
        "bandit_loss": bandit_loss,
        "bandit_action": bandit_action,
    }

    if enable_logs:
        log_path = out_root / f"log_{run_tag}_T{T}.csv"
        fields = ["instance", "problem_id"]
        for i in range(len(baseline_grid)):
            fields.append(f"baseline_{i}_WU")
        for name in models:
            fields.extend([f"{name}_WU", f"{name}_loss", f"{name}_action"])

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in range(T):
                row: Dict[str, Any] = {"instance": t + 1, "problem_id": problem_ids[t]}
                for i in range(len(baseline_grid)):
                    row[f"baseline_{i}_WU"] = float(baseline_wu[t, i])
                for name in models:
                    row[f"{name}_WU"] = float(bandit_wu[name][t])
                    row[f"{name}_loss"] = float(bandit_loss[name][t])
                    row[f"{name}_action"] = str(bandit_action[name][t])
                writer.writerow(row)
        out["log_csv_path"] = str(log_path)

    return out


# Optional compatibility alias with previous naming.
def run_boomeramg_on_problem_set(*args, **kwargs):
    return run_boomeramg(*args, **kwargs)


__all__ = [
    "run_boomeramg",
    "run_boomeramg_on_problem_set",
    "loss_raw",
    "loss_linear",
    "loss_log",
]
