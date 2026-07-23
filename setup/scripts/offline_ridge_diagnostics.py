"""
Offline ridge-regression diagnostics for (context, action) -> runtime modeling.

This script is intended to answer the question:
  "Is our linear feature map phi(x,a) even capable of explaining runtime?"

It fits ridge regression on a *time-based* train/test split (first 80% train,
last 20% test) to respect that bandit logs are typically adaptive over time.

Expected input CSV
------------------
Works with logs produced by `utils.setup_amg.run_amg_setup_experiment(...)`
and similar runners. It requires:

  Context x (either):
    - columns: context_0..context_4 (preferred), where
        x = [1, s1, s2, s3, c_diag]
    - OR columns: s1,s2,s3,c_diag (it will prepend intercept 1.0)

  Action a:
    - chosen_strong_threshold
    - chosen_max_row_sum
    - chosen_trunc_factor

  Target y (runtime):
    - by default: loss_used
      (for runtime-loss runs, this is in *seconds*)

Feature maps (paper-ready; v2 centered actions)
-----------------------------------------------
Let a = (th, mxrs, tr) and a0 = (th0, mxrs0, tr0).
Define a_tilde = a - a0 = (th~, mxrs~, tr~).

g(a_tilde) in R^9:
  [ th~,
    mxrs~,
    tr~,
    th~^2,
    mxrs~^2,
    tr~^2,
    th~*mxrs~,
    th~*tr~,
    mxrs~*tr~ ]^T

Context x is:
  x = [1, s1, s2, s3, c_diag]^T

Three ablation models:
  (1) no context: phi_noctx(a) = [1 ; g(a_tilde)]
  (2) context only: phi_ctx(x,a) = [x ; g(a_tilde)]
  (3) full: phi_full(x,a) = [x ; g ; s1*g ; s2*g]
      where s1=x[1], s2=x[2]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray  # (n, 5)
    a: np.ndarray  # (n, 3)  columns: th,mxrs,tr
    y: np.ndarray  # (n,)


def _require_cols(fieldnames: List[str], required: List[str], *, csv_path: Path) -> None:
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")


def load_dataset(csv_path: Path, *, y_col: str) -> Dataset:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(r.fieldnames)

        has_context = all(f"context_{i}" in fieldnames for i in range(5))
        has_scols = all(k in fieldnames for k in ("s1", "s2", "s3", "c_diag"))

        if not (has_context or has_scols):
            raise ValueError(
                "CSV must contain either context_0..context_4 OR s1,s2,s3,c_diag columns."
            )

        _require_cols(
            fieldnames,
            [
                "chosen_strong_threshold",
                "chosen_max_row_sum",
                "chosen_trunc_factor",
                str(y_col),
            ],
            csv_path=csv_path,
        )

        xs: List[List[float]] = []
        aas: List[List[float]] = []
        ys: List[float] = []

        for row in r:
            if has_context:
                x = [float(row[f"context_{i}"]) for i in range(5)]
            else:
                x = [1.0, float(row["s1"]), float(row["s2"]), float(row["s3"]), float(row["c_diag"])]

            th = float(row["chosen_strong_threshold"])
            mxrs = float(row["chosen_max_row_sum"])
            tr = float(row["chosen_trunc_factor"])
            y = float(row[y_col])

            if not (np.isfinite(x).all() and np.isfinite([th, mxrs, tr, y]).all()):
                continue

            xs.append(x)
            aas.append([th, mxrs, tr])
            ys.append(y)

    X = np.asarray(xs, dtype=float)
    A = np.asarray(aas, dtype=float)
    Y = np.asarray(ys, dtype=float)

    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"expected x to have shape (n,5), got {X.shape}")
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"expected a to have shape (n,3), got {A.shape}")
    if Y.ndim != 1 or Y.shape[0] != X.shape[0]:
        raise ValueError("y length must match x rows")

    return Dataset(x=X, a=A, y=Y)


def g_from_a_tilde(a_tilde: np.ndarray) -> np.ndarray:
    th, mxrs, tr = float(a_tilde[0]), float(a_tilde[1]), float(a_tilde[2])
    return np.array(
        [
            th,
            mxrs,
            tr,
            th * th,
            mxrs * mxrs,
            tr * tr,
            th * mxrs,
            th * tr,
            mxrs * tr,
        ],
        dtype=float,
    )


def build_design_matrices(
    data: Dataset,
    *,
    a0: Tuple[float, float, float],
) -> Dict[str, np.ndarray]:
    n = int(data.x.shape[0])
    a0v = np.asarray(a0, dtype=float).reshape(1, 3)
    a_tilde = data.a - a0v  # (n,3)

    G = np.vstack([g_from_a_tilde(a_tilde[i]) for i in range(n)])  # (n,9)

    x = data.x
    s1 = x[:, 1:2]  # (n,1)
    s2 = x[:, 2:3]  # (n,1)

    phi_noctx = np.hstack([np.ones((n, 1), dtype=float), G])  # (n,10)
    phi_ctx = np.hstack([x, G])  # (n,14)
    phi_full = np.hstack([x, G, s1 * G, s2 * G])  # (n,32)

    return {"noctx": phi_noctx, "ctx": phi_ctx, "full": phi_full}


def fit_ridge(Phi: np.ndarray, y: np.ndarray, *, lam: float) -> np.ndarray:
    if lam <= 0.0:
        raise ValueError("lam must be > 0")
    d = int(Phi.shape[1])
    A = Phi.T @ Phi + float(lam) * np.eye(d, dtype=float)
    b = Phi.T @ y
    return np.linalg.solve(A, b)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _quantiles(x: np.ndarray, qs: List[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="path to a bandit log CSV")
    ap.add_argument("--y-col", type=str, default="loss_used", help="column to use as y (runtime in seconds)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="time split fraction for training")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge lambda")
    ap.add_argument("--a0", type=str, default="0.25,0.90,0.00", help="action center: th0,mxrs0,tr0")
    ap.add_argument("--out-dir", type=str, default="", help="output directory (default: alongside CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    a0_parts = [p.strip() for p in str(args.a0).split(",")]
    if len(a0_parts) != 3:
        raise ValueError("--a0 must be 'th0,mxrs0,tr0'")
    a0 = (float(a0_parts[0]), float(a0_parts[1]), float(a0_parts[2]))

    data = load_dataset(csv_path, y_col=str(args.y_col))
    Phi = build_design_matrices(data, a0=a0)

    n = int(data.y.size)
    n_train = int(np.floor(float(args.train_frac) * n))
    n_train = max(1, min(n - 1, n_train))
    train_idx = slice(0, n_train)
    test_idx = slice(n_train, n)

    summary: Dict[str, object] = {
        "csv": str(csv_path),
        "out_dir": str(out_dir),
        "n": int(n),
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "train_frac": float(args.train_frac),
        "y_col": str(args.y_col),
        "lam": float(args.lam),
        "a0": {"th": float(a0[0]), "mxrs": float(a0[1]), "tr": float(a0[2])},
        "r2_test": {},
        "residuals_full_test": {},
    }

    # Fit three ablation models.
    preds: Dict[str, np.ndarray] = {}
    for name, M in Phi.items():
        theta = fit_ridge(M[train_idx], data.y[train_idx], lam=float(args.lam))
        yhat = M[test_idx] @ theta
        preds[name] = yhat
        summary["r2_test"][name] = float(r2_score(data.y[test_idx], yhat))

    # Residual diagnostics for the full model (the one used by Shared LinUCB v2).
    resid_full = data.y[test_idx] - preds["full"]
    summary["residuals_full_test"] = {
        "mean": float(np.mean(resid_full)),
        "std": float(np.std(resid_full)),
        "quantiles": _quantiles(resid_full, [0.01, 0.05, 0.50, 0.95, 0.99]),
        "y_test_quantiles": _quantiles(data.y[test_idx], [0.01, 0.05, 0.50, 0.95, 0.99]),
    }

    summary_path = out_dir / "offline_ridge_diagnostics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # ------------------------------------------------------------------
    # Plots (test set only)
    # ------------------------------------------------------------------
    y_test = data.y[test_idx]

    # Residual vs predicted (3 panels).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, ["noctx", "ctx", "full"]):
        yhat = preds[name]
        resid = y_test - yhat
        ax.scatter(yhat * 1e3, resid * 1e3, s=6, alpha=0.35)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{name}  (R²={summary['r2_test'][name]:.3f})")
        ax.set_xlabel("predicted runtime (ms)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("residual (ms)")
    fig.suptitle("Ridge residuals vs prediction (test split)")
    fig.tight_layout()
    fig.savefig(out_dir / "offline_ridge_residual_vs_pred.png", dpi=256)
    plt.close(fig)

    # Residual histogram (full model).
    plt.figure(figsize=(6.5, 3.8))
    plt.hist(resid_full * 1e3, bins=60, alpha=0.85)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("residual (ms)")
    plt.ylabel("count")
    plt.title("Full model residual histogram (test split)")
    plt.tight_layout()
    plt.savefig(out_dir / "offline_ridge_residual_hist_full.png", dpi=256)
    plt.close()

    # Residual vs action knobs (full model).
    a_test = data.a[test_idx]
    labels = ["th", "mxrs", "tr"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for j, ax in enumerate(axes):
        ax.scatter(a_test[:, j], resid_full * 1e3, s=6, alpha=0.35)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(labels[j])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("residual (ms)")
    fig.suptitle("Full model residuals vs action knobs (test split)")
    fig.tight_layout()
    fig.savefig(out_dir / "offline_ridge_residual_vs_action_full.png", dpi=256)
    plt.close(fig)

    # Console summary.
    print("=== Offline ridge diagnostics ===")
    print("CSV:", csv_path)
    print("out_dir:", out_dir)
    print(f"n={n}  train={n_train}  test={n-n_train}  lam={float(args.lam)}")
    print("Test R^2:")
    for k in ("noctx", "ctx", "full"):
        print(f"  {k:<5} R^2 = {summary['r2_test'][k]:.6f}")
    print("Full-model residuals (test), in ms:")
    q = summary["residuals_full_test"]["quantiles"]
    print(f"  mean={summary['residuals_full_test']['mean']*1e3:.3f}  std={summary['residuals_full_test']['std']*1e3:.3f}")
    print(f"  q01={q['q01']*1e3:.3f}  q05={q['q05']*1e3:.3f}  q50={q['q50']*1e3:.3f}  q95={q['q95']*1e3:.3f}  q99={q['q99']*1e3:.3f}")
    print("Saved:")
    print("  ", summary_path)
    print("  ", out_dir / "offline_ridge_residual_vs_pred.png")
    print("  ", out_dir / "offline_ridge_residual_hist_full.png")
    print("  ", out_dir / "offline_ridge_residual_vs_action_full.png")


if __name__ == "__main__":
    main()
