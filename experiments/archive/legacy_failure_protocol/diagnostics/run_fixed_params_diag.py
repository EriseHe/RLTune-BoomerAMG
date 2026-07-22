from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mode:
    name: str
    params: dict[str, Any]


DEFAULT_PARAMS: dict[str, Any] = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
}

FIXED_MODE_V2_19CUBE: dict[str, Any] = {
    "strong_threshold": 0.50,
    "max_row_sum": 1e-6,
    "trunc_factor": 0.999999,
    "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
    "interp_type": DEFAULT_PARAMS["interp_type"],
}

FIXED_MODE_TSALLIS_3CUBE: dict[str, Any] = {
    "strong_threshold": 0.05,
    "max_row_sum": 0.10,
    "trunc_factor": 0.80,
    "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
    "interp_type": DEFAULT_PARAMS["interp_type"],
}

MODES: list[Mode] = [
    Mode("fixed_(th,mxrs,tr)=(0.50,1e-6,0.999999)", FIXED_MODE_V2_19CUBE),
    Mode("fixed_(th,mxrs,tr)=(0.05,0.10,0.80)", FIXED_MODE_TSALLIS_3CUBE),
]


def _import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433 (dynamic import)

        return plt
    except Exception:
        return None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _percentiles(values: np.ndarray, ps: tuple[int, ...] = (0, 25, 50, 75, 100)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    q = np.percentile(values, list(ps))
    return {f"p{p}": float(v) for p, v in zip(ps, q)}


def _summarize_mode(rows: list[dict[str, Any]], *, mode_name: str) -> dict[str, Any]:
    mode_rows = [r for r in rows if r.get("mode") == mode_name]
    iters = np.array([r.get("iterations", np.nan) for r in mode_rows], dtype=float)
    wu = np.array([r.get("work_units", np.nan) for r in mode_rows], dtype=float)
    rt = np.array([r.get("runtime_sec", np.nan) for r in mode_rows], dtype=float)
    res = np.array([r.get("residual_norm", np.nan) for r in mode_rows], dtype=float)
    apc = np.array([r.get("ap_complexity", np.nan) for r in mode_rows], dtype=float)
    glc = np.array([r.get("grid_complexity", np.nan) for r in mode_rows], dtype=float)
    lev = np.array([r.get("num_levels", np.nan) for r in mode_rows], dtype=float)
    failed = np.array([bool(r.get("failed", False)) for r in mode_rows], dtype=bool)

    def _finite(x: np.ndarray) -> np.ndarray:
        return x[np.isfinite(x)]

    return {
        "count": int(len(mode_rows)),
        "failed_count": int(np.sum(failed)),
        "iterations": {
            "mean": float(np.mean(_finite(iters))) if _finite(iters).size else float("nan"),
            **_percentiles(_finite(iters)),
        },
        "work_units": {
            "mean": float(np.mean(_finite(wu))) if _finite(wu).size else float("nan"),
            **_percentiles(_finite(wu)),
        },
        "runtime_sec": {
            "mean": float(np.mean(_finite(rt))) if _finite(rt).size else float("nan"),
            **_percentiles(_finite(rt)),
        },
        "residual_norm": {
            "mean": float(np.mean(_finite(res))) if _finite(res).size else float("nan"),
            **_percentiles(_finite(res)),
        },
        "ap_complexity": {
            "mean": float(np.mean(_finite(apc))) if _finite(apc).size else float("nan"),
            **_percentiles(_finite(apc)),
        },
        "grid_complexity": {
            "mean": float(np.mean(_finite(glc))) if _finite(glc).size else float("nan"),
            **_percentiles(_finite(glc)),
        },
        "num_levels": {
            "mean": float(np.mean(_finite(lev))) if _finite(lev).size else float("nan"),
            **_percentiles(_finite(lev)),
        },
    }


def _main() -> int:
    # Fixed request defaults (override via env vars if needed).
    T = int(os.environ.get("T", "5000"))
    SEED = int(os.environ.get("SEED", "20260209"))
    NX = int(os.environ.get("NX", "60"))
    NY = int(os.environ.get("NY", "60"))
    NZ = int(os.environ.get("NZ", "1"))
    STENCIL = int(os.environ.get("STENCIL", "27"))
    SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
    SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))

    repo_root = Path(__file__).resolve().parents[2]
    learning_setup_dir = repo_root / "SetupPhase"
    sys.path.insert(0, str(learning_setup_dir))

    from problems.amg import stencil_27_laplace  # noqa: WPS433 (runtime import)

    solver_dir = learning_setup_dir / "solver"
    amg_lib_path = solver_dir / "libamg_setup_solver.dylib"
    if sys.platform == "win32":
        hypre_dll_path = solver_dir / "HYPRE.dll"
    elif sys.platform == "darwin":
        hypre_dll_path = repo_root / "hypre" / "install" / "lib" / "libHYPRE.dylib"
    else:
        hypre_dll_path = repo_root / "hypre" / "install" / "lib" / "libHYPRE.so"

    if not amg_lib_path.exists():
        raise FileNotFoundError(f"Missing compiled solver library: {amg_lib_path}")
    if not hypre_dll_path.exists():
        raise FileNotFoundError(f"Missing HYPRE library: {hypre_dll_path}")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(solver_dir))

    import ctypes  # noqa: WPS433 (runtime import)

    _VP = ctypes.c_void_p
    _I = ctypes.c_int
    _D = ctypes.c_double
    _ULL = ctypes.c_ulonglong
    _BIG = ctypes.c_longlong

    # -- load libraries -----------------------------------------------------
    amg = ctypes.CDLL(str(amg_lib_path))
    hypre = ctypes.CDLL(str(hypre_dll_path))

    # -- amg_setup_solver exports ------------------------------------------
    amg.amg_setup_create.restype = _VP
    amg.amg_setup_create.argtypes = [
        _I,
        _I,
        _I,  # nx,ny,nz
        _I,  # stencil_type
        _I,
        _ULL,  # rhs_type, rhs_seed
        _D,
        _D,  # k, c
        _D,
        _D,
        _D,
        _D,  # a0,a1,a2,a3
    ]

    amg.amg_setup_destroy.restype = None
    amg.amg_setup_destroy.argtypes = [_VP]

    amg.amg_setup_get_n.restype = _I
    amg.amg_setup_get_n.argtypes = [_VP]

    amg.amg_setup_get_nnz.restype = _I
    amg.amg_setup_get_nnz.argtypes = [_VP]

    # -- env layout (only the tail we need) --------------------------------
    # We intentionally skip the first 16 bytes (comm + local_num_rows + nnz + padding)
    # to avoid depending on MPI_Comm size/ABI in different HYPRE builds.
    class _AMGSetupEnv(ctypes.Structure):
        _fields_ = [
            ("_pad0", ctypes.c_byte * 16),
            ("ij_A", _VP),
            ("A", _VP),
            ("ij_b", _VP),
            ("ij_x", _VP),
            ("b", _VP),
            ("x", _VP),
            ("first_row", _BIG),
            ("last_row", _BIG),
        ]

    def _check(rc: int, name: str) -> None:
        if int(rc) != 0:
            raise RuntimeError(f"{name} failed (rc={int(rc)})")

    # -- HYPRE BoomerAMG calls ---------------------------------------------
    hypre.HYPRE_BoomerAMGCreate.restype = _I
    hypre.HYPRE_BoomerAMGCreate.argtypes = [ctypes.POINTER(_VP)]

    hypre.HYPRE_BoomerAMGDestroy.restype = _I
    hypre.HYPRE_BoomerAMGDestroy.argtypes = [_VP]

    hypre.HYPRE_BoomerAMGSetPrintLevel.restype = _I
    hypre.HYPRE_BoomerAMGSetPrintLevel.argtypes = [_VP, _I]

    hypre.HYPRE_BoomerAMGSetTol.restype = _I
    hypre.HYPRE_BoomerAMGSetTol.argtypes = [_VP, _D]

    hypre.HYPRE_BoomerAMGSetMaxIter.restype = _I
    hypre.HYPRE_BoomerAMGSetMaxIter.argtypes = [_VP, _I]

    hypre.HYPRE_BoomerAMGSetCumNnzAP.restype = _I
    hypre.HYPRE_BoomerAMGSetCumNnzAP.argtypes = [_VP, _D]

    hypre.HYPRE_BoomerAMGSetStrongThreshold.restype = _I
    hypre.HYPRE_BoomerAMGSetStrongThreshold.argtypes = [_VP, _D]

    hypre.HYPRE_BoomerAMGSetMaxRowSum.restype = _I
    hypre.HYPRE_BoomerAMGSetMaxRowSum.argtypes = [_VP, _D]

    hypre.HYPRE_BoomerAMGSetTruncFactor.restype = _I
    hypre.HYPRE_BoomerAMGSetTruncFactor.argtypes = [_VP, _D]

    hypre.HYPRE_BoomerAMGSetCoarsenType.restype = _I
    hypre.HYPRE_BoomerAMGSetCoarsenType.argtypes = [_VP, _I]

    hypre.HYPRE_BoomerAMGSetInterpType.restype = _I
    hypre.HYPRE_BoomerAMGSetInterpType.argtypes = [_VP, _I]

    hypre.HYPRE_BoomerAMGSetup.restype = _I
    hypre.HYPRE_BoomerAMGSetup.argtypes = [_VP, _VP, _VP, _VP]  # solver, A, b, x

    hypre.HYPRE_BoomerAMGSolve.restype = _I
    hypre.HYPRE_BoomerAMGSolve.argtypes = [_VP, _VP, _VP, _VP]

    hypre.HYPRE_BoomerAMGGetNumIterations.restype = _I
    hypre.HYPRE_BoomerAMGGetNumIterations.argtypes = [_VP, ctypes.POINTER(_I)]

    hypre.HYPRE_BoomerAMGGetFinalRelativeResidualNorm.restype = _I
    hypre.HYPRE_BoomerAMGGetFinalRelativeResidualNorm.argtypes = [_VP, ctypes.POINTER(_D)]

    hypre.HYPRE_BoomerAMGGetCumNnzAP.restype = _I
    hypre.HYPRE_BoomerAMGGetCumNnzAP.argtypes = [_VP, ctypes.POINTER(_D)]

    hypre.HYPRE_BoomerAMGGetGridHierarchy.restype = _I
    hypre.HYPRE_BoomerAMGGetGridHierarchy.argtypes = [_VP, ctypes.POINTER(_I)]

    # -- IJ vector ops for x reset -----------------------------------------
    hypre.HYPRE_IJVectorSetValues.restype = _I
    hypre.HYPRE_IJVectorSetValues.argtypes = [_VP, _I, ctypes.POINTER(_BIG), ctypes.POINTER(_D)]

    hypre.HYPRE_IJVectorAssemble.restype = _I
    hypre.HYPRE_IJVectorAssemble.argtypes = [_VP]

    # ---------------------------------------------------------------------
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(repo_root / "results" / "diagnostics" / "fixed_params"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up solve to pay one-time HYPRE init + JIT-y costs.
    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_27_laplace(rng=warm_rng, nx=NX, ny=NY, nz=NZ)

    def _env_create(mkw: dict[str, Any]) -> _VP:
        return amg.amg_setup_create(
            int(mkw["nx"]),
            int(mkw["ny"]),
            int(mkw["nz"]),
            int(mkw["stencil"]),
            int(mkw["rhs_type"]),
            int(mkw["rhs_seed"]),
            float(mkw["k"]),
            float(mkw["c"]),
            float(mkw["a0"]),
            float(mkw["a1"]),
            float(mkw["a2"]),
            float(mkw["a3"]),
        )

    def _reset_x(env: _AMGSetupEnv, *, n0: int, rows0: np.ndarray, zeros0: np.ndarray) -> None:
        _check(
            hypre.HYPRE_IJVectorSetValues(
                env.ij_x,
                int(n0),
                rows0.ctypes.data_as(ctypes.POINTER(_BIG)),
                zeros0.ctypes.data_as(ctypes.POINTER(_D)),
            ),
            "HYPRE_IJVectorSetValues(x=0)",
        )
        _check(hypre.HYPRE_IJVectorAssemble(env.ij_x), "HYPRE_IJVectorAssemble(x)")

    def _solve_one(env_ptr: _VP, env: _AMGSetupEnv, *, n0: int, nnz_a0: int, rows0: np.ndarray, zeros0: np.ndarray, cgrid: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        _reset_x(env, n0=n0, rows0=rows0, zeros0=zeros0)

        solver = _VP()
        rc = hypre.HYPRE_BoomerAMGCreate(ctypes.byref(solver))
        if int(rc) != 0 or not solver:
            return {"failed": True, "error": f"HYPRE_BoomerAMGCreate rc={int(rc)}"}

        try:
            _check(hypre.HYPRE_BoomerAMGSetPrintLevel(solver, 0), "HYPRE_BoomerAMGSetPrintLevel")
            _check(hypre.HYPRE_BoomerAMGSetTol(solver, float(SOLVER_TOL)), "HYPRE_BoomerAMGSetTol")
            _check(hypre.HYPRE_BoomerAMGSetMaxIter(solver, int(SOLVER_MAX_ITER)), "HYPRE_BoomerAMGSetMaxIter")
            _check(hypre.HYPRE_BoomerAMGSetCumNnzAP(solver, 1.0), "HYPRE_BoomerAMGSetCumNnzAP")

            _check(
                hypre.HYPRE_BoomerAMGSetStrongThreshold(solver, float(params["strong_threshold"])),
                "HYPRE_BoomerAMGSetStrongThreshold",
            )
            _check(
                hypre.HYPRE_BoomerAMGSetMaxRowSum(solver, float(params["max_row_sum"])),
                "HYPRE_BoomerAMGSetMaxRowSum",
            )
            _check(
                hypre.HYPRE_BoomerAMGSetTruncFactor(solver, float(params["trunc_factor"])),
                "HYPRE_BoomerAMGSetTruncFactor",
            )
            _check(
                hypre.HYPRE_BoomerAMGSetCoarsenType(solver, int(params["coarsen_type"])),
                "HYPRE_BoomerAMGSetCoarsenType",
            )
            _check(
                hypre.HYPRE_BoomerAMGSetInterpType(solver, int(params["interp_type"])),
                "HYPRE_BoomerAMGSetInterpType",
            )

            t0 = time.perf_counter()
            _check(hypre.HYPRE_BoomerAMGSetup(solver, env.A, env.b, env.x), "HYPRE_BoomerAMGSetup")
            _check(hypre.HYPRE_BoomerAMGSolve(solver, env.A, env.b, env.x), "HYPRE_BoomerAMGSolve")
            t1 = time.perf_counter()

            iters = _I()
            res = _D()
            cum = _D()
            _check(hypre.HYPRE_BoomerAMGGetNumIterations(solver, ctypes.byref(iters)), "HYPRE_BoomerAMGGetNumIterations")
            _check(
                hypre.HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, ctypes.byref(res)),
                "HYPRE_BoomerAMGGetFinalRelativeResidualNorm",
            )
            _check(hypre.HYPRE_BoomerAMGGetCumNnzAP(solver, ctypes.byref(cum)), "HYPRE_BoomerAMGGetCumNnzAP")

            # Optional diagnostic: grid hierarchy (for num_levels + grid_complexity).
            _check(
                hypre.HYPRE_BoomerAMGGetGridHierarchy(solver, cgrid.ctypes.data_as(ctypes.POINTER(_I))),
                "HYPRE_BoomerAMGGetGridHierarchy",
            )
            max_level = int(np.max(cgrid)) if cgrid.size else -1
            num_levels = int(max_level + 1) if max_level >= 0 else 0
            if num_levels > 0:
                n_per_level = np.array([np.count_nonzero(cgrid >= lvl) for lvl in range(num_levels)], dtype=float)
                grid_complexity = float(np.sum(n_per_level) / n_per_level[0]) if n_per_level[0] > 0 else float("nan")
            else:
                grid_complexity = float("nan")

            cum_nnz_ap = float(cum.value)
            ap_complexity = float(cum_nnz_ap / float(nnz_a0)) if int(nnz_a0) > 0 else float("nan")

            residual_norm = float(res.value)
            iterations = int(iters.value)
            converged = bool(np.isfinite(residual_norm) and residual_norm <= float(SOLVER_TOL) and iterations < int(SOLVER_MAX_ITER))
            work_units = float(iterations) * float(ap_complexity) if np.isfinite(ap_complexity) else float("nan")

            return {
                "failed": (not converged),
                "runtime_sec": float(t1 - t0),
                "iterations": iterations,
                "residual_norm": residual_norm,
                "nnz_a0": int(nnz_a0),
                "cum_nnz_ap": cum_nnz_ap,
                "ap_complexity": ap_complexity,
                "work_units": work_units,
                "num_levels": int(num_levels),
                "grid_complexity": float(grid_complexity),
                # NOTE: operator_complexity is not exposed via public HYPRE getters in this build.
                "operator_complexity": None,
            }
        except Exception as e:  # noqa: BLE001 (diagnostic script)
            return {"failed": True, "error": repr(e)}
        finally:
            try:
                hypre.HYPRE_BoomerAMGDestroy(solver)
            except Exception:
                pass

    # Warm up once (not logged)
    warm_env_ptr = _env_create(warm_mkw)
    if warm_env_ptr:
        try:
            warm_env = ctypes.cast(warm_env_ptr, ctypes.POINTER(_AMGSetupEnv)).contents
            warm_n0 = int(amg.amg_setup_get_n(warm_env_ptr))
            warm_first = int(warm_env.first_row)
            warm_rows0 = np.arange(warm_first, warm_first + warm_n0, dtype=np.int64)
            warm_zeros0 = np.zeros(warm_n0, dtype=np.float64)
            warm_cgrid = np.empty(warm_n0, dtype=np.int32)
            _ = _solve_one(
                warm_env_ptr,
                warm_env,
                n0=warm_n0,
                nnz_a0=int(amg.amg_setup_get_nnz(warm_env_ptr)),
                rows0=warm_rows0,
                zeros0=warm_zeros0,
                cgrid=warm_cgrid,
                params=DEFAULT_PARAMS,
            )
        finally:
            amg.amg_setup_destroy(warm_env_ptr)

    rng_inst = np.random.default_rng(SEED)
    logs: list[dict[str, Any]] = []

    progress_every = max(1, T // 100)
    for t in range(1, T + 1):
        mkw, _context, meta = stencil_27_laplace(rng=rng_inst, nx=NX, ny=NY, nz=NZ, stencil=STENCIL)
        env_ptr = _env_create(mkw)
        if not env_ptr:
            for mode in MODES:
                logs.append(
                    {
                        "t": int(t),
                        "mode": mode.name,
                        "failed": True,
                        "error": "amg_setup_create failed",
                        **{f"param_{k}": v for k, v in mode.params.items()},
                        **{f"meta_{k}": v for k, v in meta.items()},
                    }
                )
            continue

        try:
            env = ctypes.cast(env_ptr, ctypes.POINTER(_AMGSetupEnv)).contents
            n0 = int(amg.amg_setup_get_n(env_ptr))
            nnz_a0 = int(amg.amg_setup_get_nnz(env_ptr))

            first = int(env.first_row)
            rows0 = np.arange(first, first + n0, dtype=np.int64)
            zeros0 = np.zeros(n0, dtype=np.float64)
            cgrid = np.empty(n0, dtype=np.int32)

            for mode in MODES:
                out = _solve_one(env_ptr, env, n0=n0, nnz_a0=nnz_a0, rows0=rows0, zeros0=zeros0, cgrid=cgrid, params=mode.params)
                row: dict[str, Any] = {
                    "t": int(t),
                    "mode": mode.name,
                    "nx": int(NX),
                    "ny": int(NY),
                    "nz": int(NZ),
                    "stencil": int(STENCIL),
                    "solver_tol": float(SOLVER_TOL),
                    "solver_max_iter": int(SOLVER_MAX_ITER),
                    "n0": int(n0),
                    "nnz_a0": int(nnz_a0),
                    **{f"param_{k}": v for k, v in mode.params.items()},
                    **{f"meta_{k}": v for k, v in meta.items()},
                    **out,
                }
                logs.append(row)
        finally:
            amg.amg_setup_destroy(env_ptr)

        if t % progress_every == 0 or t == 1 or t == T:
            print(f"[progress] {t}/{T}", flush=True)

    # Save logs + summary
    csv_path = out_dir / f"fixed_params_diag_T{T}_{NX}x{NY}x{NZ}_st{STENCIL}_seed{SEED}.csv"
    _write_csv(logs, csv_path)

    summary = {
        "T": int(T),
        "seed": int(SEED),
        "nx": int(NX),
        "ny": int(NY),
        "nz": int(NZ),
        "stencil": int(STENCIL),
        "solver_tol": float(SOLVER_TOL),
        "solver_max_iter": int(SOLVER_MAX_ITER),
        "modes": [m.name for m in MODES],
        "note": "operator_complexity is not available via public HYPRE getters in this build; ap_complexity is cum_nnz(A,P)/nnz(A0); work_units = iterations * ap_complexity.",
        "per_mode": {m.name: _summarize_mode(logs, mode_name=m.name) for m in MODES},
        "csv": str(csv_path),
    }
    json_path = out_dir / f"fixed_params_diag_summary_T{T}_{NX}x{NY}x{NZ}_st{STENCIL}_seed{SEED}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Plot diagnostics
    plt = _import_matplotlib_pyplot()
    plot_path = None
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
        axes = np.asarray(axes).reshape(2, 2)
        colors = ["#0072B2", "#D55E00"]

        for i, mode in enumerate(MODES):
            mode_rows = [r for r in logs if r.get("mode") == mode.name and not bool(r.get("failed", False))]
            iters = np.array([r.get("iterations", np.nan) for r in mode_rows], dtype=float)
            glc = np.array([r.get("grid_complexity", np.nan) for r in mode_rows], dtype=float)
            apc = np.array([r.get("ap_complexity", np.nan) for r in mode_rows], dtype=float)
            wu = np.array([r.get("work_units", np.nan) for r in mode_rows], dtype=float)

            axes[0, 0].hist(iters[np.isfinite(iters)], bins=50, alpha=0.55, label=mode.name, color=colors[i])
            axes[0, 1].hist(wu[np.isfinite(wu)], bins=50, alpha=0.55, label=mode.name, color=colors[i])
            axes[1, 0].hist(glc[np.isfinite(glc)], bins=50, alpha=0.55, label=mode.name, color=colors[i])
            axes[1, 1].hist(apc[np.isfinite(apc)], bins=50, alpha=0.55, label=mode.name, color=colors[i])

        axes[0, 0].set_title("Iteration distribution")
        axes[0, 0].set_xlabel("iterations")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].grid(True, alpha=0.25)

        axes[0, 1].set_title("Work units (iters * ap_complexity)")
        axes[0, 1].set_xlabel("work_units")
        axes[0, 1].grid(True, alpha=0.25)

        axes[1, 0].set_title("Grid complexity (from grid hierarchy)")
        axes[1, 0].set_xlabel("grid_complexity")
        axes[1, 0].grid(True, alpha=0.25)

        axes[1, 1].set_title("A+P complexity (cum_nnz_AP / nnz(A0))")
        axes[1, 1].set_xlabel("ap_complexity")
        axes[1, 1].grid(True, alpha=0.25)

        for ax in axes.ravel():
            ax.legend(fontsize=8)

        fig.suptitle(f"Fixed-parameter BoomerAMG diagnostics  T={T}  {NX}x{NY}x{NZ}  stencil={STENCIL}  seed={SEED}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        plot_path = out_dir / f"fixed_params_diag_plots_T{T}_{NX}x{NY}x{NZ}_st{STENCIL}_seed{SEED}.png"
        fig.savefig(plot_path, dpi=256)
        plt.close(fig)

    print("CSV:", csv_path)
    print("SUMMARY:", json_path)
    if plot_path is not None:
        print("PLOT:", plot_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
