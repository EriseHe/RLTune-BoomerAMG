"""
Utility helpers that mirror the current matrix/RHS generation logic.

These functions are small, explicit mirrors of BoomerAMGRelaxEnv.reset, so
others can reproduce your exact A/b sampling and eval-case construction.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from pathlib import Path
import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # parent of utilities/
_DEFAULT_LIB_PATH = ROOT / "SolvePhase" / "hypre" / "src" / "test" / "libamg_env.dylib"

_AMG_ENV_LIB_CACHE: Dict[str, ctypes.CDLL] = {}

@dataclass(frozen=True)
class CaseSpec:
    nx: int
    ny: int
    nz: int
    stencil: int
    rhs_type: int
    rhs_seed: int
    a0: float
    a1: float
    a2: float
    a3: float


def generate_matrix_coeffs(
    rng: np.random.Generator,
    stencil: int,
    randomize_A: bool,
    a0_base: float,
    a1_base: float,
    a2_base: float,
    a3_base: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Mirrors BoomerAMGRelaxEnv.reset for coefficient generation.

    Returns (a0, a1, a2, a3, k, c) where k/c are only used for 7-pt cases.
    """
    if randomize_A:
        if stencil == 27:
            s1 = float(rng.uniform(0.5, 2.0))
            s2 = float(rng.uniform(0.5, 2.0))
            s3 = float(rng.uniform(0.5, 2.0))
            a1 = a1_base * s1
            a2 = a2_base * s2
            a3 = a3_base * s3
            c_val = float(rng.uniform(0.0, 5.0))
            a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3) + c_val
            return a0, a1, a2, a3, 1.0, 0.0
        s = float(rng.uniform(0.5, 2.0))
        a1 = a1_base * s
        a0 = -(6.0 * a1)
        return a0, a1, a2_base, a3_base, 1.0, 0.0

    if stencil == 27:
        s1, s2, s3 = 1.4, 2.0, 2.5
        a1 = a1_base * s1
        a2 = a2_base * s2
        a3 = a3_base * s3
        a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3)
        return a0, a1, a2, a3, 1.0, 0.0

    a1 = a1_base
    a0 = -(6.0 * a1)
    return a0, a1, a2_base, a3_base, 1.0, 0.0


def generate_rhs_seed(
    rng: np.random.Generator, randomize_b: bool, fixed_rhs_seed: int
) -> int:
    """Mirrors rhs_seed selection in BoomerAMGRelaxEnv.reset."""
    if randomize_b:
        return int(rng.integers(0, 2**63 - 1))
    return int(fixed_rhs_seed)


def sample_grid_choice(
    rng: np.random.Generator, grid_choices: Sequence[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Mirrors grid choice in BoomerAMGRelaxEnv.reset."""
    idx = int(rng.integers(0, len(grid_choices)))
    return tuple(int(x) for x in grid_choices[idx])


def make_eval_cases(
    seed_start: int,
    seed_count: int,
    grid_sizes: Sequence[Tuple[int, int, int]],
    randomize_A: bool,
    randomize_b: bool,
    fixed_rhs_seed: int,
    fixed_rhs_type: int,
    fixed_a_mode: bool = False,
    grid_mode: str = "zip",
) -> List[dict]:
    """
    Mirrors eval case construction from eval_and_plot/eval_default_vs_rl.
    """
    eval_seeds = list(range(int(seed_start), int(seed_start) + int(seed_count)))
    grid_sizes = list(grid_sizes)
    if fixed_a_mode:
        randomize_A = False
        randomize_b = False
        fixed_rhs_type = 1
        eval_seeds = [int(seed_start)]
        if not grid_sizes:
            grid_sizes = [(60, 60, 60)]

    if grid_mode == "full":
        cases = [
            {
                "seed": s,
                "grid": g,
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for s in eval_seeds
            for g in grid_sizes
        ]
    else:
        cases = [
            {
                "seed": s,
                "grid": grid_sizes[i % len(grid_sizes)],
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for i, s in enumerate(eval_seeds)
        ]
    return cases


def build_env_case(
    lib_path: str = "./libamg_env.dylib",
    fixed_grid: Tuple[int, int, int] = (60, 60, 60),
    fixed_stencil: int = 27,
    randomize_A: bool = True,
    randomize_b: bool = True,
    fixed_rhs_seed: int = 123456789,
    fixed_rhs_type: int = 1,
    seed: int = 0,
):
    """
    Create BoomerAMGRelaxEnv and trigger a reset (which builds A and b).

    Returns (env, info) where info contains nx/ny/nz, a0..a3, rhs_seed, r0, etc.
    """
    import importlib.util, sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    mod_path = ROOT / "SolvePhase" / "hypre" / "src" / "test" / "amg_gym_env.py"

    spec = importlib.util.spec_from_file_location("amg_gym_env", mod_path)
    amg_gym_env = importlib.util.module_from_spec(spec)
    sys.modules["amg_gym_env"] = amg_gym_env
    spec.loader.exec_module(amg_gym_env)

    BoomerAMGRelaxEnv = amg_gym_env.BoomerAMGRelaxEnv

    env = BoomerAMGRelaxEnv(
        lib_path=lib_path,
        fixed_grid=fixed_grid,
        fixed_stencil=fixed_stencil,
        randomize_A=randomize_A,
        randomize_b=randomize_b,
        fixed_rhs_seed=fixed_rhs_seed,
        fixed_rhs_type=fixed_rhs_type,
        seed=seed,
    )
    obs, info = env.reset()
    return env, info


def _resolve_lib_path(lib_path: str) -> Path:
    p = Path(lib_path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _load_amg_env_lib(lib_path: str) -> ctypes.CDLL:
    lib_path = str(_resolve_lib_path(lib_path))
    lib = _AMG_ENV_LIB_CACHE.get(lib_path)
    if lib is None:
        lib = ctypes.CDLL(lib_path)

        AMGEnv_p = ctypes.c_void_p
        lib.amg_env_create.restype = AMGEnv_p
        lib.amg_env_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,   # nx ny nz
            ctypes.c_int, ctypes.c_int,                 # stencil rhs_type
            ctypes.c_double, ctypes.c_int,              # tol max_cycles
            ctypes.c_ulonglong,                         # rhs_seed
            ctypes.c_double, ctypes.c_double,           # k c (7pt)
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double  # a0..a3 (27pt)
        ]

        lib.amg_env_get_r0.restype = ctypes.c_double
        lib.amg_env_get_r0.argtypes = [AMGEnv_p]

        lib.amg_env_get_r.restype = ctypes.c_double
        lib.amg_env_get_r.argtypes = [AMGEnv_p]

        lib.amg_env_destroy.restype = None
        lib.amg_env_destroy.argtypes = [AMGEnv_p]

        _AMG_ENV_LIB_CACHE[lib_path] = lib
    return lib


def build_laplacian_env_ij(
    lib_path: str,
    nx: int,
    ny: int,
    nz: int,
    stencil: int = 27,
    rhs_type: int = 1,
    rhs_seed: int = 123456789,
    tol: float = 1e-8,
    max_cycles: int = 30,
    k: float = 1.0,
    c: float = 0.0,
    a0: float = 1.0,
    a1: float = 1.0,
    a2: float = 1.0,
    a3: float = 0.0,
):
    """
    Build a Laplacian A via the IJ builders inside libamg_env.

    Returns (lib, env_ptr, info). Caller must destroy env_ptr.
    """
    lib = _load_amg_env_lib(lib_path)
    env = lib.amg_env_create(
        int(nx), int(ny), int(nz),
        int(stencil), int(rhs_type),
        float(tol), int(max_cycles),
        int(rhs_seed),
        float(k), float(c),
        float(a0), float(a1), float(a2), float(a3),
    )
    info = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "stencil": int(stencil),
        "rhs_type": int(rhs_type),
        "rhs_seed": int(rhs_seed),
        "r0": float(lib.amg_env_get_r0(env)),
    }
    return lib, env, info


def build_difconv_env_ij(
    lib_path: str,
    nx: int,
    ny: int,
    nz: int,
    rhs_type: int = 1,
    rhs_seed: int = 123456789,
    tol: float = 1e-8,
    max_cycles: int = 30,
    cx: float = 1.0,
    cy: float = 100.0,
    cz: float = 100.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 0.0,
):
    """
    Build a DifConv A via the IJ builders inside libamg_env (stencil=0).

    Matches amg_gym_env reset mapping:
      k,c,a0..a3  <=  cx,cy,cz,ax,ay,az
    Returns (lib, env_ptr, info). Caller must destroy env_ptr.
    """
    lib = _load_amg_env_lib(lib_path)
    env = lib.amg_env_create(
        int(nx), int(ny), int(nz),
        0, int(rhs_type),                 # stencil=0 for difconv
        float(tol), int(max_cycles),
        int(rhs_seed),
        float(cx), float(cy),
        float(cz), float(ax), float(ay), float(az),
    )
    info = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "stencil": 0,
        "rhs_type": int(rhs_type),
        "rhs_seed": int(rhs_seed),
        "cx": float(cx),
        "cy": float(cy),
        "cz": float(cz),
        "ax": float(ax),
        "ay": float(ay),
        "az": float(az),
        "r0": float(lib.amg_env_get_r0(env)),
    }
    return lib, env, info


def sample_random_difconv_case(
    rng: np.random.Generator,
    grid_min: int = 10,
    grid_max: int = 80,
    c_min: float = 1.0,
    c_max: float = 1000.0,
):
    """
    Sample (n, cx, cy, cz) the same way as BoomerAMGRelaxEnv:
      n ~ UniformInt[grid_min, grid_max]
      c* ~ Uniform[c_min, c_max]
    """
    n = int(rng.integers(int(grid_min), int(grid_max) + 1))
    cx = float(rng.uniform(float(c_min), float(c_max)))
    cy = float(rng.uniform(float(c_min), float(c_max)))
    cz = float(rng.uniform(float(c_min), float(c_max)))
    return n, cx, cy, cz


def _main():
    # lib, env, info = build_laplacian_env_ij(
    #     lib_path=str(_DEFAULT_LIB_PATH),
    #     nx=20,
    #     ny=20,
    #     nz=20,
    #     rhs_type=0,
    #     rhs_seed=123456789,
    #     tol=1e-8,
    #     max_cycles=30,
    #     k=1.0,
    #     c=0.0,
    # )
    # print("IJ Laplacian env info:", info)
    # lib.amg_env_destroy(env)

    rng = np.random.default_rng(0)
    n, cx, cy, cz = sample_random_difconv_case(
        rng, grid_min=10, grid_max=80, c_min=1.0, c_max=1000.0
    )
    lib2, env2, info2 = build_difconv_env_ij(
        lib_path=str(_DEFAULT_LIB_PATH),
        nx=n, ny=n, nz=n,
        rhs_type=1,
        rhs_seed=123456789,
        tol=1e-8,
        max_cycles=30,
        cx=cx, cy=cy, cz=cz,
        ax=0.0, ay=0.0, az=0.0,
    )
    print("IJ DifConv env info:", info2)
    lib2.amg_env_destroy(env2)


if __name__ == "__main__":
    _main()
