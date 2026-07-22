try:
    from . import _project_paths  # noqa: F401
except ImportError:
    import _project_paths  # type: ignore[no-redef]  # noqa: F401
from hypre.bindings import create_env


with create_env(
    nx=20,
    ny=20,
    nz=20,
    stencil=0,
    rhs_type=0,
    rhs_seed=1234567,
    k=1.0,
    c=100.0,
    a0=100.0,
    a1=0.0,
    a2=0.0,
    a3=0.0,
) as env:
    prepared = env.prepare_rl(params={})
    print("r0 =", prepared.initial_residual_norm)
    for index in range(10):
        residual, runtime = env.step_rl(
            relax_weight=1.0,
            sweeps_down=2,
            sweeps_up=2,
            tol=1.0e-8,
            max_cycles=30,
        )
        print(
            index + 1,
            "r=",
            residual,
            "dt=",
            runtime,
            "status=",
            env.last_step.status,
        )
        if env.last_step.status != 0:
            break
