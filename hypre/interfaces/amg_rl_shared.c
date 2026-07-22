#include <math.h>
#include <stdlib.h>

#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"

#include "amg_rl_shared.h"

double amg_rl_wall_time_sec(void)
{
    return (double) hypre_MPI_Wtime();
}

HYPRE_Real amg_rl_parvec_norm2(HYPRE_ParVector v)
{
    HYPRE_Real dot = 0.0;
    HYPRE_ParVectorInnerProd(v, v, &dot);
    return sqrt(dot);
}

HYPRE_Real amg_rl_compute_residual_norm(HYPRE_ParCSRMatrix A,
                                        HYPRE_ParVector b,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector r)
{
    HYPRE_ParVectorCopy(b, r);
    HYPRE_ParCSRMatrixMatvec(-1.0, A, x, 1.0, r);
    return amg_rl_parvec_norm2(r);
}

int amg_rl_default_relax_type(void)
{
    const char *env = getenv("AMG_RELAX_TYPE");
    if (!env || !env[0]) return 18;

    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || v <= 0 || v > 50) return 18;
    return (int) v;
}

int amg_rl_default_cycle_type(void)
{
    const char *env = getenv("AMG_CYCLE_TYPE");
    if (!env || !env[0]) return 1;

    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || v <= 0 || v > 3) return 1;
    return (int) v;
}

static void amg_rl_configure_single_cycle_solver(HYPRE_Solver solver, int relax_type, int cycle_type)
{
    HYPRE_BoomerAMGSetRelaxType(solver, relax_type);
    HYPRE_BoomerAMGSetCycleType(solver, cycle_type);
    HYPRE_BoomerAMGSetTol(solver, 0.0);
    HYPRE_BoomerAMGSetMaxIter(solver, 1);
    HYPRE_BoomerAMGSetNumSweeps(solver, 1);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 1);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 2);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 3);
}

int amg_rl_prepare_solver(HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x,
                          HYPRE_ParVector r,
                          int *relax_type_io,
                          int *cycle_type_io,
                          double *out_setup_time,
                          double *out_r0)
{
    HYPRE_Int ierr;
    int relax_type = (relax_type_io && *relax_type_io >= 0) ? *relax_type_io : amg_rl_default_relax_type();
    int cycle_type = (cycle_type_io && *cycle_type_io >= 0) ? *cycle_type_io : amg_rl_default_cycle_type();

    amg_rl_configure_single_cycle_solver(solver, relax_type, cycle_type);

    double t0 = amg_rl_wall_time_sec();
    ierr = HYPRE_BoomerAMGSetup(solver, A, b, x);
    double t1 = amg_rl_wall_time_sec();

    if (out_setup_time) *out_setup_time = (double) (t1 - t0);
    if (ierr != 0 || HYPRE_GetError() != 0) return -1;

    if (out_r0) *out_r0 = (double) amg_rl_compute_residual_norm(A, b, x, r);
    if (HYPRE_GetError() != 0 || (out_r0 && !isfinite(*out_r0))) return -1;

    if (relax_type_io) *relax_type_io = relax_type;
    if (cycle_type_io) *cycle_type_io = cycle_type;
    return 0;
}

int amg_rl_step_solver(HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,
                       HYPRE_ParVector x,
                       HYPRE_ParVector r,
                       double tol,
                       int max_cycles,
                       int *cycles_done_io,
                       double relax_weight,
                       int sweeps_down,
                       int sweeps_up,
                       int coarse_sweeps,
                       int cycle_type,
                       int relax_type,
                       int pre_relax_type,
                       int post_relax_type,
                       int coarse_relax_type,
                       int relax_order,
                       double outer_weight,
                       double add_relax_weight,
                       double level_relax_weight,
                       int level_relax_level,
                       double level_outer_weight,
                       int level_outer_level,
                       double *out_residual,
                       double *out_runtime,
                       int *out_status)
{
    HYPRE_Int ierr;
    if (!solver || !A || !b || !x || !r) return -1;

    if (sweeps_down < 1) sweeps_down = 1;
    if (sweeps_down > 10) sweeps_down = 10;
    if (sweeps_up < 1) sweeps_up = 1;
    if (sweeps_up > 10) sweeps_up = 10;
    if (coarse_sweeps < 1) coarse_sweeps = 1;
    if (coarse_sweeps > 10) coarse_sweeps = 10;

    if (cycle_type >= 0)
    {
        HYPRE_BoomerAMGSetCycleType(solver, cycle_type);
    }
    if (relax_type >= 0)
    {
        HYPRE_BoomerAMGSetRelaxType(solver, relax_type);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 1);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 2);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 3);
    }
    if (pre_relax_type >= 0)
    {
        HYPRE_BoomerAMGSetCycleRelaxType(solver, pre_relax_type, 1);
    }
    if (post_relax_type >= 0)
    {
        HYPRE_BoomerAMGSetCycleRelaxType(solver, post_relax_type, 2);
    }
    if (coarse_relax_type >= 0)
    {
        HYPRE_BoomerAMGSetCycleRelaxType(solver, coarse_relax_type, 3);
    }
    if (relax_order >= 0)
    {
        HYPRE_BoomerAMGSetRelaxOrder(solver, relax_order);
    }
    if (outer_weight >= 0.0)
    {
        HYPRE_BoomerAMGSetOuterWt(solver, outer_weight);
    }
    if (add_relax_weight >= 0.0)
    {
        HYPRE_BoomerAMGSetAddRelaxWt(solver, add_relax_weight);
    }
    if (level_relax_weight >= 0.0 && level_relax_level >= 0)
    {
        HYPRE_BoomerAMGSetLevelRelaxWt(solver, level_relax_weight, level_relax_level);
    }
    if (level_outer_weight >= 0.0 && level_outer_level >= 0)
    {
        HYPRE_BoomerAMGSetLevelOuterWt(solver, level_outer_weight, level_outer_level);
    }
    HYPRE_BoomerAMGSetRelaxWt(solver, relax_weight);
    HYPRE_BoomerAMGSetCycleNumSweeps(solver, sweeps_down, 1);
    HYPRE_BoomerAMGSetCycleNumSweeps(solver, sweeps_up, 2);
    HYPRE_BoomerAMGSetCycleNumSweeps(solver, coarse_sweeps, 3);

    double t0 = amg_rl_wall_time_sec();
    ierr = HYPRE_BoomerAMGSolve(solver, A, b, x);
    double t1 = amg_rl_wall_time_sec();

    if (out_runtime) *out_runtime = (double) (t1 - t0);
    if (ierr != 0 || HYPRE_GetError() != 0) return -1;

    double residual = (double) amg_rl_compute_residual_norm(A, b, x, r);
    if (HYPRE_GetError() != 0 || !isfinite(residual)) return -1;
    int cycles_done = cycles_done_io ? (*cycles_done_io + 1) : 0;
    int status = 0;
    if (tol > 0.0 && residual <= tol) status = 1;
    else if (max_cycles > 0 && cycles_done_io && cycles_done >= max_cycles) status = 2;

    if (cycles_done_io) *cycles_done_io = cycles_done;
    if (out_residual) *out_residual = residual;
    if (out_status) *out_status = status;
    return 0;
}
