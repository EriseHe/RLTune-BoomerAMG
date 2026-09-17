/* Deterministic timing coverage for the production RL native helper.
 * HYPRE calls advance a synthetic clock: the reported time must include
 * all work done inside prepare/step, including work preceding a failure.
 * No real sleeping or assumptions about machine throughput are involved.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "_hypre_utilities.h"
#include "HYPRE_parcsr_ls.h"
#include "amg_rl_shared.h"

static double now;
static HYPRE_Int error_flag;
static int fail_setup, fail_solve, fail_residual, nonfinite_residual;

hypre_double hypre_MPI_Wtime(void) { return now; }
HYPRE_Int HYPRE_GetError(void) { return error_flag; }
void hypre_error_handler(const char *file, HYPRE_Int line, HYPRE_Int error, const char *msg)
{
    (void) file; (void) line; (void) msg;
    error_flag = error;
}

#define INT_SETTER(name) \
    HYPRE_Int name(HYPRE_Solver s, HYPRE_Int v) \
    { (void) s; (void) v; now += 0.001; return 0; }
#define REAL_SETTER(name) \
    HYPRE_Int name(HYPRE_Solver s, HYPRE_Real v) \
    { (void) s; (void) v; now += 0.001; return 0; }
#define STAGE_SETTER(name) \
    HYPRE_Int name(HYPRE_Solver s, HYPRE_Int v, HYPRE_Int stage) \
    { (void) s; (void) v; (void) stage; now += 0.001; return 0; }
#define LEVEL_SETTER(name) \
    HYPRE_Int name(HYPRE_Solver s, HYPRE_Real v, HYPRE_Int level) \
    { (void) s; (void) v; (void) level; now += 0.001; return 0; }

INT_SETTER(HYPRE_BoomerAMGSetRelaxType)
INT_SETTER(HYPRE_BoomerAMGSetCycleType)
INT_SETTER(HYPRE_BoomerAMGSetMaxIter)
INT_SETTER(HYPRE_BoomerAMGSetNumSweeps)
INT_SETTER(HYPRE_BoomerAMGSetRelaxOrder)
REAL_SETTER(HYPRE_BoomerAMGSetTol)
REAL_SETTER(HYPRE_BoomerAMGSetOuterWt)
REAL_SETTER(HYPRE_BoomerAMGSetAddRelaxWt)
REAL_SETTER(HYPRE_BoomerAMGSetRelaxWt)
STAGE_SETTER(HYPRE_BoomerAMGSetCycleRelaxType)
STAGE_SETTER(HYPRE_BoomerAMGSetCycleNumSweeps)
LEVEL_SETTER(HYPRE_BoomerAMGSetLevelRelaxWt)
LEVEL_SETTER(HYPRE_BoomerAMGSetLevelOuterWt)

HYPRE_Int HYPRE_BoomerAMGSetup(HYPRE_Solver s, HYPRE_ParCSRMatrix a,
                              HYPRE_ParVector b, HYPRE_ParVector x)
{
    (void) s; (void) a; (void) b; (void) x;
    now += 0.010;
    return fail_setup ? (error_flag = 1) : 0;
}
HYPRE_Int HYPRE_BoomerAMGSolve(HYPRE_Solver s, HYPRE_ParCSRMatrix a,
                              HYPRE_ParVector b, HYPRE_ParVector x)
{
    (void) s; (void) a; (void) b; (void) x;
    now += 0.020;
    return fail_solve ? (error_flag = 1) : 0;
}
HYPRE_Int HYPRE_ParVectorCopy(HYPRE_ParVector x, HYPRE_ParVector y)
{
    (void) x; (void) y;
    now += 0.001;
    return 0;
}
HYPRE_Int HYPRE_ParCSRMatrixMatvec(HYPRE_Complex alpha, HYPRE_ParCSRMatrix a,
                                  HYPRE_ParVector x, HYPRE_Complex beta, HYPRE_ParVector y)
{
    (void) alpha; (void) a; (void) x; (void) beta; (void) y;
    now += 0.002;
    return fail_residual ? (error_flag = 1) : 0;
}
HYPRE_Int HYPRE_ParVectorInnerProd(HYPRE_ParVector x, HYPRE_ParVector y, HYPRE_Real *dot)
{
    (void) x; (void) y;
    now += 0.003;
    *dot = nonfinite_residual ? NAN : 1e-14;
    return 0;
}

int main(void)
{
    int failures = 0;
    int dummy;
    HYPRE_Solver solver = (HYPRE_Solver) &dummy;
    HYPRE_ParCSRMatrix a = (HYPRE_ParCSRMatrix) &dummy;
    HYPRE_ParVector b = (HYPRE_ParVector) &dummy;
    unsetenv("AMG_RELAX_TYPE");
    unsetenv("AMG_CYCLE_TYPE");
    unsetenv("AMG_COARSE_RELAX_TYPE");
    for (int prepare = 0; prepare <= 1; ++prepare)
    {
        for (int failure = 0; failure < 4; ++failure)
        {
            now = 0.0;
            error_flag = 0;
            fail_setup = prepare && failure == 1;
            fail_solve = !prepare && failure == 1;
            fail_residual = failure == 2;
            nonfinite_residual = failure == 3;
            double recorded = -1.0, residual = INFINITY;
            int rc, status = 0, cycles = 0, relax_type = -1, cycle_type = 1;
            if (prepare)
            {
                rc = amg_rl_prepare_solver(solver, a, b, b, b, &relax_type,
                                            &cycle_type, &recorded, &residual);
            }
            else
            {
                rc = amg_rl_step_solver(solver, a, b, b, b, 1e-6, 1.0, 50, &cycles,
                    1.0, 1, 1, 1, -1, -1, -1, -1, -1, -1,
                    -1.0, -1.0, -1.0, -1, -1.0, -1,
                    &residual, &recorded, &status);
            }
            int valid = fabs(recorded-now) < 1e-12 && now > 0.0
                        && ((rc == 0) == (failure == 0));
            if (!valid) ++failures;
            printf("%s failure=%d: recorded=%.6f completed_work=%.6f rc=%d %s\n",
                   prepare ? "prepare" : "step", failure, recorded, now, rc,
                   valid ? "PASS" : "FAIL");
        }
    }
    return failures ? 1 : 0;
}
