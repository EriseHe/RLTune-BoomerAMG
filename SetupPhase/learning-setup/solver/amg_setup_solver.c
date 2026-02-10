/*
 * amg_setup_solver.c
 *
 * Setup-phase bandit solver for BoomerAMG.
 * Reuses the same matrix builders (BuildIJLaplacian7pt/27pt from
 * SolvePhase/hypre/src/test/amg_cycle.c) so that setup-phase experiments
 * run on the exact same problem instances as the solve-phase RL environment.
 *
 * Does a FULL BoomerAMG solve (Setup + Solve) with configurable
 * setup-phase parameters, returning iterations + hierarchy complexity.
 *
 * API:
 *   amg_setup_create  — build matrix + vectors (same builders as solve phase)
 *   amg_setup_solve   — full solve with configurable setup params
 *   amg_setup_destroy — cleanup
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_seq_mv.h"

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"

#include <mpi.h>

/*
 * Quiet shim for the HYPRE test-driver builders (amg_cycle.c).
 *
 * We compile SolvePhase/hypre/src/test/amg_cycle.c with:
 *   -Dhypre_printf=amg_setup_quiet_printf
 * so all its hypre_printf calls are redirected here.
 */
HYPRE_Int amg_setup_quiet_printf(const char *format, ...)
{
    (void)format;
    return 0;
}

/* Same builders as amg_env.c uses — defined in amg_cycle.c */
extern HYPRE_Int BuildIJLaplacian7pt (HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

extern HYPRE_Int BuildIJLaplacian27pt(HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

/* ------------------------------------------------------------------ */
typedef struct
{
    MPI_Comm            comm;
    int                 local_num_rows;
    int                 nnz;

    HYPRE_IJMatrix      ij_A;
    HYPRE_ParCSRMatrix  A;

    HYPRE_IJVector      ij_b;
    HYPRE_IJVector      ij_x;
    HYPRE_ParVector     b;
    HYPRE_ParVector     x;

    HYPRE_BigInt        first_row;
    HYPRE_BigInt        last_row;
} AMGSetupEnv;

static int _initialized = 0;

static void _ensure_init(void)
{
    if (_initialized) return;
    int mpi_ok = 0;
    MPI_Initialized(&mpi_ok);
    if (!mpi_ok) { int ac = 0; char **av = NULL; hypre_MPI_Init(&ac, &av); }
    HYPRE_Init();
    _initialized = 1;
}

/* ------------------------------------------------------------------ */
/*  Create — same matrix/vector setup as amg_env.c                    */
/* ------------------------------------------------------------------ */
AMGSetupEnv* amg_setup_create(
    int nx, int ny, int nz,
    int stencil_type,
    int rhs_type,
    unsigned long long rhs_seed,
    double k, double c,
    double a0, double a1, double a2, double a3)
{
    _ensure_init();

    AMGSetupEnv *env = (AMGSetupEnv*) calloc(1, sizeof(AMGSetupEnv));
    env->comm = hypre_MPI_COMM_WORLD;

    /* Build matrix A — identical to amg_env.c */
    HYPRE_BigInt system_size = 0;
    HYPRE_IJMatrix ij_A = NULL;
    char *argv[32]; int argc = 0;
    char nx_s[32], ny_s[32], nz_s[32];
    sprintf(nx_s, "%d", nx); sprintf(ny_s, "%d", ny); sprintf(nz_s, "%d", nz);
    argv[argc++] = (char*) "amg_setup";
    argv[argc++] = (char*) "-n"; argv[argc++] = nx_s; argv[argc++] = ny_s; argv[argc++] = nz_s;

    if (stencil_type == 7)
    {
        char ks[64], cs[64];
        sprintf(ks, "%.17g", k); sprintf(cs, "%.17g", c);
        argv[argc++] = (char*) "-k"; argv[argc++] = ks;
        argv[argc++] = (char*) "-c"; argv[argc++] = cs;
        BuildIJLaplacian7pt(argc, argv, &system_size, &ij_A, HYPRE_MEMORY_HOST);
    }
    else
    {
        char s0[64], s1[64], s2[64], s3[64];
        sprintf(s0, "%.17g", a0); sprintf(s1, "%.17g", a1);
        sprintf(s2, "%.17g", a2); sprintf(s3, "%.17g", a3);
        argv[argc++] = (char*) "-coef27";
        argv[argc++] = s0; argv[argc++] = s1; argv[argc++] = s2; argv[argc++] = s3;
        BuildIJLaplacian27pt(argc, argv, &system_size, &ij_A, HYPRE_MEMORY_HOST);
    }

    env->ij_A = ij_A;
    void *obj = NULL;
    HYPRE_IJMatrixGetObject(ij_A, &obj);
    env->A = (HYPRE_ParCSRMatrix) obj;

    /* nnz for complexity */
    hypre_ParCSRMatrix *pA = (hypre_ParCSRMatrix*) obj;
    env->nnz = (int)(hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(pA))
                    + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(pA)));

    /* Vectors — identical to amg_env.c */
    HYPRE_BigInt fr, lr, fc, lc;
    HYPRE_ParCSRMatrixGetLocalRange(env->A, &fr, &lr, &fc, &lc);
    int n = (int)(lr - fr + 1);
    env->first_row = fr; env->last_row = lr; env->local_num_rows = n;

    HYPRE_BigInt *rows = (HYPRE_BigInt*) malloc(n * sizeof(HYPRE_BigInt));
    double       *vals = (double*)       malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) rows[i] = fr + i;

    /* b */
    HYPRE_IJVectorCreate(env->comm, fr, lr, &env->ij_b);
    HYPRE_IJVectorSetObjectType(env->ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_b);
    if (rhs_type == 0)
    {
        for (int i = 0; i < n; i++) vals[i] = 1.0;
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            unsigned long long s = rhs_seed ^ (unsigned long long)(fr + i);
            s = 6364136223846793005ULL * s + 1ULL;
            double u1 = (double)(s & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
            s = 6364136223846793005ULL * s + 1ULL;
            double u2 = (double)(s & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
            if (u1 < 1e-12) u1 = 1e-12;
            vals[i] = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
        }
    }
    HYPRE_IJVectorSetValues(env->ij_b, n, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_b);
    HYPRE_IJVectorGetObject(env->ij_b, &obj);
    env->b = (HYPRE_ParVector) obj;

    /* x = 0 */
    HYPRE_IJVectorCreate(env->comm, fc, lc, &env->ij_x);
    HYPRE_IJVectorSetObjectType(env->ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_x);
    for (int i = 0; i < n; i++) vals[i] = 0.0;
    HYPRE_IJVectorSetValues(env->ij_x, n, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_x);
    HYPRE_IJVectorGetObject(env->ij_x, &obj);
    env->x = (HYPRE_ParVector) obj;

    free(rows); free(vals);
    return env;
}

/* ------------------------------------------------------------------ */
/*  Solve — full BoomerAMG with configurable setup-phase params       */
/*  Use -1 (int) / -1.0 (double) to skip a param.                    */
/*  Resets x=0 each call.                                             */
/* ------------------------------------------------------------------ */
int amg_setup_solve(
    AMGSetupEnv *env,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double relax_wt, int relax_order, int max_coarse_size,
    double tol, int max_iter,
    int *out_iters, double *out_complexity, double *out_residual)
{
    if (!env) return -1;
    int n = env->local_num_rows;

    /* Reset x = 0 */
    HYPRE_BigInt *rows = (HYPRE_BigInt*) malloc(n * sizeof(HYPRE_BigInt));
    double       *z    = (double*)       calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) rows[i] = env->first_row + i;
    HYPRE_IJVectorSetValues(env->ij_x, n, rows, z);
    HYPRE_IJVectorAssemble(env->ij_x);

    HYPRE_Solver solver;
    HYPRE_BoomerAMGCreate(&solver);
    HYPRE_BoomerAMGSetPrintLevel(solver, 0);
    HYPRE_BoomerAMGSetTol(solver, tol);
    HYPRE_BoomerAMGSetMaxIter(solver, max_iter);
    HYPRE_BoomerAMGSetCumNnzAP(solver, 1.0);

    if (strong_threshold >= 0.0) HYPRE_BoomerAMGSetStrongThreshold(solver, strong_threshold);
    if (coarsen_type >= 0)       HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
    if (interp_type >= 0)        HYPRE_BoomerAMGSetInterpType(solver, interp_type);
    if (max_row_sum >= 0.0)      HYPRE_BoomerAMGSetMaxRowSum(solver, max_row_sum);
    if (relax_type >= 0)         HYPRE_BoomerAMGSetRelaxType(solver, relax_type);
    if (num_sweeps >= 0)         HYPRE_BoomerAMGSetNumSweeps(solver, num_sweeps);
    if (cycle_type >= 0)         HYPRE_BoomerAMGSetCycleType(solver, cycle_type);
    if (max_levels >= 0)         HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
    if (trunc_factor >= 0.0)     HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor);
    if (P_max_elmts >= 0)        HYPRE_BoomerAMGSetPMaxElmts(solver, P_max_elmts);
    if (agg_num_levels >= 0)     HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);
    if (agg_interp_type >= 0)    HYPRE_BoomerAMGSetAggInterpType(solver, agg_interp_type);
    if (relax_wt >= 0.0)         HYPRE_BoomerAMGSetRelaxWt(solver, relax_wt);
    if (relax_order >= 0)        HYPRE_BoomerAMGSetRelaxOrder(solver, relax_order);
    if (max_coarse_size >= 0)    HYPRE_BoomerAMGSetMaxCoarseSize(solver, max_coarse_size);

    HYPRE_BoomerAMGSetup(solver, env->A, env->b, env->x);
    HYPRE_BoomerAMGSolve(solver, env->A, env->b, env->x);

    HYPRE_Int k = 0; HYPRE_Real res = 0.0, cum = 0.0;
    HYPRE_BoomerAMGGetNumIterations(solver, &k);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
    HYPRE_BoomerAMGGetCumNnzAP(solver, &cum);

    *out_iters    = (int) k;
    *out_residual = (double) res;
    *out_complexity = (env->nnz > 0) ? ((double)cum / (double)env->nnz) : 1.0;

    HYPRE_BoomerAMGDestroy(solver);
    free(rows); free(z);
    return 0;
}

/* ------------------------------------------------------------------ */
void amg_setup_destroy(AMGSetupEnv *env)
{
    if (!env) return;
    if (env->ij_x) HYPRE_IJVectorDestroy(env->ij_x);
    if (env->ij_b) HYPRE_IJVectorDestroy(env->ij_b);
    if (env->ij_A) HYPRE_IJMatrixDestroy(env->ij_A);
    free(env);
}

int amg_setup_get_n(AMGSetupEnv *e)   { return e ? e->local_num_rows : 0; }
int amg_setup_get_nnz(AMGSetupEnv *e) { return e ? e->nnz : 0; }
