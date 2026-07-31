/*
 * amg_runtime.c
 *
 * Shared BoomerAMG runtime for setup and solve-phase learning.
 * Reuses the same matrix builders (BuildIJLaplacian7pt/27pt from
 * hypre/interfaces/amg_cycle.c) so that setup-phase experiments
 * run on the exact same problem instances as the solve-phase RL environment.
 *
 * Does a FULL BoomerAMG solve (Setup + Solve) with configurable
 * setup-phase parameters, returning iterations + hierarchy complexity,
 * plus separate setup / solve timings.
 *
 * This is the only public native interface used by active experiments.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_seq_mv.h"

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"

#ifdef HYPRE_HAVE_MPI
#include <mpi.h>
#endif

#include "amg_rl_shared.h"
#include "amg_runtime.h"

/*
 * Quiet shim for the HYPRE test-driver builders (amg_cycle.c).
 *
 * Kept for compatibility with builds that redirect hypre_printf while
 * compiling the shared amg_cycle.c builders.
 */
HYPRE_Int amg_setup_quiet_printf(const char *format, ...)
{
    (void)format;
    return 0;
}

/* Shared HYPRE test-driver builders defined in amg_cycle.c. */
extern HYPRE_Int BuildIJLaplacian7pt (HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

extern HYPRE_Int BuildIJLaplacian27pt(HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

/* ------------------------------------------------------------------ */
/* DifConv matrix builder used by both setup and solve workflows. */
/* ------------------------------------------------------------------ */

static inline HYPRE_Int sign_double(HYPRE_Real a)
{
    return ( (0.0 < a) - (0.0 > a) );
}

static HYPRE_ParCSRMatrix build_difconv_matrix(
    MPI_Comm comm,
    HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
    HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz,
    HYPRE_Real ax, HYPRE_Real ay, HYPRE_Real az,
    HYPRE_Int atype)
{
    HYPRE_Int num_procs, myid;
    HYPRE_Int P, Q, R;
    HYPRE_Int p, q, r;
    HYPRE_Real hinx, hiny, hinz;
    HYPRE_Int sign_prod;
    HYPRE_Real *values;

    hypre_MPI_Comm_size(comm, &num_procs);
    hypre_MPI_Comm_rank(comm, &myid);

    /* Default processor topology (matches ij.c defaults) */
    P = 1;
    Q = num_procs;
    R = 1;

    p = myid % P;
    q = (( myid - p) / P) % Q;
    r = ( myid - p - P * q) / ( P * Q );

    hinx = 1. / (HYPRE_Real)(nx + 1);
    hiny = 1. / (HYPRE_Real)(ny + 1);
    hinz = 1. / (HYPRE_Real)(nz + 1);

    /* values[7]:
     *    [0]: center
     *    [1]: X-
     *    [2]: Y-
     *    [3]: Z-
     *    [4]: X+
     *    [5]: Y+
     *    [6]: Z+
     */
    values = hypre_CTAlloc(HYPRE_Real, 7, HYPRE_MEMORY_HOST);
    values[0] = 0.0;

    if (0 == atype) /* forward scheme for conv */
    {
        values[1] = -cx / (hinx * hinx);
        values[2] = -cy / (hiny * hiny);
        values[3] = -cz / (hinz * hinz);
        values[4] = -cx / (hinx * hinx) + ax / hinx;
        values[5] = -cy / (hiny * hiny) + ay / hiny;
        values[6] = -cz / (hinz * hinz) + az / hinz;

        if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx) - 1. * ax / hinx; }
        if (ny > 1) { values[0] += 2.0 * cy / (hiny * hiny) - 1. * ay / hiny; }
        if (nz > 1) { values[0] += 2.0 * cz / (hinz * hinz) - 1. * az / hinz; }
    }
    else if (1 == atype) /* backward scheme for conv */
    {
        values[1] = -cx / (hinx * hinx) - ax / hinx;
        values[2] = -cy / (hiny * hiny) - ay / hiny;
        values[3] = -cz / (hinz * hinz) - az / hinz;
        values[4] = -cx / (hinx * hinx);
        values[5] = -cy / (hiny * hiny);
        values[6] = -cz / (hinz * hinz);

        if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx) + 1. * ax / hinx; }
        if (ny > 1) { values[0] += 2.0 * cy / (hiny * hiny) + 1. * ay / hiny; }
        if (nz > 1) { values[0] += 2.0 * cz / (hinz * hinz) + 1. * az / hinz; }
    }
    else if (3 == atype) /* upwind scheme */
    {
        sign_prod = sign_double(cx) * sign_double(ax);
        if (sign_prod == 1) /* same sign use back scheme */
        {
            values[1] = -cx / (hinx * hinx) - ax / hinx;
            values[4] = -cx / (hinx * hinx);
            if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx) + 1. * ax / hinx; }
        }
        else /* diff sign use forward scheme */
        {
            values[1] = -cx / (hinx * hinx);
            values[4] = -cx / (hinx * hinx) + ax / hinx;
            if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx) - 1. * ax / hinx; }
        }

        sign_prod = sign_double(cy) * sign_double(ay);
        if (sign_prod == 1)
        {
            values[2] = -cy / (hiny * hiny) - ay / hiny;
            values[5] = -cy / (hiny * hiny);
            if (ny > 1) { values[0] += 2.0 * cy / (hiny * hiny) + 1. * ay / hiny; }
        }
        else
        {
            values[2] = -cy / (hiny * hiny);
            values[5] = -cy / (hiny * hiny) + ay / hiny;
            if (ny > 1) { values[0] += 2.0 * cy / (hiny * hiny) - 1. * ay / hiny; }
        }

        sign_prod = sign_double(cz) * sign_double(az);
        if (sign_prod == 1)
        {
            values[3] = -cz / (hinz * hinz) - az / hinz;
            values[6] = -cz / (hinz * hinz);
            if (nz > 1) { values[0] += 2.0 * cz / (hinz * hinz) + 1. * az / hinz; }
        }
        else
        {
            values[3] = -cz / (hinz * hinz);
            values[6] = -cz / (hinz * hinz) + az / hinz;
            if (nz > 1) { values[0] += 2.0 * cz / (hinz * hinz) - 1. * az / hinz; }
        }
    }
    else /* centered difference scheme */
    {
        values[1] = -cx / (hinx * hinx) - ax / (2. * hinx);
        values[2] = -cy / (hiny * hiny) - ay / (2. * hiny);
        values[3] = -cz / (hinz * hinz) - az / (2. * hinz);
        values[4] = -cx / (hinx * hinx) + ax / (2. * hinx);
        values[5] = -cy / (hiny * hiny) + ay / (2. * hiny);
        values[6] = -cz / (hinz * hinz) + az / (2. * hinz);

        if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx); }
        if (ny > 1) { values[0] += 2.0 * cy / (hiny * hiny); }
        if (nz > 1) { values[0] += 2.0 * cz / (hinz * hinz); }
    }

    HYPRE_ParCSRMatrix A = (HYPRE_ParCSRMatrix) GenerateDifConv(
        comm, nx, ny, nz, P, Q, R, p, q, r, values);

    hypre_TFree(values, HYPRE_MEMORY_HOST);
    return A;
}

/* ------------------------------------------------------------------ */
struct AMGRuntime
{
    MPI_Comm            comm;
    int                 local_num_rows;
    int                 nnz;

    HYPRE_IJMatrix      ij_A;
    HYPRE_ParCSRMatrix  A;

    HYPRE_IJVector      ij_b;
    HYPRE_IJVector      ij_x;
    HYPRE_IJVector      ij_r;
    HYPRE_ParVector     b;
    HYPRE_ParVector     x;
    HYPRE_ParVector     r;

    HYPRE_Solver        solver;
    double              r0;
    double              r_curr;
    double              setup_time;
    int                 cycles_done;
    int                 cycle_type;
    int                 relax_type;

    HYPRE_BigInt        first_row;
    HYPRE_BigInt        last_row;
};

static int _initialized = 0;

static void clear_hypre_errors(void)
{
    HYPRE_ClearAllErrors();
    HYPRE_ClearErrorMessages();
}

static void _ensure_init(void)
{
    if (_initialized) return;
#ifdef HYPRE_HAVE_MPI
    int mpi_ok = 0;
    MPI_Initialized(&mpi_ok);
    if (!mpi_ok)
    {
        int ac = 0;
        char **av = NULL;
        hypre_MPI_Init(&ac, &av);
    }
#endif
    HYPRE_Init();
    _initialized = 1;
}

static void reset_x_to_zero(AMGRuntime *env)
{
    int n = env->local_num_rows;
    HYPRE_BigInt *rows = (HYPRE_BigInt*) malloc((size_t)n * sizeof(HYPRE_BigInt));
    double *z = (double*) calloc((size_t)n, sizeof(double));
    for (int i = 0; i < n; i++) rows[i] = env->first_row + i;
    HYPRE_IJVectorSetValues(env->ij_x, n, rows, z);
    HYPRE_IJVectorAssemble(env->ij_x);
    free(rows);
    free(z);
}

static int apply_setup_params(
    HYPRE_Solver solver,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double agg_tr, int agg_Pmx,
    double relax_wt, int relax_order, int max_coarse_size)
{
    if (strong_threshold >= 0.0) HYPRE_BoomerAMGSetStrongThreshold(solver, strong_threshold);
    if (coarsen_type >= 0)       HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
    if (interp_type >= 0)        HYPRE_BoomerAMGSetInterpType(solver, interp_type);
    if (max_row_sum >= 0.0)      HYPRE_BoomerAMGSetMaxRowSum(solver, max_row_sum);
    if (relax_type >= 0)
    {
        HYPRE_BoomerAMGSetRelaxType(solver, relax_type);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 1);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 2);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, 3);
    }
    if (num_sweeps >= 0)         HYPRE_BoomerAMGSetNumSweeps(solver, num_sweeps);
    if (cycle_type >= 0)         HYPRE_BoomerAMGSetCycleType(solver, cycle_type);
    if (max_levels >= 0)         HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
    if (trunc_factor >= 0.0)     HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor);
    if (P_max_elmts >= 0)        HYPRE_BoomerAMGSetPMaxElmts(solver, P_max_elmts);
    if (agg_num_levels >= 0)     HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);
    if (agg_interp_type >= 0)    HYPRE_BoomerAMGSetAggInterpType(solver, agg_interp_type);
    if (agg_tr >= 0.0)           HYPRE_BoomerAMGSetAggTruncFactor(solver, agg_tr);
    if (agg_Pmx >= 0)            HYPRE_BoomerAMGSetAggPMaxElmts(solver, agg_Pmx);
    if (relax_wt >= 0.0)         HYPRE_BoomerAMGSetRelaxWt(solver, relax_wt);
    if (relax_order >= 0)        HYPRE_BoomerAMGSetRelaxOrder(solver, relax_order);
    if (max_coarse_size >= 0)    HYPRE_BoomerAMGSetMaxCoarseSize(solver, max_coarse_size);
    return HYPRE_GetError() == 0 ? AMG_RUNTIME_OK : AMG_RUNTIME_ERR_SETUP;
}

/* ------------------------------------------------------------------ */
/*  Create the shared matrix/vector state.                             */
/* ------------------------------------------------------------------ */
AMG_API AMGRuntime* amg_runtime_create(
    int nx, int ny, int nz,
    int stencil_type,
    int rhs_type,
    unsigned long long rhs_seed,
    double k, double c,
    double a0, double a1, double a2, double a3)
{
    _ensure_init();
    clear_hypre_errors();

    AMGRuntime *env = (AMGRuntime*) calloc(1, sizeof(AMGRuntime));
    if (!env) return NULL;
    env->comm = hypre_MPI_COMM_WORLD;

    /* Build matrix A. */
    void *obj = NULL;
    if (stencil_type == 0)
    {
        /* DifConv: Map args as cx=k, cy=c, cz=a0, ax=a1, ay=a2, az=a3, atype=0. */
        env->A = build_difconv_matrix(
            env->comm,
            (HYPRE_BigInt) nx, (HYPRE_BigInt) ny, (HYPRE_BigInt) nz,
            (HYPRE_Real) k, (HYPRE_Real) c, (HYPRE_Real) a0,
            (HYPRE_Real) a1, (HYPRE_Real) a2, (HYPRE_Real) a3,
            0);
        env->ij_A = NULL;
    }
    else
    {
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
        HYPRE_IJMatrixGetObject(ij_A, &obj);
        env->A = (HYPRE_ParCSRMatrix) obj;
    }
    if (!env->A || HYPRE_GetError() != 0)
    {
        clear_hypre_errors();
        amg_runtime_destroy(env);
        return NULL;
    }

    /* nnz for complexity */
    hypre_ParCSRMatrix *pA = (hypre_ParCSRMatrix*) env->A;
    env->nnz = (int)(hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(pA))
                    + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(pA)));

    /* Build vectors. */
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

    /* r = 0 workspace */
    HYPRE_IJVectorCreate(env->comm, fr, lr, &env->ij_r);
    HYPRE_IJVectorSetObjectType(env->ij_r, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_r);
    for (int i = 0; i < n; i++) vals[i] = 0.0;
    HYPRE_IJVectorSetValues(env->ij_r, n, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_r);
    HYPRE_IJVectorGetObject(env->ij_r, &obj);
    env->r = (HYPRE_ParVector) obj;

    free(rows); free(vals);
    if (HYPRE_GetError() != 0)
    {
        clear_hypre_errors();
        amg_runtime_destroy(env);
        return NULL;
    }
    return env;
}

/* ------------------------------------------------------------------ */
/*  Solve — full BoomerAMG with configurable setup-phase params        */
/*  Use -1 (int) / -1.0 (double) to skip a param.                     */
/*  Resets x=0 each call.                                             */
/*  Returns separate HYPRE setup / solve timings and total runtime.   */
AMG_API int amg_runtime_solve(
    AMGRuntime *env,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double agg_tr, int agg_Pmx,
    double relax_wt, int relax_order, int max_coarse_size,
    double tol, int max_iter,
    int *out_iters, double *out_complexity, double *out_residual,
    double *out_runtime, double *out_setup_runtime, double *out_solve_runtime,
    int *out_status)
{
    HYPRE_Int ierr;
    HYPRE_Int solve_error;
    HYPRE_Real actual_tol = 0.0;
    HYPRE_Solver solver = NULL;
    double t0, t1, t2;

    if (!env || !out_iters || !out_complexity || !out_residual || !out_status)
    {
        return AMG_RUNTIME_ERR_INVALID_ARGUMENT;
    }
    *out_iters = 0;
    *out_complexity = 0.0;
    *out_residual = INFINITY;
    *out_status = AMG_RUNTIME_STATUS_CONTINUE;
    if (out_runtime) *out_runtime = 0.0;
    if (out_setup_runtime) *out_setup_runtime = 0.0;
    if (out_solve_runtime) *out_solve_runtime = 0.0;

    clear_hypre_errors();
    reset_x_to_zero(env);
    if (HYPRE_GetError() != 0)
    {
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }
    if (relax_type < 0) relax_type = amg_rl_default_relax_type();
    if (cycle_type < 0) cycle_type = amg_rl_default_cycle_type();

    ierr = HYPRE_BoomerAMGCreate(&solver);
    if (ierr != 0 || !solver || HYPRE_GetError() != 0)
    {
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_CREATE;
    }
    HYPRE_BoomerAMGSetPrintLevel(solver, 0);
    if (tol >= 0.0) HYPRE_BoomerAMGSetTol(solver, tol);
    if (max_iter >= 0) HYPRE_BoomerAMGSetMaxIter(solver, max_iter);
    HYPRE_BoomerAMGSetCumNnzAP(solver, 1.0);
    ierr = apply_setup_params(
        solver,
        strong_threshold, coarsen_type, interp_type, max_row_sum,
        relax_type, num_sweeps, cycle_type, max_levels,
        trunc_factor, P_max_elmts,
        agg_num_levels, agg_interp_type, agg_tr, agg_Pmx,
        relax_wt, relax_order, max_coarse_size
    );
    if (ierr != AMG_RUNTIME_OK || HYPRE_GetError() != 0)
    {
        HYPRE_BoomerAMGDestroy(solver);
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }

    /* hypre-only timing: Setup + Solve (exclude matrix build and x reset) */
    t0 = amg_rl_wall_time_sec();
    ierr = HYPRE_BoomerAMGSetup(solver, env->A, env->b, env->x);
    t1 = amg_rl_wall_time_sec();
    if (out_setup_runtime) { *out_setup_runtime = (double) (t1 - t0); }
    if (out_runtime) { *out_runtime = (double) (t1 - t0); }
    if (ierr != 0 || HYPRE_GetError() != 0)
    {
        HYPRE_BoomerAMGDestroy(solver);
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }

    ierr = HYPRE_BoomerAMGSolve(solver, env->A, env->b, env->x);
    t2 = amg_rl_wall_time_sec();
    if (out_solve_runtime) { *out_solve_runtime = (double) (t2 - t1); }
    if (out_runtime) { *out_runtime = (double) (t2 - t0); }
    solve_error = ierr | HYPRE_GetError();
    if (solve_error != 0 && (solve_error & ~HYPRE_ERROR_CONV) != 0)
    {
        HYPRE_BoomerAMGDestroy(solver);
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SOLVE;
    }

    /* Nonconvergence is an explicit outcome, not an execution error. */
    if ((solve_error & HYPRE_ERROR_CONV) != 0)
    {
        clear_hypre_errors();
    }

    HYPRE_Int k = 0; HYPRE_Real res = 0.0, cum = 0.0;
    HYPRE_BoomerAMGGetNumIterations(solver, &k);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
    HYPRE_BoomerAMGGetCumNnzAP(solver, &cum);
    HYPRE_BoomerAMGGetTol(solver, &actual_tol);

    if (HYPRE_GetError() != 0)
    {
        HYPRE_BoomerAMGDestroy(solver);
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SOLVE;
    }

    *out_iters = (int) k;
    *out_residual = (double) res;
    *out_complexity = (env->nnz > 0) ? ((double)cum / (double)env->nnz) : 1.0;
    *out_status = (
        actual_tol > 0.0 &&
        ((solve_error & HYPRE_ERROR_CONV) != 0 || (double) res > (double) actual_tol)
    ) ? AMG_RUNTIME_STATUS_MAX_CYCLES : AMG_RUNTIME_STATUS_CONVERGED;

    HYPRE_BoomerAMGDestroy(solver);
    clear_hypre_errors();
    if (!isfinite(*out_residual) || !isfinite(*out_complexity))
    {
        return AMG_RUNTIME_ERR_NONFINITE;
    }
    return AMG_RUNTIME_OK;
}

AMG_API int amg_runtime_prepare(
    AMGRuntime *env,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double agg_tr, int agg_Pmx,
    double relax_wt, int relax_order, int max_coarse_size,
    double *out_setup_runtime,
    double *out_r0)
{
    HYPRE_Int ierr;
    if (!env) return AMG_RUNTIME_ERR_INVALID_ARGUMENT;
    if (out_setup_runtime) *out_setup_runtime = 0.0;
    if (out_r0) *out_r0 = INFINITY;

    clear_hypre_errors();
    reset_x_to_zero(env);
    if (HYPRE_GetError() != 0)
    {
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }

    if (env->solver)
    {
        HYPRE_BoomerAMGDestroy(env->solver);
        env->solver = NULL;
    }
    clear_hypre_errors();

    ierr = HYPRE_BoomerAMGCreate(&env->solver);
    if (ierr != 0 || !env->solver || HYPRE_GetError() != 0)
    {
        env->solver = NULL;
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_CREATE;
    }
    HYPRE_BoomerAMGSetPrintLevel(env->solver, 0);
    HYPRE_BoomerAMGSetCumNnzAP(env->solver, 1.0);
    ierr = apply_setup_params(
        env->solver,
        strong_threshold, coarsen_type, interp_type, max_row_sum,
        relax_type, num_sweeps, cycle_type, max_levels,
        trunc_factor, P_max_elmts,
        agg_num_levels, agg_interp_type, agg_tr, agg_Pmx,
        relax_wt, relax_order, max_coarse_size
    );
    if (ierr != AMG_RUNTIME_OK || HYPRE_GetError() != 0)
    {
        HYPRE_BoomerAMGDestroy(env->solver);
        env->solver = NULL;
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }

    env->cycles_done = 0;
    env->relax_type = relax_type;
    env->cycle_type = cycle_type;
    if (amg_rl_prepare_solver(
            env->solver,
            env->A,
            env->b,
            env->x,
            env->r,
            &env->relax_type,
            &env->cycle_type,
            &env->setup_time,
            &env->r0) != 0)
    {
        if (out_setup_runtime) *out_setup_runtime = env->setup_time;
        HYPRE_BoomerAMGDestroy(env->solver);
        env->solver = NULL;
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SETUP;
    }

    if (out_setup_runtime) { *out_setup_runtime = env->setup_time; }
    env->r_curr = env->r0;
    if (out_r0) { *out_r0 = env->r0; }
    if (!isfinite(env->r0) || !isfinite(env->setup_time))
    {
        HYPRE_BoomerAMGDestroy(env->solver);
        env->solver = NULL;
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_NONFINITE;
    }
    return AMG_RUNTIME_OK;
}

AMG_API int amg_runtime_step(
    AMGRuntime *env,
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
    double tol,
    int max_cycles,
    double *out_residual,
    double *out_runtime,
    int *out_status)
{
    if (!env || !env->solver) return AMG_RUNTIME_ERR_INVALID_ARGUMENT;
    if (out_residual) *out_residual = INFINITY;
    if (out_runtime) *out_runtime = 0.0;
    if (out_status) *out_status = AMG_RUNTIME_STATUS_CONTINUE;

    clear_hypre_errors();

    if (amg_rl_step_solver(
            env->solver,
            env->A,
            env->b,
            env->x,
            env->r,
            tol,
            env->r0,
            max_cycles,
            &env->cycles_done,
            relax_weight,
            sweeps_down,
            sweeps_up,
            coarse_sweeps,
            cycle_type,
            relax_type,
            pre_relax_type,
            post_relax_type,
            coarse_relax_type,
            relax_order,
            outer_weight,
            add_relax_weight,
            level_relax_weight,
            level_relax_level,
            level_outer_weight,
            level_outer_level,
            &env->r_curr,
            out_runtime,
            out_status) != 0)
    {
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_SOLVE;
    }

    if (cycle_type >= 0)
    {
        env->cycle_type = cycle_type;
    }
    if (relax_type >= 0)
    {
        env->relax_type = relax_type;
    }
    if (out_residual) { *out_residual = env->r_curr; }
    if (!isfinite(env->r_curr) || (out_runtime && !isfinite(*out_runtime)))
    {
        clear_hypre_errors();
        return AMG_RUNTIME_ERR_NONFINITE;
    }
    return AMG_RUNTIME_OK;
}

/* ------------------------------------------------------------------ */
AMG_API void amg_runtime_destroy(AMGRuntime *env)
{
    if (!env) return;
    if (env->solver) HYPRE_BoomerAMGDestroy(env->solver);
    if (env->ij_r) HYPRE_IJVectorDestroy(env->ij_r);
    if (env->ij_x) HYPRE_IJVectorDestroy(env->ij_x);
    if (env->ij_b) HYPRE_IJVectorDestroy(env->ij_b);
    if (env->ij_A) HYPRE_IJMatrixDestroy(env->ij_A);
    else if (env->A) HYPRE_ParCSRMatrixDestroy(env->A);
    free(env);
}

AMG_API int amg_runtime_get_n(AMGRuntime *e)   { return e ? e->local_num_rows : 0; }
AMG_API int amg_runtime_get_nnz(AMGRuntime *e) { return e ? e->nnz : 0; }
AMG_API double amg_runtime_get_r0(AMGRuntime *e) { return e ? e->r0 : 0.0; }
AMG_API double amg_runtime_get_r(AMGRuntime *e) { return e ? e->r_curr : 0.0; }
AMG_API int amg_runtime_get_cycle(AMGRuntime *e) { return e ? e->cycles_done : 0; }
AMG_API double amg_runtime_get_setup_time(AMGRuntime *e) { return e ? e->setup_time : 0.0; }
AMG_API int amg_runtime_get_cycle_type(AMGRuntime *e) { return e ? e->cycle_type : -1; }
AMG_API int amg_runtime_get_relax_type(AMGRuntime *e) { return e ? e->relax_type : -1; }

AMG_API int amg_runtime_get_relax_weight(AMGRuntime *e, int level, double *out_weight)
{
    hypre_ParAMGData *amg_data;
    HYPRE_Real *relax_weight;
    HYPRE_Int max_levels;

    if (!e || !e->solver || !out_weight) return AMG_RUNTIME_ERR_INVALID_ARGUMENT;
    amg_data = (hypre_ParAMGData *) e->solver;
    max_levels = hypre_ParAMGDataMaxLevels(amg_data);
    relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
    if (level < 0 || level >= max_levels || !relax_weight) return AMG_RUNTIME_ERR_INVALID_ARGUMENT;

    /* Public HYPRE getters return the process-wide sticky error flag. */
    *out_weight = (double) relax_weight[level];
    return AMG_RUNTIME_OK;
}
