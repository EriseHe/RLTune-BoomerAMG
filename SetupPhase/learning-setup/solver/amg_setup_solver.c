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

#ifdef HYPRE_HAVE_MPI
#include <mpi.h>
#endif

#if defined(_WIN32)
#define AMG_API __declspec(dllexport)
#else
#define AMG_API
#endif

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

static inline HYPRE_Int sign_double(HYPRE_Real a)
{
    return ((0.0 < a) - (0.0 > a));
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

    P = 1;
    Q = num_procs;
    R = 1;

    p = myid % P;
    q = ((myid - p) / P) % Q;
    r = (myid - p - P * q) / (P * Q);

    hinx = 1. / (HYPRE_Real)(nx + 1);
    hiny = 1. / (HYPRE_Real)(ny + 1);
    hinz = 1. / (HYPRE_Real)(nz + 1);

    values = hypre_CTAlloc(HYPRE_Real, 7, HYPRE_MEMORY_HOST);
    values[0] = 0.0;

    if (0 == atype)
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
    else if (1 == atype)
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
    else if (3 == atype)
    {
        sign_prod = sign_double(cx) * sign_double(ax);
        if (sign_prod == 1)
        {
            values[1] = -cx / (hinx * hinx) - ax / hinx;
            values[4] = -cx / (hinx * hinx);
            if (nx > 1) { values[0] += 2.0 * cx / (hinx * hinx) + 1. * ax / hinx; }
        }
        else
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
    else
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

    {
        HYPRE_ParCSRMatrix A = (HYPRE_ParCSRMatrix) GenerateDifConv(
            comm, nx, ny, nz, P, Q, R, p, q, r, values);
        hypre_TFree(values, HYPRE_MEMORY_HOST);
        return A;
    }
}

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

static double wall_time_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0)
    {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart / (double) freq.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + 1e-6 * (double) tv.tv_usec;
#endif
}

/* ------------------------------------------------------------------ */
/*  Create — same matrix/vector setup as amg_env.c                    */
/* ------------------------------------------------------------------ */
AMG_API AMGSetupEnv* amg_setup_create(
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
    void *obj = NULL;
    if (stencil_type == 0)
    {
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

    /* nnz for complexity */
    hypre_ParCSRMatrix *pA = (hypre_ParCSRMatrix*) env->A;
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
AMG_API int amg_setup_solve(
    AMGSetupEnv *env,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double agg_tr, int agg_Pmx,
    double relax_wt, int relax_order, int max_coarse_size,
    double tol, int max_iter,
    int *out_iters, double *out_complexity, double *out_residual, double *out_runtime)
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
    if (agg_tr >= 0.0)           HYPRE_BoomerAMGSetAggTruncFactor(solver, agg_tr);
    if (agg_Pmx >= 0)            HYPRE_BoomerAMGSetAggPMaxElmts(solver, agg_Pmx);
    if (relax_wt >= 0.0)         HYPRE_BoomerAMGSetRelaxWt(solver, relax_wt);
    if (relax_order >= 0)        HYPRE_BoomerAMGSetRelaxOrder(solver, relax_order);
    if (max_coarse_size >= 0)    HYPRE_BoomerAMGSetMaxCoarseSize(solver, max_coarse_size);

    /* hypre-only timing: Setup + Solve (exclude matrix build and x reset) */
    double t0 = wall_time_sec();
    HYPRE_BoomerAMGSetup(solver, env->A, env->b, env->x);
    HYPRE_BoomerAMGSolve(solver, env->A, env->b, env->x);
    double t1 = wall_time_sec();
    if (out_runtime) { *out_runtime = (double) (t1 - t0); }

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
AMG_API void amg_setup_destroy(AMGSetupEnv *env)
{
    if (!env) return;
    if (env->ij_x) HYPRE_IJVectorDestroy(env->ij_x);
    if (env->ij_b) HYPRE_IJVectorDestroy(env->ij_b);
    if (env->ij_A) HYPRE_IJMatrixDestroy(env->ij_A);
    else if (env->A) HYPRE_ParCSRMatrixDestroy(env->A);
    free(env);
}

AMG_API int amg_setup_get_n(AMGSetupEnv *e)   { return e ? e->local_num_rows : 0; }
AMG_API int amg_setup_get_nnz(AMGSetupEnv *e) { return e ? e->nnz : 0; }
