// amg_env.c
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

/* ---------- small utilities ---------- */

static HYPRE_Real parvec_norm2(HYPRE_ParVector v)
{
    HYPRE_Real dot = 0.0;
    HYPRE_ParVectorInnerProd(v, v, &dot);
    return sqrt(dot);
}

static HYPRE_Real compute_residual_norm(HYPRE_ParCSRMatrix A,
                                        HYPRE_ParVector b,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector r /* workspace */)
{
    /* r <- b - A x */
    HYPRE_ParVectorCopy(b, r);
    HYPRE_ParCSRMatrixMatvec(-1.0, A, x, 1.0, r);
    return parvec_norm2(r);
}

static double wall_time_sec(void)
{
    return (double) hypre_MPI_Wtime();
}

/* Reuse Laplacian builders defined in amg_cycle.c */
extern HYPRE_Int BuildIJLaplacian7pt (HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

extern HYPRE_Int BuildIJLaplacian27pt(HYPRE_Int argc, char *argv[],
                                      HYPRE_BigInt *size,
                                      HYPRE_IJMatrix *A_ptr,
                                      HYPRE_MemoryLocation memory_location);

/* ---------- Environment struct ---------- */

typedef struct
{
    MPI_Comm           comm;

    HYPRE_IJMatrix     ij_A;
    HYPRE_ParCSRMatrix A;

    HYPRE_IJVector     ij_b;
    HYPRE_IJVector     ij_x;
    HYPRE_IJVector     ij_r;
    HYPRE_ParVector    b;
    HYPRE_ParVector    x;
    HYPRE_ParVector    r;

    HYPRE_Solver       amg_solver;

    double             tol;
    double             r0;
    double             r_curr;
    int                max_cycles;
    int                cycles_done;

    int                nx, ny, nz;
    int                stencil_type;  // 7 or 27
    int                rhs_type;      // 0=ones, 1=random
    double b_norm;
    double cos_br;  // cosine between b and current residual r
} AMGEnv;

/* Simple global guard for MPI + hypre init */
static int hypre_initialized = 0;

static void ensure_hypre_init(void)
{
    if (hypre_initialized) return;

    /* Avoid double-init if MPI already initialized (e.g., mpi4py) */
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited)
    {
        int argc = 0;
        char **argv = NULL;
        hypre_MPI_Init(&argc, &argv);
    }

    HYPRE_Init();
    hypre_initialized = 1;
}

/* ---------- Public API ---------- */

AMGEnv* amg_env_create(int nx, int ny, int nz,
    int stencil_type,
    int rhs_type,
    double tol,
    int max_cycles,
    unsigned long long rhs_seed,
    double k, double c,             // for 7pt
    double a0, double a1, double a2, double a3  // for 27pt
)
{
    ensure_hypre_init();

    AMGEnv *env = (AMGEnv*) calloc(1, sizeof(AMGEnv));
    env->comm          = hypre_MPI_COMM_WORLD;
    env->nx            = nx;
    env->ny            = ny;
    env->nz            = nz;
    env->stencil_type  = stencil_type;
    env->rhs_type      = rhs_type;
    env->tol           = tol;
    env->max_cycles    = max_cycles;
    env->cycles_done   = 0;
    env->b_norm = 0.0;
    env->cos_br = 0.0;

    /* ----------- Build matrix A (reuse amg_cycle.c builders) ----------- */
    HYPRE_BigInt system_size = 0;
    HYPRE_IJMatrix ij_A = NULL;

    /* Fake argv for the builder */
    char *argv[32];
    int argc = 0;
    char nx_str[32], ny_str[32], nz_str[32];

    sprintf(nx_str, "%d", nx);
    sprintf(ny_str, "%d", ny);
    sprintf(nz_str, "%d", nz);

    argv[argc++] = (char*) "amg_env";
    argv[argc++] = (char*) "-n";
    argv[argc++] = nx_str;
    argv[argc++] = ny_str;
    argv[argc++] = nz_str;

    if (stencil_type == 7)
    {
    char k_str[64], c_str[64];
    sprintf(k_str, "%.17g", k);
    sprintf(c_str, "%.17g", c);
    argv[argc++] = (char*) "-k"; argv[argc++] = k_str;
    argv[argc++] = (char*) "-c"; argv[argc++] = c_str;

    BuildIJLaplacian7pt(argc, argv, &system_size, &ij_A, HYPRE_MEMORY_HOST);
    }
    else
    {
    char a0s[64], a1s[64], a2s[64], a3s[64];
    sprintf(a0s, "%.17g", a0);
    sprintf(a1s, "%.17g", a1);
    sprintf(a2s, "%.17g", a2);
    sprintf(a3s, "%.17g", a3);
    argv[argc++] = (char*) "-coef27";
    argv[argc++] = a0s; argv[argc++] = a1s; argv[argc++] = a2s; argv[argc++] = a3s;

    BuildIJLaplacian27pt(argc, argv, &system_size, &ij_A, HYPRE_MEMORY_HOST);
    }

    env->ij_A = ij_A;

    void *object = NULL;
    HYPRE_IJMatrixGetObject(env->ij_A, &object);
    env->A = (HYPRE_ParCSRMatrix) object;

    /* ----------- Create vectors b, x, r ----------- */
    HYPRE_BigInt first_row, last_row, first_col, last_col;
    HYPRE_ParCSRMatrixGetLocalRange(env->A,
                                    &first_row, &last_row,
                                    &first_col, &last_col);
    int local_num_rows = (int)(last_row - first_row + 1);

    /* b */
    HYPRE_IJVectorCreate(env->comm, first_row, last_row, &env->ij_b);
    HYPRE_IJVectorSetObjectType(env->ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_b);

    /* x */
    HYPRE_IJVectorCreate(env->comm, first_col, last_col, &env->ij_x);
    HYPRE_IJVectorSetObjectType(env->ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_x);

    /* r */
    HYPRE_IJVectorCreate(env->comm, first_row, last_row, &env->ij_r);
    HYPRE_IJVectorSetObjectType(env->ij_r, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(env->ij_r);

    HYPRE_BigInt *rows = (HYPRE_BigInt*) malloc((size_t)local_num_rows * sizeof(HYPRE_BigInt));
    double *vals       = (double*)        malloc((size_t)local_num_rows * sizeof(double));

    for (int i = 0; i < local_num_rows; ++i)
        rows[i] = first_row + i;

    /* x0 = 0 */
    for (int i = 0; i < local_num_rows; ++i) vals[i] = 0.0;
    HYPRE_IJVectorSetValues(env->ij_x, local_num_rows, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_x);

    /* r initial = 0 */
    HYPRE_IJVectorSetValues(env->ij_r, local_num_rows, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_r);

    /* b = ones or deterministic "random" */
    if (rhs_type == 0)
    {
        for (int i = 0; i < local_num_rows; ++i) vals[i] = 1.0;
    }
    else
    {
        for (int i = 0; i < local_num_rows; ++i)
        {
            unsigned long long seed = rhs_seed ^ (unsigned long long)(first_row + i);
            seed = 6364136223846793005ULL * seed + 1ULL;
            double u = (double)(seed & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
            vals[i] = 2.0 * u - 1.0; /* [-1,1] */
        }
    }

    HYPRE_IJVectorSetValues(env->ij_b, local_num_rows, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_b);

    free(vals);
    free(rows);

    /* Grab ParVector objects */
    HYPRE_IJVectorGetObject(env->ij_b, &object);
    env->b = (HYPRE_ParVector) object;

    HYPRE_IJVectorGetObject(env->ij_x, &object);
    env->x = (HYPRE_ParVector) object;

    HYPRE_IJVectorGetObject(env->ij_r, &object);
    env->r = (HYPRE_ParVector) object;

    env->b_norm = (double) parvec_norm2(env->b);

    /* ----------- AMG setup: 1 V-cycle per Solve call ----------- */
    HYPRE_BoomerAMGCreate(&env->amg_solver);
    HYPRE_BoomerAMGSetTol(env->amg_solver, 0.0);   /* solve exactly one V-cycle per step */
    HYPRE_BoomerAMGSetMaxIter(env->amg_solver, 1); /* ONE V-cycle */
    HYPRE_BoomerAMGSetNumSweeps(env->amg_solver, 1);

    /* IMPORTANT FIX:
       These functions expect ParCSRMatrix/ParVector, so pass env->A/env->b/env->x directly. */
    HYPRE_BoomerAMGSetup(env->amg_solver, env->A, env->b, env->x);

    /* ----------- Compute initial residual r0 = ||b - A x0|| ----------- */
    // initial residual + initial cos(b,r)
    env->r0     = (double) compute_residual_norm(env->A, env->b, env->x, env->r);
    env->r_curr = env->r0;

    HYPRE_Real dot = 0.0;
    HYPRE_ParVectorInnerProd(env->b, env->r, &dot);
    double denom = env->b_norm * env->r_curr;
    env->cos_br = (denom > 0.0) ? (double)(dot / denom) : 0.0;  // with x0=0, this should be ~1

    return env;
}

int amg_env_step(AMGEnv* env,
    double relax_weight,
    int sweeps_down,
    int sweeps_up,
    double* r_out,
    double* dt_out,
    int* status_out)
{
    if (!env || !env->amg_solver || !env->A || !env->b || !env->x)
    return -1;

    // Clamp sweeps to a safe range
    if (sweeps_down < 1) sweeps_down = 1;
    if (sweeps_down > 10) sweeps_down = 10;
    if (sweeps_up < 1) sweeps_up = 1;
    if (sweeps_up > 10) sweeps_up = 10;

    // Update AMG params for this cycle
    HYPRE_BoomerAMGSetRelaxWt(env->amg_solver, relax_weight);

    // Separate down/up sweeps (Hypre supports this)
    // 1 = down, 2 = up in common Hypre wrappers
    HYPRE_BoomerAMGSetCycleNumSweeps(env->amg_solver, sweeps_down, 1);
    HYPRE_BoomerAMGSetCycleNumSweeps(env->amg_solver, sweeps_up,   2);

    // Time the V-cycle
    double t0 = wall_time_sec();

    // One V-cycle (your code currently uses Solve each step; keep consistent)
    HYPRE_BoomerAMGSolve(env->amg_solver, env->A, env->b, env->x);

    double t1 = wall_time_sec();
    double dt = t1 - t0;

    // Residual norm
    double r = (double) compute_residual_norm(env->A, env->b, env->x, env->r);
    env->r_curr = r;

    env->cycles_done += 1;

    // status: 0=continue, 1=converged, 2=truncated
    int st = 0;
    if (r <= env->tol) st = 1;
    else if (env->cycles_done >= env->max_cycles) st = 2;

    if (r_out) *r_out = r;
    if (dt_out) *dt_out = dt;
    if (status_out) *status_out = st;

    return 0;
}


double amg_env_get_r0(AMGEnv *env)
{
    return env ? env->r0 : 0.0;
}

double amg_env_get_r(AMGEnv *env)
{
    return env ? env->r_curr : 0.0;
}

int amg_env_get_cycle(AMGEnv *env)
{
    return env ? env->cycles_done : 0;
}

double amg_env_get_b_norm(AMGEnv *env) { return env ? env->b_norm : 0.0; }
double amg_env_get_cos_br(AMGEnv *env) { return env ? env->cos_br : 0.0; }



void amg_env_destroy(AMGEnv *env)
{
    if (!env) return;

    HYPRE_BoomerAMGDestroy(env->amg_solver);

    HYPRE_IJVectorDestroy(env->ij_b);
    HYPRE_IJVectorDestroy(env->ij_x);
    HYPRE_IJVectorDestroy(env->ij_r);
    HYPRE_IJMatrixDestroy(env->ij_A);

    free(env);
}
