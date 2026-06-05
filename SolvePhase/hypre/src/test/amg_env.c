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

#include "../../../../common/amg_rl_shared.h"

/* ---------- small utilities ---------- */

/* returns the sign of a real number: 1 positive, 0 zero, -1 negative */
static inline HYPRE_Int sign_double(HYPRE_Real a)
{
    return ( (0.0 < a) - (0.0 > a) );
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
    double             setup_time;
    int                max_cycles;
    int                cycles_done;
    int                cycle_type;  // 1=V, 2=W, etc.
    int                relax_type;  // AMG relaxation type

    int                nx, ny, nz;
    int                stencil_type;  // 7 or 27 (Laplacian), 0 = difconv
    int                rhs_type;      // 0=ones, 1=random
    double b_norm;
    double cos_br;  // cosine between b and current residual r
} AMGEnv;

void amg_env_destroy(AMGEnv *env);

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

/* ---------- Matrix builders ---------- */

static HYPRE_IJMatrix build_ij_laplacian(
    int nx, int ny, int nz,
    int stencil_type,
    double k, double c,
    double a0, double a1, double a2, double a3)
{
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

    return ij_A;
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
    env->setup_time = 0.0;

    /* ----------- Build matrix A ----------- */
    HYPRE_IJMatrix ij_A = NULL;

    if (stencil_type == 0)
    {
        /* DifConv matrix (from ij.c):
           Map args as: cx=k, cy=c, cz=a0, ax=a1, ay=a2, az=a3, atype=0. */
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
        /* Reuse Laplacian builders from amg_cycle.c (IJ interface) */
        ij_A = build_ij_laplacian(nx, ny, nz, stencil_type, k, c, a0, a1, a2, a3);
        env->ij_A = ij_A;

        void *object = NULL;
        HYPRE_IJMatrixGetObject(env->ij_A, &object);
        env->A = (HYPRE_ParCSRMatrix) object;
    }

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
            double u1 = (double)(seed & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
            seed = 6364136223846793005ULL * seed + 1ULL;
            double u2 = (double)(seed & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;

            /* Box-Muller: Gaussian(0,1) */
            if (u1 < 1e-12) u1 = 1e-12;
            double z = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
            vals[i] = z;
        }
    }

    HYPRE_IJVectorSetValues(env->ij_b, local_num_rows, rows, vals);
    HYPRE_IJVectorAssemble(env->ij_b);

    free(vals);
    free(rows);

    /* Grab ParVector objects */
    void *object = NULL;
    HYPRE_IJVectorGetObject(env->ij_b, &object);
    env->b = (HYPRE_ParVector) object;

    HYPRE_IJVectorGetObject(env->ij_x, &object);
    env->x = (HYPRE_ParVector) object;

    HYPRE_IJVectorGetObject(env->ij_r, &object);
    env->r = (HYPRE_ParVector) object;

    env->b_norm = (double) amg_rl_parvec_norm2(env->b);

    /* ----------- AMG setup: 1 V-cycle per Solve call ----------- */
    HYPRE_BoomerAMGCreate(&env->amg_solver);
    int relax_type = -1;
    int cycle_type = -1;
    if (amg_rl_prepare_solver(
            env->amg_solver,
            env->A,
            env->b,
            env->x,
            env->r,
            &relax_type,
            &cycle_type,
            &env->setup_time,
            &env->r0) != 0)
    {
        amg_env_destroy(env);
        return NULL;
    }

    /* ----------- Compute initial residual r0 = ||b - A x0|| ----------- */
    env->r_curr = env->r0;
    env->cycle_type = cycle_type;
    env->relax_type = relax_type;

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

    double r = 0.0;
    double dt = 0.0;
    int st = 0;
    if (amg_rl_step_solver(
            env->amg_solver,
            env->A,
            env->b,
            env->x,
            env->r,
            env->tol,
            env->max_cycles,
            &env->cycles_done,
            relax_weight,
            sweeps_down,
            sweeps_up,
            1,                /* coarse_sweeps */
            env->cycle_type,  /* cycle_type */
            env->relax_type,  /* relax_type */
            -1,               /* pre_relax_type */
            -1,               /* post_relax_type */
            -1,               /* coarse_relax_type */
            -1,               /* relax_order */
            -1.0,             /* outer_weight */
            -1.0,             /* add_relax_weight */
            -1.0,             /* level_relax_weight */
            -1,               /* level_relax_level */
            -1.0,             /* level_outer_weight */
            -1,               /* level_outer_level */
            &r,
            &dt,
            &st) != 0)
    {
        return -1;
    }
    env->r_curr = r;

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

double amg_env_get_setup_time(AMGEnv *env)
{
    return env ? env->setup_time : 0.0;
}

double amg_env_get_b_norm(AMGEnv *env) { return env ? env->b_norm : 0.0; }
double amg_env_get_cos_br(AMGEnv *env) { return env ? env->cos_br : 0.0; }
int amg_env_get_cycle_type(AMGEnv *env) { return env ? env->cycle_type : -1; }
int amg_env_get_relax_type(AMGEnv *env) { return env ? env->relax_type : -1; }



void amg_env_destroy(AMGEnv *env)
{
    if (!env) return;

    HYPRE_BoomerAMGDestroy(env->amg_solver);

    HYPRE_IJVectorDestroy(env->ij_b);
    HYPRE_IJVectorDestroy(env->ij_x);
    HYPRE_IJVectorDestroy(env->ij_r);
    if (env->ij_A)
    {
        HYPRE_IJMatrixDestroy(env->ij_A);
    }
    else if (env->A)
    {
        HYPRE_ParCSRMatrixDestroy(env->A);
    }

    free(env);
}
