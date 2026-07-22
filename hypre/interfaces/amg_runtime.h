#ifndef RLTUNE_AMG_RUNTIME_H
#define RLTUNE_AMG_RUNTIME_H

#if defined(_WIN32)
#define AMG_API __declspec(dllexport)
#else
#define AMG_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AMGRuntime AMGRuntime;

typedef enum
{
    AMG_RUNTIME_OK = 0,
    AMG_RUNTIME_ERR_INVALID_ARGUMENT = -1,
    AMG_RUNTIME_ERR_CREATE = -2,
    AMG_RUNTIME_ERR_SETUP = -3,
    AMG_RUNTIME_ERR_SOLVE = -4,
    AMG_RUNTIME_ERR_NONFINITE = -5
} AMGRuntimeCode;

typedef enum
{
    AMG_RUNTIME_STATUS_CONTINUE = 0,
    AMG_RUNTIME_STATUS_CONVERGED = 1,
    AMG_RUNTIME_STATUS_MAX_CYCLES = 2
} AMGRuntimeSolveStatus;

AMG_API AMGRuntime *amg_runtime_create(
    int nx, int ny, int nz,
    int stencil_type,
    int rhs_type,
    unsigned long long rhs_seed,
    double k, double c,
    double a0, double a1, double a2, double a3);

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
    int *out_status);

AMG_API int amg_runtime_prepare(
    AMGRuntime *env,
    double strong_threshold, int coarsen_type, int interp_type, double max_row_sum,
    int relax_type, int num_sweeps, int cycle_type, int max_levels,
    double trunc_factor, int P_max_elmts,
    int agg_num_levels, int agg_interp_type,
    double agg_tr, int agg_Pmx,
    double relax_wt, int relax_order, int max_coarse_size,
    double *out_setup_runtime,
    double *out_r0);

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
    int *out_status);

AMG_API void amg_runtime_destroy(AMGRuntime *env);
AMG_API int amg_runtime_get_n(AMGRuntime *env);
AMG_API int amg_runtime_get_nnz(AMGRuntime *env);
AMG_API double amg_runtime_get_r0(AMGRuntime *env);
AMG_API double amg_runtime_get_r(AMGRuntime *env);
AMG_API int amg_runtime_get_cycle(AMGRuntime *env);
AMG_API double amg_runtime_get_setup_time(AMGRuntime *env);
AMG_API int amg_runtime_get_cycle_type(AMGRuntime *env);
AMG_API int amg_runtime_get_relax_type(AMGRuntime *env);
AMG_API int amg_runtime_get_relax_weight(AMGRuntime *env, int level, double *out_weight);

#ifdef __cplusplus
}
#endif

#endif
