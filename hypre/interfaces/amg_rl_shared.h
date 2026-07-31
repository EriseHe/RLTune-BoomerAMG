#ifndef AMG_RL_SHARED_H
#define AMG_RL_SHARED_H

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

double amg_rl_wall_time_sec(void);
HYPRE_Real amg_rl_parvec_norm2(HYPRE_ParVector v);
HYPRE_Real amg_rl_compute_residual_norm(HYPRE_ParCSRMatrix A,
                                        HYPRE_ParVector b,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector r);

int amg_rl_default_relax_type(void);
int amg_rl_default_cycle_type(void);

int amg_rl_prepare_solver(HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x,
                          HYPRE_ParVector r,
                          int *relax_type_io,
                          int *cycle_type_io,
                          double *out_setup_time,
                          double *out_r0);

int amg_rl_step_solver(HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,
                       HYPRE_ParVector x,
                       HYPRE_ParVector r,
                       double tol,
                       double initial_residual,
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
                       int *out_status);

#ifdef __cplusplus
}
#endif

#endif
