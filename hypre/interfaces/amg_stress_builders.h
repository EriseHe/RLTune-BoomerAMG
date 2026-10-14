#ifndef RLTUNE_AMG_STRESS_BUILDERS_H
#define RLTUNE_AMG_STRESS_BUILDERS_H

/* Exploratory 3D builders; all calls currently require one MPI rank. */
HYPRE_ParCSRMatrix amg_build_stress_matrix(
    MPI_Comm comm, int nx, int ny, int nz, int kind,
    double corner, double background, double center);

#endif
