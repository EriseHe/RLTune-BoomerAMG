/* Project-owned adapters for HYPRE's stock 3D test matrices.
 * 102/103: unmodified GenerateRSVarDifConv types 2/3, corner=epsilon.
 * 100: GenerateVarDifConv's geometry and sparsity, with its three material
 *      coefficients exposed as parameters. This is an adapted family.
 * Homogeneous Dirichlet boundaries and face sampling match the stock source.
 */
#include <math.h>
#include "_hypre_parcsr_ls.h"
#include "HYPRE_parcsr_ls.h"
#include "amg_stress_builders.h"

static double material(double x, double y, double z,
                       double corner, double background, double center)
{
    if ((x < 0.1 || x > 0.9) && (y < 0.1 || y > 0.9) &&
        (z < 0.1 || z > 0.9)) return corner;
    if (x >= 0.1 && x <= 0.9 && y >= 0.1 && y <= 0.9 &&
        z >= 0.1 && z <= 0.9) return center;
    return background;
}

HYPRE_ParCSRMatrix amg_build_stress_matrix(
    MPI_Comm comm, int nx, int ny, int nz, int kind,
    double corner, double background, double center)
{
    int size;
    hypre_MPI_Comm_size(comm, &size);
    if (size != 1 || nx < 2 || ny < 2 || nz < 2 ||
        !isfinite(corner) || corner <= 0.0) return NULL;
    HYPRE_ParVector rhs = NULL;
    HYPRE_ParCSRMatrix matrix = NULL;
    if (kind == 102 || kind == 103)
    {
        matrix = GenerateRSVarDifConv(comm, nx, ny, nz, 1, 1, 1, 0, 0, 0,
                                      corner, &rhs, kind - 100);
        if (rhs) HYPRE_ParVectorDestroy(rhs);
        return matrix;
    }
    if (kind != 100 || !isfinite(background) || background <= 0.0 ||
        !isfinite(center) || center <= 0.0) return NULL;

    matrix = GenerateVarDifConv(comm, nx, ny, nz, 1, 1, 1, 0, 0, 0, 1.0, &rhs);
    if (rhs) HYPRE_ParVectorDestroy(rhs);
    if (!matrix) return NULL;
    hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)matrix);
    HYPRE_Int *row_ptr = hypre_CSRMatrixI(diag);
    HYPRE_Int *columns = hypre_CSRMatrixJ(diag);
    HYPRE_Complex *values = hypre_CSRMatrixData(diag);
    double hx = 1.0 / (nx + 1), hy = 1.0 / (ny + 1), hz = 1.0 / (nz + 1);
    for (int row = 0; row < nx * ny * nz; ++row)
    {
        double x = (row % nx + 1) * hx;
        double y = ((row / nx) % ny + 1) * hy;
        double z = (row / (nx * ny) + 1) * hz;
        double weights[6] = {
            -material(x - 0.5 * hx, y, z, corner, background, center) / hx / hx,
            -material(x + 0.5 * hx, y, z, corner, background, center) / hx / hx,
            -material(x, y - 0.5 * hy, z, corner, background, center) / hy / hy,
            -material(x, y + 0.5 * hy, z, corner, background, center) / hy / hy,
            -material(x, y, z - 0.5 * hz, corner, background, center) / hz / hz,
            -material(x, y, z + 0.5 * hz, corner, background, center) / hz / hz
        };
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j)
        {
            int delta = columns[j] - row;
            if (delta == 0)
                values[j] = -(weights[0] + weights[1] + weights[2] +
                              weights[3] + weights[4] + weights[5]);
            else if (delta == -1) values[j] = weights[0];
            else if (delta == 1) values[j] = weights[1];
            else if (delta == -nx) values[j] = weights[2];
            else if (delta == nx) values[j] = weights[3];
            else if (delta == -nx * ny) values[j] = weights[4];
            else if (delta == nx * ny) values[j] = weights[5];
        }
    }
    return matrix;
}
