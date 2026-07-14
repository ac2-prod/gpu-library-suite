#include <mkl_spblas.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int make_poisson2d_csr(int nx, int ny, int *row, int *col, double *val) {
  int p = 0, offset = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      p = iy * nx + ix;
      row[p] = offset;
      if (iy > 0) {
        col[offset] = p - nx;
        val[offset++] = -1.0;
      }
      if (ix > 0) {
        col[offset] = p - 1;
        val[offset++] = -1.0;
      }
      col[offset] = p;
      val[offset++] = 4.0;
      if (ix + 1 < nx) {
        col[offset] = p + 1;
        val[offset++] = -1.0;
      }
      if (iy + 1 < ny) {
        col[offset] = p + nx;
        val[offset++] = -1.0;
      }
    }
  row[nx * ny] = offset;
  return offset;
}

int main(void) {
  const int nx = 1024, ny = 1024, n = nx * ny, nnz = 5 * n - 2 * nx - 2 * ny;
  int *row = (int *)malloc((size_t)(n + 1) * sizeof(*row));
  int *col = (int *)malloc((size_t)nnz * sizeof(*col));
  double *val = (double *)malloc((size_t)nnz * sizeof(*val));
  double *x = (double *)malloc((size_t)n * sizeof(*x));
  double *y = (double *)malloc((size_t)n * sizeof(*y));
  sparse_matrix_t matrix = NULL;
  int status = EXIT_FAILURE;
  double max_error = 0.0;
  if (!row || !col || !val || !x || !y) {
    fprintf(stderr, "CSR allocation failed\n");
    goto cleanup;
  }
  if (make_poisson2d_csr(nx, ny, row, col, val) != nnz) {
    fprintf(stderr, "CSR generation failed\n");
    goto cleanup;
  }
  for (int i = 0; i < n; ++i) {
    x[i] = 1.0;
    y[i] = 1.0;
  }
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.mode = SPARSE_FILL_MODE_FULL;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  if (mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, n, n, row,
                              row + 1, col, val) != SPARSE_STATUS_SUCCESS ||
      mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix, descr, x,
                      1.0, y) != SPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "oneMKL Sparse SpMV failed\n");
    goto cleanup;
  }
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int neighbors =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      double expected = 5.0 - neighbors;
      max_error = fmax(max_error, fabs(y[iy * nx + ix] - expected));
    }
  printf("CPU Poisson CSR SpMV complete; max error = %.17g\n", max_error);
  status = max_error <= 1.0e-12 ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
  if (matrix != NULL &&
      mkl_sparse_destroy(matrix) != SPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "mkl_sparse_destroy failed\n");
    status = EXIT_FAILURE;
  }
  free(y);
  free(x);
  free(val);
  free(col);
  free(row);
  return status;
}
