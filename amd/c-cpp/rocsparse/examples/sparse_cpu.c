#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#elif _POSIX_C_SOURCE < 200809L
#error "AMD CPU library headers require _POSIX_C_SOURCE >= 200809L"
#endif

#ifdef GPU_SUITE_USE_MKL_SPARSE
#include <mkl_spblas.h>
#elif defined(GPU_SUITE_USE_AOCL_SPARSE)
#include <aoclsparse.h>
#include <limits.h>
_Static_assert(sizeof(aoclsparse_int) * CHAR_BIT == 32,
               "AOCL Sparse requires LP64 32-bit indices");
_Static_assert(_Generic((aoclsparse_int)0, int: 1, default: 0),
               "AOCL Sparse index type must match int");
#endif

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
#ifdef GPU_SUITE_USE_MKL_SPARSE
  sparse_matrix_t matrix = NULL;
#elif defined(GPU_SUITE_USE_AOCL_SPARSE)
  aoclsparse_matrix matrix = NULL;
  aoclsparse_mat_descr descr = NULL;
#endif
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
#ifdef GPU_SUITE_USE_MKL_SPARSE
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
#elif defined(GPU_SUITE_USE_AOCL_SPARSE)
  const double alpha = 1.0, beta = 1.0;
  if (aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success ||
      aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero) != aoclsparse_status_success ||
      aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general) != aoclsparse_status_success ||
      aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit) != aoclsparse_status_success ||
      aoclsparse_create_dcsr(&matrix, aoclsparse_index_base_zero, n, n, nnz,
                            row, col, val) != aoclsparse_status_success ||
      aoclsparse_set_mv_hint(matrix, aoclsparse_operation_none, descr, 1) != aoclsparse_status_success ||
      aoclsparse_optimize(matrix) != aoclsparse_status_success ||
      aoclsparse_dmv(aoclsparse_operation_none, &alpha, matrix, descr, x,
                    &beta, y) != aoclsparse_status_success) {
    fprintf(stderr, "AOCL Sparse SpMV failed\n");
    goto cleanup;
  }
#else
  // Explicit reference backend, not a replacement for a requested library.
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = row[i]; j < row[i + 1]; ++j)
      sum += val[j] * x[col[j]];
    y[i] = sum + y[i];
  }
#endif
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int neighbors =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      double expected = 5.0 - neighbors;
      max_error = isfinite(y[iy * nx + ix]) ? fmax(max_error, fabs(y[iy * nx + ix] - expected)) : INFINITY;
    }
#ifdef GPU_SUITE_USE_MKL_SPARSE
  printf("oneMKL CPU Poisson CSR SpMV complete; max error = %.17g\n", max_error);
#elif defined(GPU_SUITE_USE_AOCL_SPARSE)
  printf("AOCL Sparse CPU Poisson CSR SpMV complete; max error = %.17g\n", max_error);
#else
  printf("Reference CPU Poisson CSR SpMV complete; max error = %.17g\n", max_error);
#endif
  status = max_error <= 1.0e-12 ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
#ifdef GPU_SUITE_USE_MKL_SPARSE
  if (matrix != NULL &&
      mkl_sparse_destroy(matrix) != SPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "mkl_sparse_destroy failed\n");
    status = EXIT_FAILURE;
  }
#elif defined(GPU_SUITE_USE_AOCL_SPARSE)
  if (matrix && aoclsparse_destroy(&matrix) != aoclsparse_status_success) {
    fprintf(stderr, "aoclsparse_destroy failed\n");
    status = EXIT_FAILURE;
  }
  if (descr && aoclsparse_destroy_mat_descr(descr) != aoclsparse_status_success) {
    fprintf(stderr, "aoclsparse_destroy_mat_descr failed\n");
    status = EXIT_FAILURE;
  }
#endif
  free(y);
  free(x);
  free(val);
  free(col);
  free(row);
  return status;
}
