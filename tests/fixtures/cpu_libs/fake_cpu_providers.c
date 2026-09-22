#include "cblas.h"
#include "lapacke.h"
#include "mkl_spblas.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int fixture_nonfinite(double *value) {
  const char *requested = getenv("GPU_SUITE_TEST_NONFINITE");
  if (requested == NULL)
    return 0;
  if (strcmp(requested, "nan") == 0)
    *value = NAN;
  else if (strcmp(requested, "inf") == 0)
    *value = INFINITY;
  else if (strcmp(requested, "-inf") == 0)
    *value = -INFINITY;
  else
    return 0;
  return 1;
}

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
                 CBLAS_TRANSPOSE trans_b, int m, int n, int k, double alpha,
                 const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc) {
  if (order != CblasColMajor || trans_a != CblasNoTrans ||
      trans_b != CblasNoTrans) {
    return;
  }
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < m; ++row) {
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        sum += a[row + (size_t)inner * (size_t)lda] *
               b[inner + (size_t)column * (size_t)ldb];
      }
      c[row + (size_t)column * (size_t)ldc] =
          alpha * sum + beta * c[row + (size_t)column * (size_t)ldc];
    }
  }
  if (m > 0 && n > 0)
    (void)fixture_nonfinite(&c[0]);
}

lapack_int LAPACKE_dgetrf(int matrix_layout, lapack_int m, lapack_int n,
                          double *a, lapack_int lda, lapack_int *ipiv) {
  if (matrix_layout != LAPACK_COL_MAJOR)
    return -1;
  const lapack_int count = m < n ? m : n;
  for (lapack_int column = 0; column < count; ++column) {
    lapack_int pivot = column;
    for (lapack_int row = column + 1; row < m; ++row) {
      if (fabs(a[row + (size_t)column * (size_t)lda]) >
          fabs(a[pivot + (size_t)column * (size_t)lda])) {
        pivot = row;
      }
    }
    ipiv[column] = pivot + 1;
    if (a[pivot + (size_t)column * (size_t)lda] == 0.0)
      return column + 1;
    if (pivot != column) {
      for (lapack_int j = 0; j < n; ++j) {
        const size_t first = column + (size_t)j * (size_t)lda;
        const size_t second = pivot + (size_t)j * (size_t)lda;
        const double temporary = a[first];
        a[first] = a[second];
        a[second] = temporary;
      }
    }
    for (lapack_int row = column + 1; row < m; ++row) {
      a[row + (size_t)column * (size_t)lda] /=
          a[column + (size_t)column * (size_t)lda];
      for (lapack_int j = column + 1; j < n; ++j) {
        a[row + (size_t)j * (size_t)lda] -=
            a[row + (size_t)column * (size_t)lda] *
            a[column + (size_t)j * (size_t)lda];
      }
    }
  }
  return 0;
}

lapack_int LAPACKE_dgetrs(int matrix_layout, char trans, lapack_int n,
                          lapack_int nrhs, const double *a, lapack_int lda,
                          const lapack_int *ipiv, double *b, lapack_int ldb) {
  if (matrix_layout != LAPACK_COL_MAJOR || trans != 'N')
    return -1;
  for (lapack_int column = 0; column < n; ++column) {
    const lapack_int pivot = ipiv[column] - 1;
    if (pivot != column) {
      for (lapack_int rhs = 0; rhs < nrhs; ++rhs) {
        const size_t first = column + (size_t)rhs * (size_t)ldb;
        const size_t second = pivot + (size_t)rhs * (size_t)ldb;
        const double temporary = b[first];
        b[first] = b[second];
        b[second] = temporary;
      }
    }
  }
  for (lapack_int rhs = 0; rhs < nrhs; ++rhs) {
    for (lapack_int row = 0; row < n; ++row) {
      for (lapack_int inner = 0; inner < row; ++inner) {
        b[row + (size_t)rhs * (size_t)ldb] -=
            a[row + (size_t)inner * (size_t)lda] *
            b[inner + (size_t)rhs * (size_t)ldb];
      }
    }
    for (lapack_int row = n; row-- > 0;) {
      for (lapack_int inner = row + 1; inner < n; ++inner) {
        b[row + (size_t)rhs * (size_t)ldb] -=
            a[row + (size_t)inner * (size_t)lda] *
            b[inner + (size_t)rhs * (size_t)ldb];
      }
      b[row + (size_t)rhs * (size_t)ldb] /= a[row + (size_t)row * (size_t)lda];
    }
  }
  if (n > 0 && nrhs > 0)
    (void)fixture_nonfinite(&b[0]);
  return 0;
}

lapack_int LAPACKE_dgesv(int matrix_layout, lapack_int n, lapack_int nrhs,
                         double *a, lapack_int lda, lapack_int *ipiv, double *b,
                         lapack_int ldb) {
  const lapack_int factor_info =
      LAPACKE_dgetrf(matrix_layout, n, n, a, lda, ipiv);
  if (factor_info != 0)
    return factor_info;
  return LAPACKE_dgetrs(matrix_layout, 'N', n, nrhs, a, lda, ipiv, b, ldb);
}

struct gpu_suite_fake_sparse_matrix {
  sparse_index_base_t indexing;
  MKL_INT rows;
  MKL_INT columns;
  MKL_INT *rows_start;
  MKL_INT *rows_end;
  MKL_INT *column_indices;
  double *values;
};

sparse_status_t mkl_sparse_d_create_csr(sparse_matrix_t *matrix,
                                        sparse_index_base_t indexing,
                                        MKL_INT rows, MKL_INT columns,
                                        MKL_INT *rows_start, MKL_INT *rows_end,
                                        MKL_INT *column_indices,
                                        double *values) {
  if (matrix == NULL || rows_start == NULL || rows_end == NULL ||
      column_indices == NULL || values == NULL || rows <= 0 || columns <= 0) {
    return SPARSE_STATUS_INVALID_VALUE;
  }
  sparse_matrix_t created = malloc(sizeof(*created));
  if (created == NULL)
    return SPARSE_STATUS_ALLOC_FAILED;
  created->indexing = indexing;
  created->rows = rows;
  created->columns = columns;
  created->rows_start = rows_start;
  created->rows_end = rows_end;
  created->column_indices = column_indices;
  created->values = values;
  *matrix = created;
  return SPARSE_STATUS_SUCCESS;
}

sparse_status_t mkl_sparse_set_mv_hint(sparse_matrix_t matrix,
                                       sparse_operation_t operation,
                                       struct matrix_descr descriptor,
                                       MKL_INT expected_calls) {
  (void)descriptor;
  return matrix != NULL && operation == SPARSE_OPERATION_NON_TRANSPOSE &&
                 expected_calls > 0
             ? SPARSE_STATUS_SUCCESS
             : SPARSE_STATUS_INVALID_VALUE;
}

sparse_status_t mkl_sparse_optimize(sparse_matrix_t matrix) {
  return matrix != NULL ? SPARSE_STATUS_SUCCESS : SPARSE_STATUS_INVALID_VALUE;
}

sparse_status_t mkl_sparse_d_mv(sparse_operation_t operation, double alpha,
                                sparse_matrix_t matrix,
                                struct matrix_descr descriptor, const double *x,
                                double beta, double *y) {
  (void)descriptor;
  if (matrix == NULL || x == NULL || y == NULL ||
      operation != SPARSE_OPERATION_NON_TRANSPOSE) {
    return SPARSE_STATUS_INVALID_VALUE;
  }
  const MKL_INT adjustment = matrix->indexing == SPARSE_INDEX_BASE_ONE ? 1 : 0;
  for (MKL_INT row = 0; row < matrix->rows; ++row) {
    double sum = 0.0;
    const MKL_INT begin = matrix->rows_start[row] - adjustment;
    const MKL_INT end = matrix->rows_end[row] - adjustment;
    for (MKL_INT index = begin; index < end; ++index) {
      const MKL_INT column = matrix->column_indices[index] - adjustment;
      sum += matrix->values[index] * x[column];
    }
    y[row] = alpha * sum + beta * y[row];
  }
  if (matrix->rows > 0)
    (void)fixture_nonfinite(&y[0]);
  return SPARSE_STATUS_SUCCESS;
}

sparse_status_t mkl_sparse_destroy(sparse_matrix_t matrix) {
  free(matrix);
  return SPARSE_STATUS_SUCCESS;
}
