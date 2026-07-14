#ifndef GPU_SUITE_TEST_MKL_SPBLAS_H
#define GPU_SUITE_TEST_MKL_SPBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int MKL_INT;

typedef enum {
  SPARSE_STATUS_SUCCESS = 0,
  SPARSE_STATUS_NOT_INITIALIZED = 1,
  SPARSE_STATUS_ALLOC_FAILED = 2,
  SPARSE_STATUS_INVALID_VALUE = 3
} sparse_status_t;
typedef enum {
  SPARSE_INDEX_BASE_ZERO = 0,
  SPARSE_INDEX_BASE_ONE = 1
} sparse_index_base_t;
typedef enum { SPARSE_OPERATION_NON_TRANSPOSE = 10 } sparse_operation_t;
typedef enum { SPARSE_MATRIX_TYPE_GENERAL = 20 } sparse_matrix_type_t;
typedef enum { SPARSE_FILL_MODE_FULL = 30 } sparse_fill_mode_t;
typedef enum { SPARSE_DIAG_NON_UNIT = 40 } sparse_diag_type_t;

struct matrix_descr {
  sparse_matrix_type_t type;
  sparse_fill_mode_t mode;
  sparse_diag_type_t diag;
};

struct gpu_suite_fake_sparse_matrix;
typedef struct gpu_suite_fake_sparse_matrix *sparse_matrix_t;

sparse_status_t mkl_sparse_d_create_csr(sparse_matrix_t *matrix,
                                        sparse_index_base_t indexing,
                                        MKL_INT rows, MKL_INT columns,
                                        MKL_INT *rows_start, MKL_INT *rows_end,
                                        MKL_INT *column_indices,
                                        double *values);
sparse_status_t mkl_sparse_set_mv_hint(sparse_matrix_t matrix,
                                       sparse_operation_t operation,
                                       struct matrix_descr descriptor,
                                       MKL_INT expected_calls);
sparse_status_t mkl_sparse_optimize(sparse_matrix_t matrix);
sparse_status_t mkl_sparse_d_mv(sparse_operation_t operation, double alpha,
                                sparse_matrix_t matrix,
                                struct matrix_descr descriptor, const double *x,
                                double beta, double *y);
sparse_status_t mkl_sparse_destroy(sparse_matrix_t matrix);

#ifdef __cplusplus
}
#endif

#endif
