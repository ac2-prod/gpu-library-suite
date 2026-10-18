#ifndef GPU_SUITE_FIXTURE_AOCLSPARSE_H
#define GPU_SUITE_FIXTURE_AOCLSPARSE_H
/* Independent host fixture for the AOCL Sparse 5.1 C API subset; not AOCL. */
#ifdef FIXTURE_ILP64
typedef long long aoclsparse_int;
#else
typedef int aoclsparse_int;
#endif
typedef enum { aoclsparse_status_success = 0, aoclsparse_status_internal_error = 1 } aoclsparse_status;
typedef enum { aoclsparse_index_base_zero = 0 } aoclsparse_index_base;
typedef enum { aoclsparse_operation_none = 0 } aoclsparse_operation;
typedef enum { aoclsparse_matrix_type_general = 0 } aoclsparse_matrix_type;
typedef enum { aoclsparse_diag_type_non_unit = 0 } aoclsparse_diag_type;
typedef struct fixture_matrix *aoclsparse_matrix;
typedef struct fixture_descriptor *aoclsparse_mat_descr;
#ifdef __cplusplus
extern "C" {
#endif
const char *aoclsparse_get_version(void);
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr *descr);
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr descr, aoclsparse_index_base base);
aoclsparse_status aoclsparse_set_mat_type(aoclsparse_mat_descr descr, aoclsparse_matrix_type type);
aoclsparse_status aoclsparse_set_mat_diag_type(aoclsparse_mat_descr descr, aoclsparse_diag_type diag);
aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix *matrix, aoclsparse_index_base base,
    aoclsparse_int m, aoclsparse_int n, aoclsparse_int nnz,
    aoclsparse_int *row, aoclsparse_int *col, double *val);
aoclsparse_status aoclsparse_dmv(aoclsparse_operation op, const double *alpha,
    aoclsparse_matrix matrix, const aoclsparse_mat_descr descr,
    const double *x, const double *beta, double *y);
aoclsparse_status aoclsparse_set_mv_hint(aoclsparse_matrix matrix,
    aoclsparse_operation op, const aoclsparse_mat_descr descr,
    aoclsparse_int expected_no_of_calls);
aoclsparse_status aoclsparse_optimize(aoclsparse_matrix matrix);
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix *matrix);
aoclsparse_status aoclsparse_destroy_mat_descr(aoclsparse_mat_descr descr);
#ifdef __cplusplus
}
#endif
#endif
