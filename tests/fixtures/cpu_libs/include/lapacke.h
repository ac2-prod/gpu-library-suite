#ifndef GPU_SUITE_TEST_LAPACKE_H
#define GPU_SUITE_TEST_LAPACKE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int lapack_int;

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

lapack_int LAPACKE_dgesv(int matrix_layout, lapack_int n, lapack_int nrhs,
                         double *a, lapack_int lda, lapack_int *ipiv, double *b,
                         lapack_int ldb);
lapack_int LAPACKE_dgetrf(int matrix_layout, lapack_int m, lapack_int n,
                          double *a, lapack_int lda, lapack_int *ipiv);
lapack_int LAPACKE_dgetrs(int matrix_layout, char trans, lapack_int n,
                          lapack_int nrhs, const double *a, lapack_int lda,
                          const lapack_int *ipiv, double *b, lapack_int ldb);

#ifdef __cplusplus
}
#endif

#endif
