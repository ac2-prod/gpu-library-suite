#ifndef GPU_SUITE_TEST_CUBLAS_V2_H
#define GPU_SUITE_TEST_CUBLAS_V2_H

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;

enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_OP_N = 0 };

cublasStatus_t cublasCreate(cublasHandle_t *handle);
cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t trans_a,
                           cublasOperation_t trans_b, int m, int n, int k,
                           const double *alpha, const double *a, int lda,
                           const double *b, int ldb, const double *beta,
                           double *c, int ldc);

#endif
