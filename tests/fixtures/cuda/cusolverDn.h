#ifndef GPU_SUITE_TEST_CUSOLVER_DN_H
#define GPU_SUITE_TEST_CUSOLVER_DN_H

#include "cublas_v2.h"

typedef void *cusolverDnHandle_t;
typedef int cusolverStatus_t;

enum { CUSOLVER_STATUS_SUCCESS = 0 };
typedef enum {
  MAJOR_VERSION = 0,
  MINOR_VERSION = 1,
  PATCH_LEVEL = 2
} libraryPropertyType;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int *value);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, double *a, int lda,
                                             int *workspace_size);
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                  double *a, int lda, double *workspace,
                                  int *pivots, int *info);
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans, int n, int nrhs,
                                  const double *a, int lda, const int *pivots,
                                  double *b, int ldb, int *info);

#endif
