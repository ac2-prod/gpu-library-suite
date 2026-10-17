#ifndef GPU_SUITE_FAKE_ROCBLAS_H
#define GPU_SUITE_FAKE_ROCBLAS_H
// Host-only API fixture; not an AMD library implementation.
#include <stddef.h>
typedef struct fake_blas_handle *rocblas_handle;
enum rocblas_status { rocblas_status_success, rocblas_status_internal_error };
enum rocblas_operation { rocblas_operation_none };
rocblas_status rocblas_create_handle(rocblas_handle *);
rocblas_status rocblas_destroy_handle(rocblas_handle);
rocblas_status rocblas_dgemm(rocblas_handle, rocblas_operation, rocblas_operation,
    int, int, int, const double *, const double *, int, const double *, int,
    const double *, double *, int);
rocblas_status rocblas_get_version_string(char *, size_t);
#endif
