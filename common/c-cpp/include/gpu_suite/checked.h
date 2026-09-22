#ifndef GPU_SUITE_CHECKED_H
#define GPU_SUITE_CHECKED_H

#include "gpu_suite/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool gpu_suite_checked_add_size(size_t left, size_t right, size_t *result);
bool gpu_suite_checked_mul_size(size_t left, size_t right, size_t *result);
bool gpu_suite_checked_mul3_size(size_t first, size_t second, size_t third,
                                 size_t *result);
bool gpu_suite_checked_bytes(size_t element_count, size_t element_size,
                             size_t *result);
bool gpu_suite_checked_u64_to_int(uint64_t value, int *result);
bool gpu_suite_checked_u64_to_size(uint64_t value, size_t *result);
bool gpu_suite_checked_poisson2d_dimensions(uint64_t nx_value,
                                            uint64_t ny_value, int *nx,
                                            int *ny, int *point_count,
                                            int *nonzero_count,
                                            size_t *row_offset_count);
bool gpu_suite_finite_absolute_error(double observed, double expected,
                                     double *error);
bool gpu_suite_finite_max_update(double candidate, double *maximum);

#ifdef __cplusplus
}
#endif

#endif
