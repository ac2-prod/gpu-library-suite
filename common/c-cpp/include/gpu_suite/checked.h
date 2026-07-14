#ifndef GPU_SUITE_CHECKED_H
#define GPU_SUITE_CHECKED_H

#include "gpu_suite/common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool gpu_suite_checked_add_size(size_t left, size_t right, size_t *result);
bool gpu_suite_checked_mul_size(size_t left, size_t right, size_t *result);
bool gpu_suite_checked_u64_to_size(uint64_t value, size_t *result);

#ifdef __cplusplus
}
#endif

#endif
