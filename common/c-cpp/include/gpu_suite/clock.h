#ifndef GPU_SUITE_CLOCK_H
#define GPU_SUITE_CLOCK_H

#include "gpu_suite/common.h"

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int gpu_suite_clock_resolution(double *resolution_sec, char *error,
                               size_t error_size);
int gpu_suite_clock_now(struct timespec *value, char *error, size_t error_size);
double gpu_suite_clock_elapsed(const struct timespec *start,
                               const struct timespec *end);
int gpu_suite_utc_timestamp(char output[GPU_SUITE_TIMESTAMP_CAPACITY],
                            char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
