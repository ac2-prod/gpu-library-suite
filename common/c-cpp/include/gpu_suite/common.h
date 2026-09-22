#ifndef GPU_SUITE_COMMON_H
#define GPU_SUITE_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_SUITE_RUN_ID_CAPACITY 129
#define GPU_SUITE_LABEL_CAPACITY 256
#define GPU_SUITE_PATH_CAPACITY 4096
#define GPU_SUITE_NAME_CAPACITY 96
#define GPU_SUITE_ORDER_CAPACITY 32
#define GPU_SUITE_TIMESTAMP_CAPACITY 25

enum {
  GPU_SUITE_OK = 0,
  GPU_SUITE_ERROR_INVALID = 1,
  GPU_SUITE_ERROR_OVERFLOW = 2,
  GPU_SUITE_ERROR_NOMEM = 3,
  GPU_SUITE_ERROR_IO = 4,
  GPU_SUITE_ERROR_FORMAT = 5,
  GPU_SUITE_ERROR_EXISTS = 6
};

bool gpu_suite_utf8_validate(const char *text);

#ifdef __cplusplus
}
#endif

#endif
