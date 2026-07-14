#include "gpu_suite/checked.h"

#include <stdint.h>

bool gpu_suite_checked_add_size(size_t left, size_t right, size_t *result) {
  if (result == NULL || left > SIZE_MAX - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool gpu_suite_checked_mul_size(size_t left, size_t right, size_t *result) {
  if (result == NULL || (left != 0U && right > SIZE_MAX / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool gpu_suite_checked_u64_to_size(uint64_t value, size_t *result) {
  if (result == NULL || value > (uint64_t)SIZE_MAX) {
    return false;
  }
  *result = (size_t)value;
  return true;
}
