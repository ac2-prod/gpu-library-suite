#include "gpu_suite/checked.h"

#include "test_support.h"
#include <stdint.h>

int main(void) {
  size_t value = 0U;
  CHECK(gpu_suite_checked_add_size(2U, 3U, &value));
  CHECK(value == 5U);
  CHECK(!gpu_suite_checked_add_size(SIZE_MAX, 1U, &value));
  CHECK(gpu_suite_checked_mul_size(6U, 7U, &value));
  CHECK(value == 42U);
  CHECK(!gpu_suite_checked_mul_size(SIZE_MAX, 2U, &value));
  CHECK(gpu_suite_checked_mul_size(0U, SIZE_MAX, &value));
  CHECK(value == 0U);
  CHECK(gpu_suite_checked_u64_to_size(1U, &value));
  CHECK(value == 1U);
  CHECK(!gpu_suite_checked_add_size(1U, 1U, NULL));
  CHECK(!gpu_suite_checked_mul_size(1U, 1U, NULL));
  CHECK(!gpu_suite_checked_u64_to_size(1U, NULL));
  return 0;
}
