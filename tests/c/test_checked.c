#include "gpu_suite/checked.h"

#include "test_support.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>

int main(void) {
  size_t value = 0U;
  int int_value = 0;
  int nx = 0;
  int ny = 0;
  int points = 0;
  int nonzeros = 0;
  double real_value = 0.0;
  CHECK(gpu_suite_checked_add_size(2U, 3U, &value));
  CHECK(value == 5U);
  CHECK(!gpu_suite_checked_add_size(SIZE_MAX, 1U, &value));
  CHECK(gpu_suite_checked_mul_size(6U, 7U, &value));
  CHECK(value == 42U);
  CHECK(!gpu_suite_checked_mul_size(SIZE_MAX, 2U, &value));
  CHECK(gpu_suite_checked_mul_size(0U, SIZE_MAX, &value));
  CHECK(value == 0U);
  CHECK(gpu_suite_checked_mul3_size(2U, 3U, 4U, &value));
  CHECK(value == 24U);
  CHECK(!gpu_suite_checked_mul3_size(SIZE_MAX, 2U, 1U, &value));
  CHECK(!gpu_suite_checked_mul3_size(1U, SIZE_MAX, 2U, &value));
  CHECK(gpu_suite_checked_bytes(4U, sizeof(double), &value));
  CHECK(value == 4U * sizeof(double));
  CHECK(!gpu_suite_checked_bytes(SIZE_MAX, 2U, &value));
  CHECK(gpu_suite_checked_u64_to_int((uint64_t)INT_MAX, &int_value));
  CHECK(int_value == INT_MAX);
  CHECK(!gpu_suite_checked_u64_to_int((uint64_t)INT_MAX + 1U, &int_value));
  CHECK(gpu_suite_checked_u64_to_size(1U, &value));
  CHECK(value == 1U);
  CHECK(gpu_suite_checked_poisson2d_dimensions(
      2U, 2U, &nx, &ny, &points, &nonzeros, &value));
  CHECK(nx == 2 && ny == 2 && points == 4 && nonzeros == 12 && value == 5U);
  CHECK(gpu_suite_checked_poisson2d_dimensions(
      1U, 1U, &nx, &ny, &points, &nonzeros, &value));
  CHECK(points == 1 && nonzeros == 1 && value == 2U);
  CHECK(!gpu_suite_checked_poisson2d_dimensions(
      (uint64_t)INT_MAX, 2U, &nx, &ny, &points, &nonzeros, &value));
  CHECK(!gpu_suite_checked_poisson2d_dimensions(
      0U, 1U, &nx, &ny, &points, &nonzeros, &value));
  CHECK(gpu_suite_finite_absolute_error(3.0, 1.0, &real_value));
  CHECK(real_value == 2.0);
  CHECK(!gpu_suite_finite_absolute_error(NAN, 1.0, &real_value));
  CHECK(!gpu_suite_finite_absolute_error(INFINITY, 1.0, &real_value));
  CHECK(!gpu_suite_finite_absolute_error(-INFINITY, 1.0, &real_value));
  real_value = 1.0;
  CHECK(gpu_suite_finite_max_update(2.0, &real_value));
  CHECK(real_value == 2.0);
  CHECK(!gpu_suite_finite_max_update(NAN, &real_value));
  CHECK(!gpu_suite_finite_max_update(INFINITY, &real_value));
  CHECK(!gpu_suite_finite_max_update(-INFINITY, &real_value));
  CHECK(!gpu_suite_checked_add_size(1U, 1U, NULL));
  CHECK(!gpu_suite_checked_mul_size(1U, 1U, NULL));
  CHECK(!gpu_suite_checked_mul3_size(1U, 1U, 1U, NULL));
  CHECK(!gpu_suite_checked_bytes(1U, 1U, NULL));
  CHECK(!gpu_suite_checked_u64_to_int(1U, NULL));
  CHECK(!gpu_suite_checked_u64_to_size(1U, NULL));
  CHECK(!gpu_suite_checked_poisson2d_dimensions(
      1U, 1U, NULL, &ny, &points, &nonzeros, &value));
  CHECK(!gpu_suite_finite_absolute_error(1.0, 1.0, NULL));
  CHECK(!gpu_suite_finite_max_update(1.0, NULL));
  return 0;
}
