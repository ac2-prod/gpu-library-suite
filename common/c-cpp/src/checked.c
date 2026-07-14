#include "gpu_suite/checked.h"

#include <limits.h>
#include <math.h>
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

bool gpu_suite_checked_mul3_size(size_t first, size_t second, size_t third,
                                 size_t *result) {
  size_t product;
  return gpu_suite_checked_mul_size(first, second, &product) &&
         gpu_suite_checked_mul_size(product, third, result);
}

bool gpu_suite_checked_bytes(size_t element_count, size_t element_size,
                             size_t *result) {
  return gpu_suite_checked_mul_size(element_count, element_size, result);
}

bool gpu_suite_checked_u64_to_int(uint64_t value, int *result) {
  if (result == NULL || value > (uint64_t)INT_MAX) {
    return false;
  }
  *result = (int)value;
  return true;
}

bool gpu_suite_checked_u64_to_size(uint64_t value, size_t *result) {
  if (result == NULL || value > (uint64_t)SIZE_MAX) {
    return false;
  }
  *result = (size_t)value;
  return true;
}

bool gpu_suite_checked_poisson2d_dimensions(uint64_t nx_value,
                                            uint64_t ny_value, int *nx,
                                            int *ny, int *point_count,
                                            int *nonzero_count,
                                            size_t *row_offset_count) {
  size_t nx_size;
  size_t ny_size;
  size_t points;
  size_t five_points;
  size_t twice_nx;
  size_t twice_ny;
  size_t boundary;
  size_t nonzeros;
  if (nx == NULL || ny == NULL || point_count == NULL ||
      nonzero_count == NULL || row_offset_count == NULL || nx_value == 0U ||
      ny_value == 0U || !gpu_suite_checked_u64_to_int(nx_value, nx) ||
      !gpu_suite_checked_u64_to_int(ny_value, ny) ||
      !gpu_suite_checked_u64_to_size(nx_value, &nx_size) ||
      !gpu_suite_checked_u64_to_size(ny_value, &ny_size) ||
      !gpu_suite_checked_mul_size(nx_size, ny_size, &points) ||
      points > (size_t)INT_MAX ||
      !gpu_suite_checked_mul_size(points, 5U, &five_points) ||
      !gpu_suite_checked_mul_size(nx_size, 2U, &twice_nx) ||
      !gpu_suite_checked_mul_size(ny_size, 2U, &twice_ny) ||
      !gpu_suite_checked_add_size(twice_nx, twice_ny, &boundary) ||
      boundary > five_points) {
    return false;
  }
  nonzeros = five_points - boundary;
  if (nonzeros > (size_t)INT_MAX ||
      !gpu_suite_checked_add_size(points, 1U, row_offset_count)) {
    return false;
  }
  *point_count = (int)points;
  *nonzero_count = (int)nonzeros;
  return true;
}

bool gpu_suite_finite_absolute_error(double observed, double expected,
                                     double *error) {
  double difference;
  if (error == NULL || !isfinite(observed) || !isfinite(expected)) {
    return false;
  }
  difference = fabs(observed - expected);
  if (!isfinite(difference)) {
    return false;
  }
  *error = difference;
  return true;
}

bool gpu_suite_finite_max_update(double candidate, double *maximum) {
  if (maximum == NULL || !isfinite(candidate) || !isfinite(*maximum)) {
    return false;
  }
  if (candidate > *maximum) {
    *maximum = candidate;
  }
  return true;
}
