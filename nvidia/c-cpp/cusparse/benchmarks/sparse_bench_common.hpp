#ifndef GPU_SUITE_CUSPARSE_SPARSE_BENCH_COMMON_HPP
#define GPU_SUITE_CUSPARSE_SPARSE_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpu_suite_cusparse {

inline gpu_suite::VerificationOutcome
set_spmv_verification(gpu_suite_result &result,
                      const gpu_suite_options &options,
                      const std::vector<double> &values, int nx, int ny,
                      int updates) {
  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }

  double maximum = 0.0;
  double reference_scale = 0.0;
  bool finite = true;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int neighbor_count =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      const double product = 4.0 - neighbor_count;
      double expected = 1.0;
      for (int update = 0; update < updates; ++update) {
        expected = options.alpha * product + options.beta * expected;
        if (!std::isfinite(expected))
          finite = false;
      }
      double error = 0.0;
      if (!gpu_suite_finite_absolute_error(
              values[static_cast<std::size_t>(iy) *
                         static_cast<std::size_t>(nx) +
                     static_cast<std::size_t>(ix)],
              expected,
              &error) ||
          !gpu_suite_finite_max_update(error, &maximum))
        finite = false;
      if (std::isfinite(expected) &&
          !gpu_suite_finite_max_update(std::fabs(expected), &reference_scale))
        finite = false;
    }
  }
  if (!std::isfinite(reference_scale))
    return gpu_suite::VerificationOutcome::construction_error;
  if (gpu_suite_verification_add_absolute_relative_threshold(
          &result, "max_abs_error", reference_scale, options.abs_tolerance,
          options.rel_tolerance) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!finite) {
    if (gpu_suite_json_add_null(result.verification_metrics,
                                "max_abs_error") != GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    result.verification_primary_metric = "max_abs_error";
    result.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }
  if (gpu_suite::json_add_double(result.verification_metrics, "max_abs_error",
                                 maximum) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  const bool pass = maximum <= options.abs_tolerance +
                                   options.rel_tolerance * reference_scale;
  result.verification_primary_metric = "max_abs_error";
  result.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

} // namespace gpu_suite_cusparse

#endif
