#ifndef GPU_SUITE_THRUST_REDUCE_BENCH_COMMON_HPP
#define GPU_SUITE_THRUST_REDUCE_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <cmath>

namespace gpu_suite_thrust {

inline gpu_suite::VerificationOutcome
set_reduction_verification(gpu_suite_result &result,
                           const gpu_suite_options &options, double value,
                           std::size_t count) {
  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }
  const double expected = static_cast<double>(count);
  if (gpu_suite_verification_add_absolute_relative_threshold(
          &result, "absolute_error", expected, options.abs_tolerance,
          options.rel_tolerance) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  double absolute_error = 0.0;
  if (!gpu_suite_finite_absolute_error(value, expected, &absolute_error)) {
    if (gpu_suite_json_add_null(result.verification_metrics,
                                "absolute_error") != GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    result.verification_primary_metric = "absolute_error";
    result.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }
  if (gpu_suite::json_add_double(result.verification_metrics, "absolute_error",
                                 absolute_error) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  const bool pass = absolute_error <=
                    options.abs_tolerance + options.rel_tolerance * expected;
  result.verification_primary_metric = "absolute_error";
  result.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

} // namespace gpu_suite_thrust

#endif
