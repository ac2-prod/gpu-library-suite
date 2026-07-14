#ifndef GPU_SUITE_THRUST_REDUCE_BENCH_COMMON_HPP
#define GPU_SUITE_THRUST_REDUCE_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <cmath>

namespace gpu_suite_thrust {

inline bool set_reduction_verification(gpu_suite_result &result,
                                       const gpu_suite_options &options,
                                       double value, std::size_t count) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return true;
  }
  const double expected = static_cast<double>(count);
  const double absolute_error = std::fabs(value - expected);
  if (!std::isfinite(absolute_error)) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "nonfinite";
    return false;
  }
  gpu_suite::json_add_double(result.verification_metrics, "absolute_error",
                             absolute_error);
  gpu_suite_json_value *threshold = gpu_suite_json_object();
  gpu_suite::json_add_string(threshold, "method", "absolute-plus-relative");
  gpu_suite::json_add_double(threshold, "reference_scale", expected);
  gpu_suite::json_add_double(threshold, "abs_tolerance", options.abs_tolerance);
  gpu_suite::json_add_double(threshold, "rel_tolerance", options.rel_tolerance);
  gpu_suite_json_object_set(result.verification_thresholds, "absolute_error",
                            threshold);
  const bool pass = absolute_error <=
                    options.abs_tolerance + options.rel_tolerance * expected;
  result.verification_primary_metric = "absolute_error";
  result.verification_status = pass ? "pass" : "failure";
  return pass;
}

} // namespace gpu_suite_thrust

#endif
