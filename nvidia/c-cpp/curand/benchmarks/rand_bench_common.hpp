#ifndef GPU_SUITE_CURAND_RAND_BENCH_COMMON_HPP
#define GPU_SUITE_CURAND_RAND_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpu_suite_curand {

inline bool set_random_verification(gpu_suite_result &result,
                                    const gpu_suite_options &options,
                                    const std::vector<double> &values,
                                    const char *algorithm,
                                    const char *backend_interval) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (result.verification_metrics == nullptr ||
      result.verification_thresholds == nullptr ||
      gpu_suite::json_add_string(result.parameters, "generator_algorithm",
                                 algorithm) != GPU_SUITE_OK ||
      gpu_suite::json_add_string(result.parameters, "distribution_interval",
                                 backend_interval) != GPU_SUITE_OK ||
      gpu_suite_json_add_int(result.parameters, "verification_sample_count",
                             static_cast<int64_t>(values.size())) !=
          GPU_SUITE_OK) {
    result.verification_status = "failure";
    return false;
  }
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return true;
  }

  double observed_min = 1.0;
  double observed_max = 0.0;
  double sum = 0.0;
  double moment_sum = 0.0;
  for (double value : values) {
    observed_min = std::min(observed_min, value);
    observed_max = std::max(observed_max, value);
    sum += value;
    const double centered = value - 0.5;
    moment_sum += centered * centered;
  }
  const double count = static_cast<double>(values.size());
  const double sample_mean = sum / count;
  const double second_moment = moment_sum / count;
  const double mean_bound =
      options.sigma_multiplier * std::sqrt(1.0 / (12.0 * count));
  const double moment_bound =
      options.sigma_multiplier * std::sqrt(1.0 / (180.0 * count));
  if (!std::isfinite(sample_mean) || !std::isfinite(second_moment) ||
      !std::isfinite(observed_min) || !std::isfinite(observed_max)) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "nonfinite";
    return false;
  }

  bool encoded =
      gpu_suite::json_add_double(result.verification_metrics, "observed_min",
                                 observed_min) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics, "observed_max",
                                 observed_max) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics, "sample_mean",
                                 sample_mean) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics,
                                 "second_central_moment_about_half",
                                 second_moment) == GPU_SUITE_OK;

  gpu_suite_json_value *range = gpu_suite_json_object();
  gpu_suite_json_value *mean = gpu_suite_json_object();
  gpu_suite_json_value *moment = gpu_suite_json_object();
  encoded = encoded && range != nullptr && mean != nullptr && moment != nullptr;
  if (encoded) {
    encoded =
        gpu_suite::json_add_string(range, "method", "inclusive-range") ==
            GPU_SUITE_OK &&
        gpu_suite::json_add_double(range, "lower_bound", 0.0) == GPU_SUITE_OK &&
        gpu_suite::json_add_double(range, "upper_bound", 1.0) == GPU_SUITE_OK &&
        gpu_suite::json_add_string(range, "backend_interval",
                                   backend_interval) == GPU_SUITE_OK &&
        gpu_suite::json_add_string(
            mean, "method", "uniform-mean-sigma-bound") == GPU_SUITE_OK &&
        gpu_suite::json_add_double(mean, "expected_mean",
                                   options.expected_mean) == GPU_SUITE_OK &&
        gpu_suite::json_add_double(mean, "sigma_multiplier",
                                   options.sigma_multiplier) == GPU_SUITE_OK &&
        gpu_suite::json_add_double(mean, "absolute_bound", mean_bound) ==
            GPU_SUITE_OK &&
        gpu_suite::json_add_string(
            moment, "method", "uniform-second-central-moment-sigma-bound") ==
            GPU_SUITE_OK &&
        gpu_suite::json_add_double(moment, "expected_second_central_moment",
                                   options.expected_second_central_moment) ==
            GPU_SUITE_OK &&
        gpu_suite::json_add_double(moment, "sigma_multiplier",
                                   options.sigma_multiplier) == GPU_SUITE_OK &&
        gpu_suite::json_add_double(moment, "absolute_bound", moment_bound) ==
            GPU_SUITE_OK;
  }
  if (encoded) {
    encoded =
        gpu_suite_json_object_set(result.verification_thresholds,
                                  "observed_range", range) == GPU_SUITE_OK &&
        gpu_suite_json_object_set(result.verification_thresholds, "sample_mean",
                                  mean) == GPU_SUITE_OK &&
        gpu_suite_json_object_set(result.verification_thresholds,
                                  "second_central_moment_about_half",
                                  moment) == GPU_SUITE_OK;
  }
  if (!encoded) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "failure";
    return false;
  }

  const bool range_ok = observed_min >= 0.0 && observed_max <= 1.0;
  const bool mean_ok =
      std::fabs(sample_mean - options.expected_mean) <= mean_bound;
  const bool moment_ok =
      std::fabs(second_moment - options.expected_second_central_moment) <=
      moment_bound;
  const bool pass = range_ok && mean_ok && moment_ok;
  result.verification_primary_metric = "sample_mean";
  result.verification_status = pass ? "pass" : "failure";
  return pass;
}

} // namespace gpu_suite_curand

#endif
