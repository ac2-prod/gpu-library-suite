#ifndef GPU_SUITE_CURAND_RAND_BENCH_COMMON_HPP
#define GPU_SUITE_CURAND_RAND_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace gpu_suite_curand {

inline int add_random_thresholds(gpu_suite_result &result,
                                 const gpu_suite_options &options,
                                 const char *backend_interval,
                                 double mean_bound, double moment_bound) {
  gpu_suite_json_value *range = gpu_suite_json_object();
  gpu_suite_json_value *mean = gpu_suite_json_object();
  gpu_suite_json_value *moment = gpu_suite_json_object();
  int status = range != nullptr && mean != nullptr && moment != nullptr
                   ? GPU_SUITE_OK
                   : GPU_SUITE_ERROR_NOMEM;
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_string(range, "method", "inclusive-range");
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(range, "lower_bound", 0.0);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(range, "upper_bound", 1.0);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_string(range, "backend_interval",
                                       backend_interval);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_string(mean, "method",
                                       "uniform-mean-sigma-bound");
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(mean, "expected_mean",
                                       options.expected_mean);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(mean, "sigma_multiplier",
                                       options.sigma_multiplier);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(mean, "absolute_bound", mean_bound);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_string(
        moment, "method", "uniform-second-central-moment-sigma-bound");
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(
        moment, "expected_second_central_moment",
        options.expected_second_central_moment);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(moment, "sigma_multiplier",
                                       options.sigma_multiplier);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_add_double(moment, "absolute_bound", moment_bound);
  if (status == GPU_SUITE_OK) {
    status = gpu_suite_json_object_set(result.verification_thresholds,
                                       "observed_range", range);
    if (status == GPU_SUITE_OK)
      range = nullptr;
  }
  if (status == GPU_SUITE_OK) {
    status = gpu_suite_json_object_set(result.verification_thresholds,
                                       "sample_mean", mean);
    if (status == GPU_SUITE_OK)
      mean = nullptr;
  }
  if (status == GPU_SUITE_OK) {
    status = gpu_suite_json_object_set(
        result.verification_thresholds,
        "second_central_moment_about_half", moment);
    if (status == GPU_SUITE_OK)
      moment = nullptr;
  }
  gpu_suite_json_free(range);
  gpu_suite_json_free(mean);
  gpu_suite_json_free(moment);
  return status;
}

inline gpu_suite::VerificationOutcome
set_random_verification(gpu_suite_result &result,
                        const gpu_suite_options &options,
                        const std::vector<double> &values,
                        const char *algorithm, const char *backend_interval) {
  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK ||
      gpu_suite::json_add_string(
          result.parameters,
          options.implementation == GPU_SUITE_IMPLEMENTATION_CPU
              ? "cpu_engine"
              : "generator_algorithm",
          algorithm) != GPU_SUITE_OK ||
      gpu_suite::json_add_string(result.parameters, "distribution_interval",
                                 backend_interval) != GPU_SUITE_OK ||
      values.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      gpu_suite_json_add_int(result.parameters, "verification_sample_count",
                             static_cast<int64_t>(values.size())) !=
          GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }

  double observed_min = 1.0;
  double observed_max = 0.0;
  double sum = 0.0;
  double moment_sum = 0.0;
  bool finite = true;
  for (double value : values) {
    if (!std::isfinite(value)) {
      finite = false;
      continue;
    }
    if (value < observed_min)
      observed_min = value;
    if (value > observed_max)
      observed_max = value;
    sum += value;
    const double centered = value - 0.5;
    moment_sum += centered * centered;
    if (!std::isfinite(sum) || !std::isfinite(centered) ||
        !std::isfinite(moment_sum))
      finite = false;
  }
  const double count = static_cast<double>(values.size());
  const double sample_mean = sum / count;
  const double second_moment = moment_sum / count;
  const double mean_bound =
      options.sigma_multiplier * std::sqrt(1.0 / (12.0 * count));
  const double moment_bound =
      options.sigma_multiplier * std::sqrt(1.0 / (180.0 * count));
  if (add_random_thresholds(result, options, backend_interval, mean_bound,
                            moment_bound) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  finite = finite && std::isfinite(sample_mean) &&
           std::isfinite(second_moment) && std::isfinite(observed_min) &&
           std::isfinite(observed_max);
  if (!finite) {
    if (gpu_suite_json_add_null(result.verification_metrics, "observed_min") !=
            GPU_SUITE_OK ||
        gpu_suite_json_add_null(result.verification_metrics, "observed_max") !=
            GPU_SUITE_OK ||
        gpu_suite_json_add_null(result.verification_metrics, "sample_mean") !=
            GPU_SUITE_OK ||
        gpu_suite_json_add_null(result.verification_metrics,
                                "second_central_moment_about_half") !=
            GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    result.verification_primary_metric = "sample_mean";
    result.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }

  const bool encoded =
      gpu_suite::json_add_double(result.verification_metrics, "observed_min",
                                 observed_min) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics, "observed_max",
                                 observed_max) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics, "sample_mean",
                                 sample_mean) == GPU_SUITE_OK &&
      gpu_suite::json_add_double(result.verification_metrics,
                                 "second_central_moment_about_half",
                                 second_moment) == GPU_SUITE_OK;
  if (!encoded)
    return gpu_suite::VerificationOutcome::construction_error;

  const bool range_ok = observed_min >= 0.0 && observed_max <= 1.0;
  const bool mean_ok =
      std::fabs(sample_mean - options.expected_mean) <= mean_bound;
  const bool moment_ok =
      std::fabs(second_moment - options.expected_second_central_moment) <=
      moment_bound;
  const bool pass = range_ok && mean_ok && moment_ok;
  result.verification_primary_metric = "sample_mean";
  result.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

} // namespace gpu_suite_curand

#endif
