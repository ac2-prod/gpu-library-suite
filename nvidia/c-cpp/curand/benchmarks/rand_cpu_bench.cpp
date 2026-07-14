#include "gpu_suite/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <vector>

namespace {

void reset_engine(std::mt19937_64 &engine, const gpu_suite_options &options) {
  engine.seed(options.seed);
  engine.discard(options.offset);
}

void generate(std::mt19937_64 &engine,
              std::uniform_real_distribution<double> &distribution,
              std::vector<double> &values) {
  for (double &value : values) {
    value = distribution(engine);
  }
}

int set_verification(gpu_suite_result &result, const gpu_suite_options &options,
                     const std::vector<double> &values) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (result.verification_metrics == nullptr ||
      result.verification_thresholds == nullptr) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (gpu_suite::json_add_string(result.parameters, "cpu_engine",
                                 "std::mt19937_64") != GPU_SUITE_OK ||
      gpu_suite::json_add_string(result.parameters, "distribution_interval",
                                 "[0,1)") != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (gpu_suite_json_add_int(result.parameters, "verification_sample_count",
                             static_cast<int64_t>(values.size())) !=
      GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return GPU_SUITE_OK;
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
  if (!std::isfinite(sample_mean) || !std::isfinite(second_moment)) {
    result.verification_status = "nonfinite";
    return GPU_SUITE_ERROR_FORMAT;
  }
  if (gpu_suite::json_add_double(result.verification_metrics, "observed_min",
                                 observed_min) != GPU_SUITE_OK ||
      gpu_suite::json_add_double(result.verification_metrics, "observed_max",
                                 observed_max) != GPU_SUITE_OK ||
      gpu_suite::json_add_double(result.verification_metrics, "sample_mean",
                                 sample_mean) != GPU_SUITE_OK ||
      gpu_suite::json_add_double(result.verification_metrics,
                                 "second_central_moment_about_half",
                                 second_moment) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  gpu_suite_json_value *range = gpu_suite_json_object();
  gpu_suite_json_value *mean = gpu_suite_json_object();
  gpu_suite_json_value *moment = gpu_suite_json_object();
  if (range == nullptr || mean == nullptr || moment == nullptr) {
    gpu_suite_json_free(range);
    gpu_suite_json_free(mean);
    gpu_suite_json_free(moment);
    return GPU_SUITE_ERROR_NOMEM;
  }
  int status = gpu_suite::json_add_string(range, "method", "inclusive-range");
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(range, "lower_bound", 0.0);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(range, "upper_bound", 1.0);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_string(range, "backend_interval", "[0,1)");
  if (status == GPU_SUITE_OK)
    status =
        gpu_suite::json_add_string(mean, "method", "uniform-mean-sigma-bound");
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(mean, "expected_mean",
                                        options.expected_mean);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(mean, "sigma_multiplier",
                                        options.sigma_multiplier);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(mean, "absolute_bound", mean_bound);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_string(
        moment, "method", "uniform-second-central-moment-sigma-bound");
  if (status == GPU_SUITE_OK)
    status =
        gpu_suite::json_add_double(moment, "expected_second_central_moment",
                                   options.expected_second_central_moment);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(moment, "sigma_multiplier",
                                        options.sigma_multiplier);
  if (status == GPU_SUITE_OK)
    status = gpu_suite::json_add_double(moment, "absolute_bound", moment_bound);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_json_object_set(result.verification_thresholds,
                                       "observed_range", range);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(range);
    gpu_suite_json_free(mean);
    gpu_suite_json_free(moment);
    return status;
  }
  if (gpu_suite_json_object_set(result.verification_thresholds, "sample_mean",
                                mean) != GPU_SUITE_OK) {
    gpu_suite_json_free(mean);
    gpu_suite_json_free(moment);
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (gpu_suite_json_object_set(result.verification_thresholds,
                                "second_central_moment_about_half",
                                moment) != GPU_SUITE_OK) {
    gpu_suite_json_free(moment);
    return GPU_SUITE_ERROR_NOMEM;
  }
  result.verification_primary_metric = "sample_mean";
  const bool range_ok = observed_min >= 0.0 && observed_max <= 1.0;
  const bool mean_ok =
      std::fabs(sample_mean - options.expected_mean) <= mean_bound;
  const bool moment_ok =
      std::fabs(second_moment - options.expected_second_central_moment) <=
      moment_bound;
  result.verification_status =
      range_ok && mean_ok && moment_ok ? "pass" : "failure";
  return GPU_SUITE_OK;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CURAND,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CURAND);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK ||
      std::string(options.cpu_backend) != "cpu-std-random-serial") {
    std::fprintf(stderr, "%s%s\n", error,
                 parsed == GPU_SUITE_PARSE_OK ? "unsupported CPU backend" : "");
    return EXIT_FAILURE;
  }
  std::size_t count = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count)) {
    std::fprintf(stderr, "size does not fit host representation\n");
    return EXIT_FAILURE;
  }
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;
  try {
    std::vector<double> values(count);
    std::mt19937_64 engine(options.seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    reset_engine(engine, options);
    for (int warmup = 0; warmup < options.warmup; ++warmup)
      generate(engine, distribution, values);
    reset_engine(engine, options);
    for (int trial = 0; trial < options.trials; ++trial) {
      gpu_suite_result result;
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec start;
      struct timespec end;
      double elapsed_total = 0.0;
      bool operation_ok = true;
      if (!gpu_suite::initialize_result(result, options, trial))
        return EXIT_FAILURE;
      if (gpu_suite_utc_timestamp(start_timestamp, error, sizeof(error)) !=
          GPU_SUITE_OK)
        operation_ok = false;
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE && operation_ok) {
        reset_engine(engine, options);
        if (gpu_suite_clock_now(&start, error, sizeof(error)) != GPU_SUITE_OK)
          operation_ok = false;
        for (int repeat = 0; repeat < options.repeat && operation_ok; ++repeat)
          generate(engine, distribution, values);
        if (operation_ok &&
            gpu_suite_clock_now(&end, error, sizeof(error)) != GPU_SUITE_OK)
          operation_ok = false;
        if (operation_ok)
          elapsed_total = gpu_suite_clock_elapsed(&start, &end);
      } else if (operation_ok) {
        for (int repeat = 0; repeat < options.repeat && operation_ok;
             ++repeat) {
          if (gpu_suite_clock_now(&start, error, sizeof(error)) !=
              GPU_SUITE_OK) {
            operation_ok = false;
            break;
          }
          std::mt19937_64 repeat_engine(options.seed);
          repeat_engine.discard(options.offset);
          generate(repeat_engine, distribution, values);
          if (gpu_suite_clock_now(&end, error, sizeof(error)) != GPU_SUITE_OK) {
            operation_ok = false;
            break;
          }
          elapsed_total += gpu_suite_clock_elapsed(&start, &end);
        }
      }
      if (operation_ok &&
          gpu_suite_utc_timestamp(end_timestamp, error, sizeof(error)) !=
              GPU_SUITE_OK)
        operation_ok = false;
      if (operation_ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed_total / options.repeat);
        const int verified = set_verification(result, options, values);
        const bool pass =
            verified == GPU_SUITE_OK &&
            (std::string(result.verification_status) == "pass" ||
             std::string(result.verification_status) == "skipped");
        result.attempted = true;
        result.failure_origin = pass ? nullptr : "verification";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message = pass ? "" : "cuRAND statistical verification failed";
        any_failure = any_failure || !pass;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "timing failed";
        any_failure = true;
      }
      if (!writer.write(result)) {
        gpu_suite_result_destroy(&result);
        return EXIT_FAILURE;
      }
      gpu_suite_result_destroy(&result);
      if (!operation_ok) {
        (void)gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                         "prior-failure",
                                         "prior timing failure");
        break;
      }
    }
  } catch (const std::bad_alloc &) {
    (void)gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                     "host allocation failed");
    any_failure = true;
  }
  if (!writer.close())
    return EXIT_FAILURE;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
