#include "gpu_suite/benchmark.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <numeric>
#include <string>
#include <vector>

static double reduce_values(const std::vector<double> &values) {
#ifdef GPU_SUITE_USE_OPENMP
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (std::size_t index = 0; index < values.size(); ++index) {
    sum += values[index] * values[index];
  }
  return sum;
#else
  return std::transform_reduce(values.begin(), values.end(), 0.0,
                               std::plus<double>(),
                               [](double value) { return value * value; });
#endif
}

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_THRUST);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK ||
      std::string(options.cpu_backend) != GPU_SUITE_COMPILED_CPU_BACKEND) {
    std::fprintf(stderr, "%s%s\n", error,
                 parsed == GPU_SUITE_PARSE_OK
                     ? "CPU backend does not match this binary"
                     : "");
    return EXIT_FAILURE;
  }
  std::size_t count = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count))
    return EXIT_FAILURE;
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;
  try {
    const std::vector<double> values(count, 1.0);
    for (int warmup = 0; warmup < options.warmup; ++warmup)
      (void)reduce_values(values);
    for (int trial = 0; trial < options.trials; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial))
        return EXIT_FAILURE;
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec start, end;
      double value = 0.0;
      double elapsed = 0.0;
      bool ok = true;
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok = gpu_suite_measurement_start(start_timestamp, &start, error,
                                         sizeof(error)) == GPU_SUITE_OK;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat)
          value = reduce_values(values);
        ok = ok && gpu_suite_measurement_end(&end, end_timestamp, error,
                                              sizeof(error)) == GPU_SUITE_OK;
        if (ok)
          elapsed = gpu_suite_clock_elapsed(&start, &end);
      } else if (ok) {
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          ok = (repeat == 0
                    ? gpu_suite_measurement_start(start_timestamp, &start,
                                                  error, sizeof(error))
                    : gpu_suite_clock_now(&start, error, sizeof(error))) ==
               GPU_SUITE_OK;
          if (!ok)
            break;
          value = reduce_values(values);
          ok = (repeat + 1 == options.repeat
                    ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                                sizeof(error))
                    : gpu_suite_clock_now(&end, error, sizeof(error))) ==
               GPU_SUITE_OK;
          if (ok)
            elapsed += gpu_suite_clock_elapsed(&start, &end);
        }
      }
      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed / options.repeat);
        gpu_suite_json_free(result.verification_metrics);
        gpu_suite_json_free(result.verification_thresholds);
        result.verification_metrics = gpu_suite_json_object();
        result.verification_thresholds = gpu_suite_json_object();
        const double expected = static_cast<double>(count);
        const double absolute_error = std::fabs(value - expected);
        gpu_suite::json_add_double(result.verification_metrics,
                                   "absolute_error", absolute_error);
        gpu_suite_json_value *threshold = gpu_suite_json_object();
        gpu_suite::json_add_string(threshold, "method",
                                   "absolute-plus-relative");
        gpu_suite::json_add_double(threshold, "reference_scale", expected);
        gpu_suite::json_add_double(threshold, "abs_tolerance",
                                   options.abs_tolerance);
        gpu_suite::json_add_double(threshold, "rel_tolerance",
                                   options.rel_tolerance);
        gpu_suite_json_object_set(result.verification_thresholds,
                                  "absolute_error", threshold);
        result.verification_primary_metric = "absolute_error";
        const bool pass =
            !options.verify ||
            absolute_error <=
                options.abs_tolerance + options.rel_tolerance * expected;
        result.verification_status =
            options.verify ? (pass ? "pass" : "failure") : "skipped";
        result.attempted = true;
        result.failure_origin = pass ? nullptr : "verification";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message = pass ? "" : "transform-reduce verification failed";
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
      if (!ok) {
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
