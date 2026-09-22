#include "gpu_suite/benchmark.hpp"
#include "reduce_bench_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <numeric>
#include <stdexcept>
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
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  std::size_t bytes = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_bytes(count, sizeof(double), &bytes)) {
    (void)bytes;
    std::fprintf(stderr, "reduction size or byte count is unsupported\n");
    (void)gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "reduction size exceeds host integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
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
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_thrust::set_reduction_verification(result, options,
                                                          value, count);
        const bool constructed =
            outcome != gpu_suite::VerificationOutcome::construction_error;
        const bool pass = outcome == gpu_suite::VerificationOutcome::pass;
        result.attempted = true;
        result.failure_origin =
            pass ? nullptr : (constructed ? "verification" : "benchmark");
        if (!constructed)
          result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message =
            pass ? ""
                 : (constructed ? "transform-reduce verification failed"
                                : "verification result construction failed");
        any_failure = any_failure || !pass;
        ok = ok && constructed;
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
  } catch (const std::length_error &) {
    (void)gpu_suite::emit_unmeasured(
        writer, options, 0, true, "benchmark",
        "host vector size exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    (void)gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                     "host allocation failed");
    any_failure = true;
  }
  if (!writer.close())
    return EXIT_FAILURE;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
