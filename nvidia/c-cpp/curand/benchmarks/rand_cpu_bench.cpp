#include "gpu_suite/benchmark.hpp"
#include "rand_bench_common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <stdexcept>
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
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  std::size_t bytes = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_bytes(count, sizeof(double), &bytes)) {
    (void)bytes;
    std::fprintf(stderr, "random output size or byte count is unsupported\n");
    gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "random output size exceeds host integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
  bool any_failure = false;
  try {
    std::vector<double> values(count);
    std::mt19937_64 engine(options.seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      reset_engine(engine, options);
      for (int warmup = 0; warmup < options.warmup; ++warmup)
        generate(engine, distribution, values);
      reset_engine(engine, options);
    } else {
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        std::mt19937_64 warmup_engine(options.seed);
        warmup_engine.discard(options.offset);
        generate(warmup_engine, distribution, values);
      }
    }
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
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        reset_engine(engine, options);
        if (gpu_suite_measurement_start(start_timestamp, &start, error,
                                        sizeof(error)) != GPU_SUITE_OK)
          operation_ok = false;
        for (int repeat = 0; repeat < options.repeat && operation_ok; ++repeat)
          generate(engine, distribution, values);
        if (operation_ok && gpu_suite_measurement_end(
                                &end, end_timestamp, error,
                                sizeof(error)) != GPU_SUITE_OK)
          operation_ok = false;
        if (operation_ok)
          elapsed_total = gpu_suite_clock_elapsed(&start, &end);
      } else if (operation_ok) {
        for (int repeat = 0; repeat < options.repeat && operation_ok;
             ++repeat) {
          const int start_status =
              repeat == 0
                  ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                                sizeof(error))
                  : gpu_suite_clock_now(&start, error, sizeof(error));
          if (start_status != GPU_SUITE_OK) {
            operation_ok = false;
            break;
          }
          std::mt19937_64 repeat_engine(options.seed);
          repeat_engine.discard(options.offset);
          generate(repeat_engine, distribution, values);
          const int end_status =
              repeat + 1 == options.repeat
                  ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                              sizeof(error))
                  : gpu_suite_clock_now(&end, error, sizeof(error));
          if (end_status != GPU_SUITE_OK) {
            operation_ok = false;
            break;
          }
          elapsed_total += gpu_suite_clock_elapsed(&start, &end);
        }
      }
      if (operation_ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed_total / options.repeat);
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_curand::set_random_verification(
                result, options, values, "std::mt19937_64", "[0,1)");
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
                 : (constructed ? "cuRAND statistical verification failed"
                                : "verification result construction failed");
        any_failure = any_failure || !pass;
        operation_ok = operation_ok && constructed;
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
