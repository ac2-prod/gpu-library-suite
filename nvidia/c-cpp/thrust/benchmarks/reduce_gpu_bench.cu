#include "gpu_suite/benchmark.hpp"
#include "reduce_bench_common.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_success(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}

struct SquareValue {
  __host__ __device__ double operator()(double value) const {
    return value * value;
  }
};

double transform_reduce(const thrust::device_vector<double> &values) {
  return thrust::transform_reduce(values.begin(), values.end(), SquareValue{},
                                  0.0, thrust::plus<double>());
}

bool run_end_to_end_pipeline(
    const gpu_suite_options &options, const std::vector<double> &host_values,
    int repeat_index, bool measure,
    char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY], char *error,
    std::size_t error_size, double *elapsed_total, double &reduced_value) {
  struct timespec start;
  struct timespec end;
  bool ok = !measure ||
            (repeat_index == 0
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK;
  std::unique_ptr<thrust::device_vector<double>> current;
  if (ok) {
    current.reset(new thrust::device_vector<double>(host_values));
    reduced_value = transform_reduce(*current);
    ok = cuda_success(cudaDeviceSynchronize(), "end-to-end synchronize");
  }
  if (ok && measure) {
    ok = (repeat_index + 1 == options.repeat
              ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                          error_size)
              : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
    if (ok)
      *elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  current.reset();
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_THRUST);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
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
    gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "reduction size exceeds host integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
  bool any_failure = false;
  try {
    const std::vector<double> host_values(count, 1.0);
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }
    std::unique_ptr<thrust::device_vector<double>> persistent;
    double warmup_value = 0.0;
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      persistent.reset(new thrust::device_vector<double>(host_values));
      for (int warmup = 0; warmup < options.warmup; ++warmup)
        warmup_value = transform_reduce(*persistent);
      if (!cuda_success(cudaDeviceSynchronize(), "warmup synchronize")) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "Thrust warmup failed");
        writer.close();
        return EXIT_FAILURE;
      }
    } else {
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        if (!run_end_to_end_pipeline(
                options, host_values, 0, false, nullptr, nullptr, error,
                sizeof(error), nullptr, warmup_value)) {
          gpu_suite::emit_unmeasured(
              writer, options, 0, true, "benchmark",
              "Thrust end-to-end warmup failed");
          writer.close();
          return EXIT_FAILURE;
        }
      }
    }

    for (int trial = 0; trial < options.trials; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial)) {
        any_failure = true;
        break;
      }
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec start;
      struct timespec end;
      double reduced_value = 0.0;
      double elapsed_total = 0.0;
      bool ok = true;
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok = cuda_success(cudaDeviceSynchronize(),
                          "pre-timing synchronize") &&
             gpu_suite_measurement_start(start_timestamp, &start, error,
                                          sizeof(error)) == GPU_SUITE_OK;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          reduced_value = transform_reduce(*persistent);
        }
        ok = ok && cuda_success(cudaDeviceSynchronize(),
                                "post-timing synchronize") &&
             gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error)) == GPU_SUITE_OK;
        if (ok)
          elapsed_total = gpu_suite_clock_elapsed(&start, &end);
      } else {
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          ok = run_end_to_end_pipeline(
              options, host_values, repeat, true, start_timestamp,
              end_timestamp, error, sizeof(error), &elapsed_total,
              reduced_value);
        }
      }
      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed_total / options.repeat);
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_thrust::set_reduction_verification(
                result, options, reduced_value, count);
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
        result.message = pass ? ""
                              : (constructed
                                     ? "Thrust reduction verification failed"
                                     : "verification result construction failed");
        any_failure = any_failure || !pass;
        ok = ok && constructed;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "Thrust reduction pipeline failed";
        any_failure = true;
      }
      if (!writer.write(result)) {
        gpu_suite_result_destroy(&result);
        return EXIT_FAILURE;
      }
      gpu_suite_result_destroy(&result);
      if (!ok) {
        gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                   "prior-failure", "prior Thrust failure");
        break;
      }
    }
  } catch (const std::length_error &) {
    std::fprintf(stderr, "Thrust device vector length is unsupported\n");
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "device vector length exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "Thrust host/device allocation failed\n");
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host or device allocation failed");
    any_failure = true;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "Thrust exception: %s\n", exception.what());
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "Thrust allocation/execution failed");
    any_failure = true;
  }
  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
