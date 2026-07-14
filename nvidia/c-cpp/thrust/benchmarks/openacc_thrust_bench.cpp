#include "gpu_suite/benchmark.hpp"
#include "reduce_bench_common.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
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

double transform_reduce(double *values, std::size_t count) {
  double result = 0.0;
#pragma acc host_data use_device(values)
  {
    thrust::device_ptr<double> begin = thrust::device_pointer_cast(values);
    result = thrust::transform_reduce(begin, begin + count, SquareValue{}, 0.0,
                                      thrust::plus<double>());
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
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
  if (!gpu_suite_checked_u64_to_size(options.size, &count)) {
    std::fprintf(stderr, "size does not fit host representation\n");
    return EXIT_FAILURE;
  }

  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;
  try {
    std::vector<double> values(count, 1.0);
    double *values_ptr = values.data();
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
#pragma acc data copyin(values_ptr[0 : count])
      {
        for (int warmup = 0; warmup < options.warmup; ++warmup) {
          (void)transform_reduce(values_ptr, count);
        }
        bool setup_ok =
            cuda_success(cudaDeviceSynchronize(), "warmup synchronize");
        for (int trial = 0; trial < options.trials && setup_ok; ++trial) {
          gpu_suite_result result;
          if (!gpu_suite::initialize_result(result, options, trial)) {
            setup_ok = false;
            any_failure = true;
            break;
          }
          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          struct timespec start;
          struct timespec end;
          double reduced_value = 0.0;
          bool ok =
              cuda_success(cudaDeviceSynchronize(),
                           "pre-timing synchronize") &&
              gpu_suite_measurement_start(start_timestamp, &start, error,
                                           sizeof(error)) == GPU_SUITE_OK;
          for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
            reduced_value = transform_reduce(values_ptr, count);
          }
          ok = ok && cuda_success(cudaDeviceSynchronize(),
                                  "post-timing synchronize") &&
               gpu_suite_measurement_end(&end, end_timestamp, error,
                                          sizeof(error)) == GPU_SUITE_OK;
          if (ok) {
            const double elapsed = gpu_suite_clock_elapsed(&start, &end);
            result.measurement_start_timestamp = start_timestamp;
            result.measurement_end_timestamp = end_timestamp;
            result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
            result.elapsed_sec =
                gpu_suite_optional_double_value(elapsed / options.repeat);
            const bool pass = gpu_suite_thrust::set_reduction_verification(
                result, options, reduced_value, count);
            result.attempted = true;
            result.failure_origin = pass ? nullptr : "verification";
            result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
            result.status = pass ? "success" : "failure";
            result.message = pass ? "" : "OpenACC Thrust verification failed";
            any_failure = any_failure || !pass;
          } else {
            result.attempted = true;
            result.failure_origin = "benchmark";
            result.verification_status = "skipped";
            result.exit_code = gpu_suite_optional_int_value(1);
            result.status = "failure";
            result.message = "OpenACC Thrust compute pipeline failed";
            any_failure = true;
          }
          if (!writer.write(result)) {
            ok = false;
            any_failure = true;
          }
          gpu_suite_result_destroy(&result);
          if (!ok) {
            gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                       "prior-failure",
                                       "prior OpenACC Thrust failure");
            break;
          }
        }
        if (!setup_ok) {
          gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                     "OpenACC Thrust warmup failed");
          any_failure = true;
        }
      }
    } else {
      for (int trial = 0; trial < options.trials; ++trial) {
        gpu_suite_result result;
        if (!gpu_suite::initialize_result(result, options, trial)) {
          any_failure = true;
          break;
        }
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        double reduced_value = 0.0;
        double elapsed_total = 0.0;
        bool ok = true;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          struct timespec start;
          struct timespec end;
          ok = (repeat == 0
                    ? gpu_suite_measurement_start(start_timestamp, &start,
                                                  error, sizeof(error))
                    : gpu_suite_clock_now(&start, error, sizeof(error))) ==
               GPU_SUITE_OK;
#pragma acc data copyin(values_ptr[0 : count])
          {
            if (ok) {
              reduced_value = transform_reduce(values_ptr, count);
              ok = cuda_success(cudaDeviceSynchronize(),
                                "end-to-end synchronize");
            }
          }
          if (ok)
            ok = (repeat + 1 == options.repeat
                      ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                                  sizeof(error))
                      : gpu_suite_clock_now(&end, error, sizeof(error))) ==
                 GPU_SUITE_OK;
          if (ok)
            elapsed_total += gpu_suite_clock_elapsed(&start, &end);
        }
        if (ok) {
          result.measurement_start_timestamp = start_timestamp;
          result.measurement_end_timestamp = end_timestamp;
          result.elapsed_total_sec =
              gpu_suite_optional_double_value(elapsed_total);
          result.elapsed_sec =
              gpu_suite_optional_double_value(elapsed_total / options.repeat);
          const bool pass = gpu_suite_thrust::set_reduction_verification(
              result, options, reduced_value, count);
          result.attempted = true;
          result.failure_origin = pass ? nullptr : "verification";
          result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          result.status = pass ? "success" : "failure";
          result.message = pass ? "" : "OpenACC Thrust verification failed";
          any_failure = any_failure || !pass;
        } else {
          result.attempted = true;
          result.failure_origin = "benchmark";
          result.verification_status = "skipped";
          result.exit_code = gpu_suite_optional_int_value(1);
          result.status = "failure";
          result.message = "OpenACC end-to-end Thrust pipeline failed";
          any_failure = true;
        }
        if (!writer.write(result)) {
          ok = false;
          any_failure = true;
        }
        gpu_suite_result_destroy(&result);
        if (!ok) {
          gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                     "prior-failure",
                                     "prior OpenACC Thrust failure");
          break;
        }
      }
    }
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "Thrust exception: %s\n", exception.what());
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "OpenACC Thrust execution failed");
    any_failure = true;
  }
  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
