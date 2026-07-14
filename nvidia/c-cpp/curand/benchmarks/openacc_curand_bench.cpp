#include "gpu_suite/benchmark.hpp"
#include "rand_bench_common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

bool cuda_success(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}

bool curand_success(curandStatus_t status, const char *api) {
  if (status == CURAND_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuRAND status %d\n", api,
               static_cast<int>(status));
  return false;
}

bool configure_generator(curandGenerator_t generator,
                         const gpu_suite_options &options) {
  return curand_success(
             curandSetPseudoRandomGeneratorSeed(generator, options.seed),
             "curandSetPseudoRandomGeneratorSeed") &&
         curand_success(curandSetGeneratorOffset(generator, options.offset),
                        "curandSetGeneratorOffset");
}

bool generate(curandGenerator_t generator, double *values, std::size_t count) {
  bool ok = true;
#pragma acc host_data use_device(values)
  {
    ok = curand_success(curandGenerateUniformDouble(generator, values, count),
                        "curandGenerateUniformDouble");
  }
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CURAND,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CURAND);
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
    std::vector<double> values(count);
    double *values_ptr = values.data();
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      curandGenerator_t generator = nullptr;
      bool setup_ok =
          curand_success(
              curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT),
              "curandCreateGenerator") &&
          configure_generator(generator, options);
#pragma acc data copyout(values_ptr[0 : count])
      {
        for (int warmup = 0; warmup < options.warmup && setup_ok; ++warmup) {
          setup_ok = generate(generator, values_ptr, count) &&
                     cuda_success(cudaDeviceSynchronize(),
                                  "warmup synchronize");
        }
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
          bool ok =
              configure_generator(generator, options) &&
              cuda_success(cudaDeviceSynchronize(),
                           "pre-timing synchronize") &&
              gpu_suite_measurement_start(start_timestamp, &start, error,
                                           sizeof(error)) == GPU_SUITE_OK;
          for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
            ok = generate(generator, values_ptr, count);
          }
          ok = ok && cuda_success(cudaDeviceSynchronize(),
                                  "post-timing synchronize") &&
               gpu_suite_measurement_end(&end, end_timestamp, error,
                                          sizeof(error)) == GPU_SUITE_OK;
#pragma acc update self(values_ptr[0 : count])
          ok = ok && cuda_success(cudaDeviceSynchronize(),
                                  "OpenACC update self random output");
          if (ok) {
            const double elapsed = gpu_suite_clock_elapsed(&start, &end);
            result.measurement_start_timestamp = start_timestamp;
            result.measurement_end_timestamp = end_timestamp;
            result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
            result.elapsed_sec =
                gpu_suite_optional_double_value(elapsed / options.repeat);
            const bool pass = gpu_suite_curand::set_random_verification(
                result, options, values, "CURAND_RNG_PSEUDO_DEFAULT", "(0,1]");
            result.attempted = true;
            result.failure_origin = pass ? nullptr : "verification";
            result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
            result.status = pass ? "success" : "failure";
            result.message =
                pass ? "" : "OpenACC cuRAND statistical verification failed";
            any_failure = any_failure || !pass;
          } else {
            result.attempted = true;
            result.failure_origin = "benchmark";
            result.verification_status = "skipped";
            result.exit_code = gpu_suite_optional_int_value(1);
            result.status = "failure";
            result.message = "OpenACC cuRAND compute pipeline failed";
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
                                       "prior OpenACC cuRAND failure");
            break;
          }
        }
      }
      if (generator != nullptr)
        curandDestroyGenerator(generator);
      if (!setup_ok) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "OpenACC cuRAND setup/warmup failed");
        any_failure = true;
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
        double elapsed_total = 0.0;
        bool ok = true;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          curandGenerator_t generator = nullptr;
          struct timespec start;
          struct timespec end;
          ok = (repeat == 0
                    ? gpu_suite_measurement_start(start_timestamp, &start,
                                                  error, sizeof(error))
                    : gpu_suite_clock_now(&start, error, sizeof(error))) ==
                   GPU_SUITE_OK &&
               curand_success(
                   curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT),
                   "curandCreateGenerator") &&
               configure_generator(generator, options);
#pragma acc data copyout(values_ptr[0 : count])
          {
            if (ok) {
              ok = generate(generator, values_ptr, count) &&
                   cuda_success(cudaDeviceSynchronize(),
                                "generation synchronize");
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
          if (generator != nullptr)
            curandDestroyGenerator(generator);
        }
        if (ok) {
          result.measurement_start_timestamp = start_timestamp;
          result.measurement_end_timestamp = end_timestamp;
          result.elapsed_total_sec =
              gpu_suite_optional_double_value(elapsed_total);
          result.elapsed_sec =
              gpu_suite_optional_double_value(elapsed_total / options.repeat);
          const bool pass = gpu_suite_curand::set_random_verification(
              result, options, values, "CURAND_RNG_PSEUDO_DEFAULT", "(0,1]");
          result.attempted = true;
          result.failure_origin = pass ? nullptr : "verification";
          result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          result.status = pass ? "success" : "failure";
          result.message =
              pass ? "" : "OpenACC cuRAND statistical verification failed";
          any_failure = any_failure || !pass;
        } else {
          result.attempted = true;
          result.failure_origin = "benchmark";
          result.verification_status = "skipped";
          result.exit_code = gpu_suite_optional_int_value(1);
          result.status = "failure";
          result.message = "OpenACC end-to-end cuRAND pipeline failed";
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
                                     "prior OpenACC cuRAND failure");
          break;
        }
      }
    }
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  }
  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
