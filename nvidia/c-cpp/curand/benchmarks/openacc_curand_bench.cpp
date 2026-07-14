#include "gpu_suite/benchmark.hpp"
#include "rand_bench_common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

bool configure_generator(curandGenerator_t generator,
                         const gpu_suite_options &options) {
  return curandSetPseudoRandomGeneratorSeed(generator, options.seed) ==
             CURAND_STATUS_SUCCESS &&
         curandSetGeneratorOffset(generator, options.offset) ==
             CURAND_STATUS_SUCCESS;
}

bool generate(curandGenerator_t generator, double *values, std::size_t count) {
  bool ok = true;
#pragma acc host_data use_device(values)
  {
    ok = curandGenerateUniformDouble(generator, values, count) ==
         CURAND_STATUS_SUCCESS;
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
    if (cudaSetDevice(options.device) != cudaSuccess) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      curandGenerator_t generator = nullptr;
      bool setup_ok =
          curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) ==
              CURAND_STATUS_SUCCESS &&
          configure_generator(generator, options);
#pragma acc data copyout(values_ptr[0 : count])
      {
        for (int warmup = 0; warmup < options.warmup && setup_ok; ++warmup) {
          setup_ok = generate(generator, values_ptr, count) &&
                     cudaDeviceSynchronize() == cudaSuccess;
        }
        for (int trial = 0; trial < options.trials && setup_ok; ++trial) {
          gpu_suite_result result;
          gpu_suite::initialize_result(result, options, trial);
          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          struct timespec start;
          struct timespec end;
          bool ok =
              configure_generator(generator, options) &&
              gpu_suite_utc_timestamp(start_timestamp, error, sizeof(error)) ==
                  GPU_SUITE_OK &&
              cudaDeviceSynchronize() == cudaSuccess &&
              gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK;
          for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
            ok = generate(generator, values_ptr, count);
          }
          ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
               gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
#pragma acc update self(values_ptr[0 : count])
          ok = ok && gpu_suite_utc_timestamp(end_timestamp, error,
                                             sizeof(error)) == GPU_SUITE_OK;
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
          writer.write(result);
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
        gpu_suite::initialize_result(result, options, trial);
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        double elapsed_total = 0.0;
        bool ok = gpu_suite_utc_timestamp(start_timestamp, error,
                                          sizeof(error)) == GPU_SUITE_OK;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          curandGenerator_t generator = nullptr;
          struct timespec start;
          struct timespec end;
          ok = gpu_suite_clock_now(&start, error, sizeof(error)) ==
                   GPU_SUITE_OK &&
               curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) ==
                   CURAND_STATUS_SUCCESS &&
               configure_generator(generator, options);
#pragma acc data copyout(values_ptr[0 : count])
          {
            if (ok) {
              ok = generate(generator, values_ptr, count) &&
                   cudaDeviceSynchronize() == cudaSuccess;
            }
          }
          ok = ok &&
               gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
          if (ok)
            elapsed_total += gpu_suite_clock_elapsed(&start, &end);
          if (generator != nullptr)
            curandDestroyGenerator(generator);
        }
        ok = ok && gpu_suite_utc_timestamp(end_timestamp, error,
                                           sizeof(error)) == GPU_SUITE_OK;
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
        writer.write(result);
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
