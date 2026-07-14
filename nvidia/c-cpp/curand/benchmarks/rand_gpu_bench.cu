#include "gpu_suite/benchmark.hpp"
#include "rand_bench_common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

struct RandomContext {
  curandGenerator_t generator = nullptr;
  double *values = nullptr;
};

bool cuda_success(cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool curand_success(curandStatus_t status, const char *operation) {
  if (status == CURAND_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuRAND status %d\n", operation,
               static_cast<int>(status));
  return false;
}

void destroy_context(RandomContext &context) {
  if (context.generator != nullptr)
    curandDestroyGenerator(context.generator);
  if (context.values != nullptr)
    cudaFree(context.values);
  context = RandomContext{};
}

bool create_context(RandomContext &context, std::size_t bytes,
                    const gpu_suite_options &options) {
  return cuda_success(cudaMalloc(reinterpret_cast<void **>(&context.values),
                                 bytes),
                      "cudaMalloc random output") &&
         curand_success(curandCreateGenerator(&context.generator,
                                              CURAND_RNG_PSEUDO_DEFAULT),
                        "curandCreateGenerator") &&
         curand_success(curandSetPseudoRandomGeneratorSeed(context.generator,
                                                           options.seed),
                        "curandSetPseudoRandomGeneratorSeed") &&
         curand_success(
             curandSetGeneratorOffset(context.generator, options.offset),
             "curandSetGeneratorOffset");
}

bool reset_generator(RandomContext &context, const gpu_suite_options &options) {
  return curand_success(curandSetPseudoRandomGeneratorSeed(context.generator,
                                                           options.seed),
                        "reset generator seed") &&
         curand_success(
             curandSetGeneratorOffset(context.generator, options.offset),
             "reset generator offset");
}

bool run_end_to_end_pipeline(
    const gpu_suite_options &options, std::size_t count, std::size_t bytes,
    std::vector<double> &values, int repeat_index, bool measure,
    char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY], char *error,
    std::size_t error_size, double *elapsed_total) {
  RandomContext context;
  struct timespec start;
  struct timespec end;
  bool ok = !measure ||
            (repeat_index == 0
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK;
  ok = ok && create_context(context, bytes, options) &&
       curand_success(curandGenerateUniformDouble(context.generator,
                                                  context.values, count),
                      "curandGenerateUniformDouble") &&
       cuda_success(cudaDeviceSynchronize(), "generation synchronize") &&
       cuda_success(cudaMemcpy(values.data(), context.values, bytes,
                               cudaMemcpyDeviceToHost),
                    "copy random output");
  if (ok && measure) {
    ok = (repeat_index + 1 == options.repeat
              ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                          error_size)
              : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
    if (ok)
      *elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  destroy_context(context);
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CURAND,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
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
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  std::size_t bytes = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_bytes(count, sizeof(double), &bytes)) {
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
    RandomContext persistent;
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cuRAND setup failed");
      destroy_context(persistent);
      writer.close();
      return EXIT_FAILURE;
    }
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      if (!create_context(persistent, bytes, options)) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "cuRAND compute setup failed");
        destroy_context(persistent);
        writer.close();
        return EXIT_FAILURE;
      }
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        if (!curand_success(
                curandGenerateUniformDouble(persistent.generator,
                                            persistent.values, count),
                "cuRAND warmup") ||
            !cuda_success(cudaDeviceSynchronize(), "warmup synchronize")) {
          gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                     "cuRAND warmup failed");
          destroy_context(persistent);
          writer.close();
          return EXIT_FAILURE;
        }
      }
      if (!reset_generator(persistent, options)) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "cuRAND state restoration failed");
        destroy_context(persistent);
        writer.close();
        return EXIT_FAILURE;
      }
    } else {
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        if (!run_end_to_end_pipeline(options, count, bytes, values, 0, false,
                                     nullptr, nullptr, error, sizeof(error),
                                     nullptr)) {
          gpu_suite::emit_unmeasured(
              writer, options, 0, true, "benchmark",
              "cuRAND end-to-end warmup failed");
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
      double elapsed_total = 0.0;
      bool ok = true;

      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok = reset_generator(persistent, options) &&
             cuda_success(cudaDeviceSynchronize(), "pre-timing synchronize") &&
             gpu_suite_measurement_start(start_timestamp, &start, error,
                                          sizeof(error)) == GPU_SUITE_OK;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          ok = curand_success(curandGenerateUniformDouble(persistent.generator,
                                                          persistent.values,
                                                          count),
                              "curandGenerateUniformDouble");
        }
        ok = ok &&
             cuda_success(cudaDeviceSynchronize(),
                          "post-generation synchronize") &&
             gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error)) == GPU_SUITE_OK;
        if (ok) {
          elapsed_total = gpu_suite_clock_elapsed(&start, &end);
          ok = cuda_success(cudaMemcpy(values.data(), persistent.values,
                                       bytes,
                                       cudaMemcpyDeviceToHost),
                            "copy random output");
        }
      } else {
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          ok = run_end_to_end_pipeline(
              options, count, bytes, values, repeat, true, start_timestamp,
              end_timestamp, error, sizeof(error), &elapsed_total);
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
            gpu_suite_curand::set_random_verification(
                result, options, values, "CURAND_RNG_PSEUDO_DEFAULT",
                "(0,1]");
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
        ok = ok && constructed;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "cuRAND pipeline failed";
        any_failure = true;
      }
      if (!writer.write(result)) {
        gpu_suite_result_destroy(&result);
        destroy_context(persistent);
        return EXIT_FAILURE;
      }
      gpu_suite_result_destroy(&result);
      if (!ok) {
        gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                   "prior-failure", "prior cuRAND failure");
        break;
      }
    }
    destroy_context(persistent);
  } catch (const std::length_error &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host vector size exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  }
  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
