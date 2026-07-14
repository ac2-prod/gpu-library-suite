#include "fft_bench_result.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

struct DeviceContext {
  cufftComplex *input = nullptr;
  cufftComplex *output = nullptr;
  cufftHandle plan = 0;
  bool plan_created = false;
};

bool cuda_ok(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool cufft_ok(cufftResult status, const char *operation) {
  if (status == CUFFT_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed with cuFFT status %d\n", operation,
               static_cast<int>(status));
  return false;
}

void destroy_context(DeviceContext &context) {
  if (context.plan_created) {
    (void)cufft_ok(cufftDestroy(context.plan), "cufftDestroy");
  }
  if (context.output != nullptr) {
    (void)cuda_ok(cudaFree(context.output), "cudaFree(output)");
  }
  if (context.input != nullptr) {
    (void)cuda_ok(cudaFree(context.input), "cudaFree(input)");
  }
  context = DeviceContext{};
}

bool create_context(DeviceContext &context, int nfft, int batch,
                    std::size_t bytes) {
  int length[1] = {nfft};
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&context.input), bytes),
               "cudaMalloc(input)") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&context.output), bytes),
               "cudaMalloc(output)")) {
    destroy_context(context);
    return false;
  }
  if (!cufft_ok(cufftPlanMany(&context.plan, 1, length, nullptr, 1, nfft,
                              nullptr, 1, nfft, CUFFT_C2C, batch),
                "cufftPlanMany")) {
    destroy_context(context);
    return false;
  }
  context.plan_created = true;
  return true;
}

void canonical_host(std::vector<cufftComplex> &input,
                    std::vector<cufftComplex> &output) {
  for (cufftComplex &value : input) {
    value.x = 1.0F;
    value.y = 0.0F;
  }
  for (cufftComplex &value : output) {
    value.x = 0.0F;
    value.y = 0.0F;
  }
}

bool restore_device(const DeviceContext &context,
                    const std::vector<cufftComplex> &input, std::size_t bytes) {
  return cuda_ok(cudaMemcpy(context.input, input.data(), bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(H2D input)") &&
         cuda_ok(cudaMemset(context.output, 0, bytes), "cudaMemset(output)");
}

bool warmup_compute(const DeviceContext &context,
                    const gpu_suite_options &options) {
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    if (!cufft_ok(cufftExecC2C(context.plan, context.input, context.output,
                               CUFFT_FORWARD),
                  "cufftExecC2C warm-up")) {
      return false;
    }
  }
  return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize warm-up");
}

bool warmup_end_to_end(const gpu_suite_options &options,
                       std::vector<cufftComplex> &input,
                       std::vector<cufftComplex> &output, std::size_t bytes) {
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    DeviceContext context;
    canonical_host(input, output);
    if (!create_context(context, static_cast<int>(options.size),
                        static_cast<int>(options.batch), bytes) ||
        !restore_device(context, input, bytes) ||
        !cufft_ok(cufftExecC2C(context.plan, context.input, context.output,
                               CUFFT_FORWARD),
                  "cufftExecC2C warm-up") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize warm-up") ||
        !cuda_ok(cudaMemcpy(output.data(), context.output, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(D2H warm-up)")) {
      destroy_context(context);
      return false;
    }
    destroy_context(context);
  }
  canonical_host(input, output);
  return true;
}

bool run_compute_trial(const DeviceContext &context,
                       const gpu_suite_options &options,
                       const std::vector<cufftComplex> &input,
                       std::vector<cufftComplex> &output, std::size_t bytes,
                       char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                       char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                       double &elapsed_total, const char *&message) {
  char error[256] = {0};
  struct timespec start;
  struct timespec end;
  if (!restore_device(context, input, bytes) ||
      !cuda_ok(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize before timing")) {
    message = "could not restore canonical device state";
    return false;
  }
  if (gpu_suite_measurement_start(start_timestamp, &start, error,
                                  sizeof(error)) != GPU_SUITE_OK) {
    message = "could not read measurement start clock";
    return false;
  }
  for (int repeat = 0; repeat < options.repeat; ++repeat) {
    if (!cufft_ok(cufftExecC2C(context.plan, context.input, context.output,
                               CUFFT_FORWARD),
                  "cufftExecC2C")) {
      message = "cufftExecC2C failed";
      return false;
    }
  }
  if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize after timing") ||
      gpu_suite_measurement_end(&end, end_timestamp, error, sizeof(error)) !=
          GPU_SUITE_OK) {
    message = "could not complete synchronized timing";
    return false;
  }
  elapsed_total = gpu_suite_clock_elapsed(&start, &end);
  if (!cuda_ok(cudaMemcpy(output.data(), context.output, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(D2H output)")) {
    message = "could not retrieve output";
    return false;
  }
  return true;
}

bool run_end_to_end_trial(const gpu_suite_options &options,
                          std::vector<cufftComplex> &input,
                          std::vector<cufftComplex> &output, std::size_t bytes,
                          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                          double &elapsed_total, const char *&message) {
  char error[256] = {0};
  elapsed_total = 0.0;
  for (int repeat = 0; repeat < options.repeat; ++repeat) {
    DeviceContext context;
    struct timespec start;
    struct timespec end;
    canonical_host(input, output);
    const int start_status =
        repeat == 0
            ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                          sizeof(error))
            : gpu_suite_clock_now(&start, error, sizeof(error));
    if (start_status != GPU_SUITE_OK) {
      message = "could not read measurement start clock";
      return false;
    }
    if (!create_context(context, static_cast<int>(options.size),
                        static_cast<int>(options.batch), bytes) ||
        !restore_device(context, input, bytes) ||
        !cufft_ok(cufftExecC2C(context.plan, context.input, context.output,
                               CUFFT_FORWARD),
                  "cufftExecC2C") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !cuda_ok(cudaMemcpy(output.data(), context.output, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(D2H output)")) {
      destroy_context(context);
      message = "end-to-end cuFFT pipeline failed";
      return false;
    }
    const int end_status =
        repeat + 1 == options.repeat
            ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error))
            : gpu_suite_clock_now(&end, error, sizeof(error));
    if (end_status != GPU_SUITE_OK) {
      destroy_context(context);
      message = "could not read measurement end clock";
      return false;
    }
    elapsed_total += gpu_suite_clock_elapsed(&start, &end);
    destroy_context(context);
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_fft_bench::Writer writer;
  std::size_t count = 0;
  std::size_t bytes = 0;
  bool any_failure = false;

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUFFT);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    gpu_suite_options_usage(stderr, argv[0], GPU_SUITE_BENCHMARK_CUFFT);
    return EXIT_FAILURE;
  }
  if (!gpu_suite_fft_bench::open_writer(options, writer)) {
    return EXIT_FAILURE;
  }
  if (options.size > static_cast<std::uint64_t>(INT_MAX) ||
      options.batch > static_cast<std::uint64_t>(INT_MAX) ||
      !gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_mul_size(
          count, static_cast<std::size_t>(options.batch), &count) ||
      !gpu_suite_checked_mul_size(count, sizeof(cufftComplex), &bytes)) {
    std::fprintf(stderr, "cuFFT dimensions overflow host representation\n");
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "problem dimensions overflow host representation");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }
  if (!cuda_ok(cudaSetDevice(options.device), "cudaSetDevice")) {
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, true, "benchmark", "cudaSetDevice failed");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }

  try {
    std::vector<cufftComplex> input(count);
    std::vector<cufftComplex> output(count);
    canonical_host(input, output);
    DeviceContext compute_context;

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      if (!create_context(compute_context, static_cast<int>(options.size),
                          static_cast<int>(options.batch), bytes) ||
          !restore_device(compute_context, input, bytes) ||
          !warmup_compute(compute_context, options)) {
        (void)gpu_suite_fft_bench::emit_unmeasured(
            writer, options, 0, true, "benchmark",
            "cuFFT setup/warm-up failed");
        destroy_context(compute_context);
        (void)gpu_suite_fft_bench::close_writer(writer);
        return EXIT_FAILURE;
      }
    } else if (!warmup_end_to_end(options, input, output, bytes)) {
      (void)gpu_suite_fft_bench::emit_unmeasured(
          writer, options, 0, true, "benchmark",
          "cuFFT end-to-end warm-up failed");
      (void)gpu_suite_fft_bench::close_writer(writer);
      return EXIT_FAILURE;
    }

    for (int trial = 0; trial < options.trials; ++trial) {
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      double elapsed_total = 0.0;
      const char *message = "";
      canonical_host(input, output);
      const bool operation_ok =
          options.scope == GPU_SUITE_SCOPE_COMPUTE
              ? run_compute_trial(compute_context, options, input, output,
                                  bytes, start_timestamp, end_timestamp,
                                  elapsed_total, message)
              : run_end_to_end_trial(options, input, output, bytes,
                                     start_timestamp, end_timestamp,
                                     elapsed_total, message);
      bool row_success = false;
      if (!gpu_suite_fft_bench::emit_measured(
              writer, options, trial, start_timestamp, end_timestamp,
              elapsed_total, output.data(), count, operation_ok, message,
              row_success)) {
        destroy_context(compute_context);
        (void)gpu_suite_fft_bench::close_writer(writer);
        return EXIT_FAILURE;
      }
      any_failure = any_failure || !row_success;
      if (!operation_ok) {
        if (trial + 1 < options.trials) {
          (void)gpu_suite_fft_bench::emit_unmeasured(
              writer, options, trial + 1, false, "prior-failure", message);
        }
        break;
      }
    }
    destroy_context(compute_context);
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "host allocation failed\n");
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, true, "benchmark", "host allocation failed");
    any_failure = true;
  }
  if (!gpu_suite_fft_bench::close_writer(writer)) {
    return EXIT_FAILURE;
  }
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
