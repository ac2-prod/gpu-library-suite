#include "rocfft_context.hpp"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

using gpu_suite_rocfft::hip_ok;
struct DeviceContext {
  float2 *input = nullptr;
  float2 *output = nullptr;
  gpu_suite_rocfft::PlanContext fft;
};

bool destroy_context(DeviceContext &context) {
  bool ok = context.fft.destroy();
  if (context.output) ok = hip_ok(hipFree(context.output), "hipFree(output)") && ok;
  if (context.input) ok = hip_ok(hipFree(context.input), "hipFree(input)") && ok;
  context.input = nullptr;
  context.output = nullptr;
  return ok;
}

bool create_context(DeviceContext &context, int nfft, int batch,
                    std::size_t bytes) {
  if (!context.fft.create(static_cast<std::size_t>(nfft), static_cast<std::size_t>(batch)) ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&context.input), bytes), "hipMalloc(input)") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&context.output), bytes), "hipMalloc(output)")) {
    (void)destroy_context(context);
    return false;
  }
  return true;
}

void canonical_host(std::vector<float2> &input,
                    std::vector<float2> &output) {
  for (float2 &value : input) {
    value.x = 1.0F;
    value.y = 0.0F;
  }
  for (float2 &value : output) {
    value.x = 0.0F;
    value.y = 0.0F;
  }
}

bool restore_device(const DeviceContext &context,
                    const std::vector<float2> &input, std::size_t bytes) {
  return hip_ok(hipMemcpy(context.input, input.data(), bytes,
                            hipMemcpyHostToDevice),
                 "hipMemcpy(H2D input)") &&
         hip_ok(hipMemset(context.output, 0, bytes), "hipMemset(output)");
}

bool warmup_compute(const DeviceContext &context,
                    const gpu_suite_options &options,
                    const std::vector<float2> &input, std::size_t bytes) {
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    if (!restore_device(context, input, bytes) ||
        !context.fft.execute(context.input, context.output, 1)) return false;
  }
  return true;
}

bool warmup_end_to_end(const gpu_suite_options &options,
                       std::vector<float2> &input,
                       std::vector<float2> &output, std::size_t bytes,
                       int nfft, int batch) {
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    DeviceContext context;
    canonical_host(input, output);
    if (!create_context(context, nfft, batch, bytes) ||
        !restore_device(context, input, bytes) ||
        !context.fft.execute(context.input, context.output, 1) ||
        !hip_ok(hipMemcpy(output.data(), context.output, bytes,
                            hipMemcpyDeviceToHost),
                 "hipMemcpy(D2H warm-up)")) {
      destroy_context(context);
      return false;
    }
    if (!destroy_context(context)) return false;
  }
  canonical_host(input, output);
  return true;
}

bool run_compute_trial(const DeviceContext &context,
                       const gpu_suite_options &options,
                       const std::vector<float2> &input,
                       std::vector<float2> &output, std::size_t bytes,
                       char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                       char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                       double &elapsed_total, const char *&message) {
  char error[256] = {0};
  struct timespec start;
  struct timespec end;
  elapsed_total = 0.0;
  for (int repeat = 0; repeat < options.repeat; ++repeat) {
    // rocFFT may overwrite out-of-place input. Restore each operation outside
    // its measured segment, keeping the plan, arrays and workspace alive.
    if (!restore_device(context, input, bytes) ||
        !hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize before timing")) {
      message = "could not restore canonical device state";
      return false;
    }
    const int start_status = repeat == 0
        ? gpu_suite_measurement_start(start_timestamp, &start, error, sizeof(error))
        : gpu_suite_clock_now(&start, error, sizeof(error));
    if (start_status != GPU_SUITE_OK ||
        !context.fft.execute(context.input, context.output, 1)) {
      message = "could not execute synchronized FFT segment";
      return false;
    }
    const int end_status = repeat + 1 == options.repeat
        ? gpu_suite_measurement_end(&end, end_timestamp, error, sizeof(error))
        : gpu_suite_clock_now(&end, error, sizeof(error));
    if (end_status != GPU_SUITE_OK) {
      message = "could not read measurement end clock";
      return false;
    }
    elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  if (!hip_ok(hipMemcpy(output.data(), context.output, bytes,
                          hipMemcpyDeviceToHost),
               "hipMemcpy(D2H output)")) {
    message = "could not retrieve output";
    return false;
  }
  return true;
}

bool run_end_to_end_trial(const gpu_suite_options &options,
                          std::vector<float2> &input,
                          std::vector<float2> &output, std::size_t bytes,
                          int nfft, int batch,
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
    if (!create_context(context, nfft, batch, bytes) ||
        !restore_device(context, input, bytes) ||
        !context.fft.execute(context.input, context.output, 1) ||
        !hip_ok(hipMemcpy(output.data(), context.output, bytes,
                            hipMemcpyDeviceToHost),
                 "hipMemcpy(D2H output)")) {
      destroy_context(context);
      message = "end-to-end rocFFT pipeline failed";
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
    if (!destroy_context(context)) {
      message = "rocFFT cleanup failed"; return false;
    }
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
  int nfft = 0;
  int batch = 0;
  bool any_failure = false;

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCFFT,
                         GPU_SUITE_IMPLEMENTATION_HIP);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCFFT);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    gpu_suite_options_usage(stderr, argv[0], GPU_SUITE_BENCHMARK_ROCFFT);
    return EXIT_FAILURE;
  }
  if (!gpu_suite_fft_bench::open_writer(options, writer)) {
    return EXIT_FAILURE;
  }
  if (!gpu_suite_checked_u64_to_int(options.size, &nfft) ||
      !gpu_suite_checked_u64_to_int(options.batch, &batch) ||
      !gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_mul_size(
          count, static_cast<std::size_t>(batch), &count) ||
      !gpu_suite_checked_bytes(count, sizeof(float2), &bytes)) {
    std::fprintf(stderr, "rocFFT dimensions overflow host representation\n");
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "problem dimensions overflow host representation");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(options.device), "hipSetDevice")) {
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, true, "benchmark", "hipSetDevice failed");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }

  gpu_suite_rocfft::RuntimeContext runtime;
  if (!runtime.initialize()) {
    (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, true,
        "benchmark", "rocFFT global initialization failed");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }

  try {
    std::vector<float2> input(count);
    std::vector<float2> output(count);
    canonical_host(input, output);
    DeviceContext compute_context;

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      if (!create_context(compute_context, nfft, batch, bytes) ||
          !restore_device(compute_context, input, bytes) ||
          !warmup_compute(compute_context, options, input, bytes)) {
        (void)gpu_suite_fft_bench::emit_unmeasured(
            writer, options, 0, true, "benchmark",
            "rocFFT setup/warm-up failed");
        destroy_context(compute_context);
        (void)gpu_suite_fft_bench::close_writer(writer);
        return EXIT_FAILURE;
      }
    } else if (!warmup_end_to_end(options, input, output, bytes, nfft,
                                  batch)) {
      (void)gpu_suite_fft_bench::emit_unmeasured(
          writer, options, 0, true, "benchmark",
          "rocFFT end-to-end warm-up failed");
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
              : run_end_to_end_trial(options, input, output, bytes, nfft,
                                     batch,
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
    if (!destroy_context(compute_context)) any_failure = true;
  } catch (const std::length_error &) {
    std::fprintf(stderr, "host vector size exceeds max_size\n");
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, true, "benchmark",
        "host vector size exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "host allocation failed\n");
    (void)gpu_suite_fft_bench::emit_unmeasured(
        writer, options, 0, true, "benchmark", "host allocation failed");
    any_failure = true;
  }
  if (!runtime.destroy()) any_failure = true;
  if (!gpu_suite_fft_bench::close_writer(writer)) {
    return EXIT_FAILURE;
  }
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
