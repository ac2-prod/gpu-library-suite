#include "rocfft_context.hpp"
#include <omp.h>
#include <cstdlib>
#include <new>
#include <stdexcept>

#ifndef _OPENMP
#error "OpenMP target support is required"
#endif

namespace {
using gpu_suite_rocfft::PlanContext;
using gpu_suite_rocfft::canonical_host;
using gpu_suite_rocfft::hip_ok;

bool mapped_to_hip_device(const void *pointer, int hip_device) {
  hipPointerAttribute_t attributes{};
  return hip_ok(hipPointerGetAttributes(&attributes, pointer),
                "hipPointerGetAttributes(OpenMP allocation)") &&
         attributes.type == hipMemoryTypeDevice &&
         attributes.device == hip_device;
}

bool prove_offload_device(int hip_device, int omp_device) {
  if (omp_device < 0 || omp_device >= omp_get_num_devices()) {
    std::fprintf(stderr, "OpenMP target device unavailable\n");
    return false;
  }
  int initial_device = 1;
#pragma omp target device(omp_device) map(from: initial_device)
  { initial_device = omp_is_initial_device(); }
  if (initial_device) {
    std::fprintf(stderr, "OpenMP host fallback rejected\n");
    return false;
  }
  float2 value[1] = {{1.0F, 0.0F}};
  float2 *mapped = value;
  bool matching = false;
#pragma omp target data device(omp_device) map(to: mapped[0:1]) use_device_ptr(mapped)
  { matching = mapped_to_hip_device(mapped, hip_device); }
  if (!matching) {
    std::fprintf(stderr, "OpenMP allocation is not on requested HIP device\n");
    return false;
  }
  std::fprintf(stderr, "OpenMP offload verified: omp_device=%d, HIP allocation device=%d\n",
               omp_device, hip_device);
  return true;
}

bool mapped_execute(const PlanContext &context, float2 *input,
                    float2 *output, int repeat, int hip_device,
                    int omp_device) {
  bool ok = false;
#pragma omp target data device(omp_device) use_device_ptr(input, output)
  {
    // Validate HIP ownership of both actual mappings before using either pointer.
    ok = mapped_to_hip_device(input, hip_device) &&
         mapped_to_hip_device(output, hip_device) &&
         context.execute(input, output, repeat);
  }
  return ok;
}

bool warmup_end_to_end(const gpu_suite_options &options,
                      std::vector<float2> &input, std::vector<float2> &output,
                      std::size_t count, int omp_device) {
  float2 *input_data = input.data();
  float2 *output_data = output.data();
  for (int warmup = 0; warmup < options.warmup; ++warmup) {
    PlanContext context;
    canonical_host(input, output);
    if (!context.create(options.size, options.batch)) return false;
    bool ok = false;
#pragma omp target data device(omp_device) map(to: input_data[0:count]) map(from: output_data[0:count])
    {
      ok = mapped_execute(context, input_data, output_data, 1,
                          options.device, omp_device);
    }
    const bool cleaned = context.destroy();
    if (!ok || !cleaned) return false;
  }
  canonical_host(input, output);
  return true;
}

bool run_end_to_end_trial(const gpu_suite_options &options,
                         std::vector<float2> &input, std::vector<float2> &output,
                         std::size_t count, int omp_device,
                         char *start_timestamp, char *end_timestamp,
                         double &elapsed_total, const char *&message) {
  float2 *input_data = input.data();
  float2 *output_data = output.data();
  char error[256] = {0};
  elapsed_total = 0.0;
  for (int repeat = 0; repeat < options.repeat; ++repeat) {
    PlanContext context;
    struct timespec start, end;
    canonical_host(input, output);
    const int start_status = repeat == 0
        ? gpu_suite_measurement_start(start_timestamp, &start, error, sizeof(error))
        : gpu_suite_clock_now(&start, error, sizeof(error));
    if (start_status != GPU_SUITE_OK) {
      message = "could not read measurement start clock";
      return false;
    }
    if (!context.create(options.size, options.batch)) {
      message = "rocFFT setup failed";
      return false;
    }
    bool ok = false;
#pragma omp target data device(omp_device) map(to: input_data[0:count]) map(from: output_data[0:count])
    {
      ok = mapped_execute(context, input_data, output_data, 1,
                          options.device, omp_device);
    } // Completion and map(from:) precede the end timestamp.
    const int end_status = repeat + 1 == options.repeat
        ? gpu_suite_measurement_end(&end, end_timestamp, error, sizeof(error))
        : gpu_suite_clock_now(&end, error, sizeof(error));
    const bool cleaned = context.destroy(); // Explicit cleanup is untimed.
    if (!ok || end_status != GPU_SUITE_OK || !cleaned) {
      message = "OpenMP rocFFT end-to-end pipeline failed";
      return false;
    }
    elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  return true;
}

bool run_compute(const gpu_suite_options &options,
                 gpu_suite_fft_bench::Writer &writer,
                 std::vector<float2> &input, std::vector<float2> &output,
                 std::size_t count, int omp_device) {
  PlanContext context;
  if (!context.create(options.size, options.batch)) {
    (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, true,
                                              "benchmark", "rocFFT setup failed");
    return false;
  }
  float2 *input_data = input.data();
  float2 *output_data = output.data();
  bool any_failure = false;
  bool prepared = false;
  char error[256] = {0};
#pragma omp target data device(omp_device) map(to: input_data[0:count]) map(from: output_data[0:count])
  {
    prepared = true;
    for (int warmup = 0; warmup < options.warmup && prepared; ++warmup) {
      canonical_host(input, output);
#pragma omp target update device(omp_device) to(input_data[0:count], output_data[0:count])
      prepared = mapped_execute(context, input_data, output_data, 1,
                                 options.device, omp_device);
    }
    if (!prepared) {
      (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, true,
                                                "benchmark", "OpenMP rocFFT warm-up failed");
      any_failure = true;
    }
    for (int trial = 0; trial < options.trials && prepared; ++trial) {
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec start, end;
      double elapsed_total = 0.0;
      const char *message = "";
      bool operation_ok = true;
      for (int repeat = 0; repeat < options.repeat && operation_ok; ++repeat) {
        // Out-of-place rocFFT input may be destructive; restore before every
        // separately timed FFT while retaining the mapping and workspace.
        canonical_host(input, output);
#pragma omp target update device(omp_device) to(input_data[0:count], output_data[0:count])
        operation_ok = hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize before timing");
#pragma omp target data device(omp_device) use_device_ptr(input_data, output_data)
        {
          operation_ok = operation_ok &&
              mapped_to_hip_device(input_data, options.device) &&
              mapped_to_hip_device(output_data, options.device);
          if (operation_ok) {
            const int start_status = repeat == 0
                ? gpu_suite_measurement_start(start_timestamp, &start, error, sizeof(error))
                : gpu_suite_clock_now(&start, error, sizeof(error));
            operation_ok = start_status == GPU_SUITE_OK;
          }
          if (operation_ok) {
            operation_ok = context.execute(input_data, output_data, 1);
          }
          if (operation_ok) {
            const int end_status = repeat + 1 == options.repeat
                ? gpu_suite_measurement_end(&end, end_timestamp, error, sizeof(error))
                : gpu_suite_clock_now(&end, error, sizeof(error));
            operation_ok = end_status == GPU_SUITE_OK;
          }
        }
        if (operation_ok) elapsed_total += gpu_suite_clock_elapsed(&start, &end);
      }
      if (operation_ok) {
#pragma omp target update device(omp_device) from(output_data[0:count])
      } else {
        message = "OpenMP rocFFT compute trial failed";
      }
      bool row_success = false;
      if (!gpu_suite_fft_bench::emit_measured(writer, options, trial,
              start_timestamp, end_timestamp, elapsed_total, output.data(),
              count, operation_ok, message, row_success)) {
        any_failure = true;
        break;
      }
      any_failure = any_failure || !row_success;
      if (!operation_ok) {
        if (trial + 1 < options.trials) {
          (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, trial + 1,
                                                    false, "prior-failure", message);
        }
        break;
      }
    }
  }
  const bool cleaned = context.destroy();
  return !any_failure && cleaned;
}
} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCFFT,
                         GPU_SUITE_IMPLEMENTATION_OPENMP);
  const auto parsed = gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCFFT);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }
  gpu_suite_fft_bench::Writer writer;
  if (!gpu_suite_fft_bench::open_writer(options, writer)) return EXIT_FAILURE;
  std::size_t count = 0, bytes = 0;
  int nfft = 0, batch = 0;
  const int omp_device = omp_get_default_device();
  if (!gpu_suite_rocfft::dimensions(options, count, bytes, nfft, batch)) {
    (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, false,
                                              "prerequisite", "problem dimensions overflow host representation");
    (void)gpu_suite_fft_bench::close_writer(writer);
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(options.device), "hipSetDevice") ||
      !prove_offload_device(options.device, omp_device)) {
    (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, false,
                                              "prerequisite", "OpenMP/HIP device interoperability not proven");
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
  bool any_failure = false;
  try {
    std::vector<float2> input(count), output(count);
    canonical_host(input, output);
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      any_failure = !run_compute(options, writer, input, output, count, omp_device);
    } else if (!warmup_end_to_end(options, input, output, count, omp_device)) {
      (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, true,
                                                "benchmark", "OpenMP rocFFT warm-up failed");
      any_failure = true;
    } else {
      for (int trial = 0; trial < options.trials; ++trial) {
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        double elapsed_total = 0.0;
        const char *message = "";
        const bool operation_ok = run_end_to_end_trial(options, input, output,
            count, omp_device, start_timestamp, end_timestamp, elapsed_total, message);
        bool row_success = false;
        if (!gpu_suite_fft_bench::emit_measured(writer, options, trial,
                start_timestamp, end_timestamp, elapsed_total, output.data(),
                count, operation_ok, message, row_success)) {
          any_failure = true; break;
        }
        any_failure = any_failure || !row_success;
        if (!operation_ok) {
          if (trial + 1 < options.trials) {
            (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, trial + 1,
                                                      false, "prior-failure", message);
          }
          break;
        }
      }
    }
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "host allocation failed: %s\n", exception.what());
    (void)gpu_suite_fft_bench::emit_unmeasured(writer, options, 0, true,
                                              "benchmark", "host allocation failed");
    any_failure = true;
  }
  if (!runtime.destroy()) any_failure = true;
  if (!gpu_suite_fft_bench::close_writer(writer)) return EXIT_FAILURE;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
