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

bool create_plan(cufftHandle &plan, const gpu_suite_options &options) {
  int length[1] = {static_cast<int>(options.size)};
  return cufft_ok(cufftPlanMany(&plan, 1, length, nullptr, 1, length[0],
                                nullptr, 1, length[0], CUFFT_C2C,
                                static_cast<int>(options.batch)),
                  "cufftPlanMany");
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

bool execute_device_plan(cufftHandle plan, cufftComplex *input_data,
                         cufftComplex *output_data, int repeat,
                         const char *label) {
  bool success = true;
#pragma acc host_data use_device(input_data, output_data)
  {
    for (int iteration = 0; iteration < repeat; ++iteration) {
      if (!cufft_ok(cufftExecC2C(plan, input_data, output_data, CUFFT_FORWARD),
                    label)) {
        success = false;
        break;
      }
    }
  }
  if (success) {
    success = cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
  return success;
}

bool warmup_end_to_end(const gpu_suite_options &options,
                       std::vector<cufftComplex> &input,
                       std::vector<cufftComplex> &output, std::size_t count) {
  cufftComplex *input_data = input.data();
  cufftComplex *output_data = output.data();
  for (int warmup = 0; warmup < options.warmup; ++warmup) {
    cufftHandle plan = 0;
    canonical_host(input, output);
    if (!create_plan(plan, options)) {
      return false;
    }
    bool success = true;
#pragma acc data copyin(input_data[0 : count]) copyout(output_data[0 : count])
    {
      success = execute_device_plan(plan, input_data, output_data, 1,
                                    "cufftExecC2C warm-up");
    }
    const bool destroyed = cufft_ok(cufftDestroy(plan), "cufftDestroy");
    if (!success || !destroyed) {
      return false;
    }
  }
  canonical_host(input, output);
  return true;
}

bool run_end_to_end_trial(const gpu_suite_options &options,
                          std::vector<cufftComplex> &input,
                          std::vector<cufftComplex> &output, std::size_t count,
                          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                          double &elapsed_total, const char *&message) {
  cufftComplex *input_data = input.data();
  cufftComplex *output_data = output.data();
  char error[256] = {0};
  elapsed_total = 0.0;
  for (int repeat = 0; repeat < options.repeat; ++repeat) {
    cufftHandle plan = 0;
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
    if (!create_plan(plan, options)) {
      message = "cufftPlanMany failed";
      return false;
    }
    bool success = true;
#pragma acc data copyin(input_data[0 : count]) copyout(output_data[0 : count])
    {
      success =
          execute_device_plan(plan, input_data, output_data, 1, "cufftExecC2C");
    }
    const int end_status =
        repeat + 1 == options.repeat
            ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error))
            : gpu_suite_clock_now(&end, error, sizeof(error));
    if (!success || end_status != GPU_SUITE_OK) {
      (void)cufft_ok(cufftDestroy(plan), "cufftDestroy");
      message = success ? "could not read measurement end clock"
                        : "OpenACC cuFFT pipeline failed";
      return false;
    }
    elapsed_total += gpu_suite_clock_elapsed(&start, &end);
    if (!cufft_ok(cufftDestroy(plan), "cufftDestroy")) {
      message = "cufftDestroy failed";
      return false;
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
  bool any_failure = false;

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
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
    (void)bytes;
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
    cufftComplex *input_data = input.data();
    cufftComplex *output_data = output.data();
    canonical_host(input, output);

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      cufftHandle plan = 0;
      if (!create_plan(plan, options)) {
        (void)gpu_suite_fft_bench::emit_unmeasured(
            writer, options, 0, true, "benchmark", "cufftPlanMany failed");
        (void)gpu_suite_fft_bench::close_writer(writer);
        return EXIT_FAILURE;
      }
      bool fatal_failure = false;
      const char *fatal_message = "";
#pragma acc data copyin(input_data[0 : count]) copyout(output_data[0 : count])
      {
        if (options.warmup > 0 &&
            !execute_device_plan(plan, input_data, output_data, options.warmup,
                                 "cufftExecC2C warm-up")) {
          fatal_failure = true;
          fatal_message = "OpenACC cuFFT warm-up failed";
        }
        canonical_host(input, output);
#pragma acc update device(input_data[0 : count], output_data[0 : count])

        for (int trial = 0; trial < options.trials && !fatal_failure; ++trial) {
          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          struct timespec start;
          struct timespec end;
          double elapsed_total = 0.0;
          const char *message = "";
          canonical_host(input, output);
#pragma acc update device(input_data[0 : count], output_data[0 : count])
          bool operation_ok =
              cuda_ok(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize before timing") &&
              gpu_suite_measurement_start(start_timestamp, &start, error,
                                           sizeof(error)) == GPU_SUITE_OK;
          if (operation_ok) {
            operation_ok = execute_device_plan(plan, input_data, output_data,
                                               options.repeat, "cufftExecC2C");
          }
          if (operation_ok) {
            operation_ok = gpu_suite_measurement_end(
                               &end, end_timestamp, error, sizeof(error)) ==
                           GPU_SUITE_OK;
          }
          if (operation_ok) {
            elapsed_total = gpu_suite_clock_elapsed(&start, &end);
#pragma acc update self(output_data[0 : count])
          } else {
            message = "OpenACC cuFFT compute trial failed";
          }
          bool row_success = false;
          if (!gpu_suite_fft_bench::emit_measured(
                  writer, options, trial, start_timestamp, end_timestamp,
                  elapsed_total, output.data(), count, operation_ok, message,
                  row_success)) {
            fatal_failure = true;
            fatal_message = "result output failed";
            break;
          }
          any_failure = any_failure || !row_success;
          if (!operation_ok) {
            fatal_failure = true;
            fatal_message = message;
            if (trial + 1 < options.trials) {
              (void)gpu_suite_fft_bench::emit_unmeasured(
                  writer, options, trial + 1, false, "prior-failure", message);
            }
          }
        }
      }
      if (!cufft_ok(cufftDestroy(plan), "cufftDestroy")) {
        fatal_failure = true;
        fatal_message = "cufftDestroy failed";
      }
      if (fatal_failure) {
        std::fprintf(stderr, "%s\n", fatal_message);
        any_failure = true;
      }
    } else {
      if (!warmup_end_to_end(options, input, output, count)) {
        (void)gpu_suite_fft_bench::emit_unmeasured(
            writer, options, 0, true, "benchmark",
            "OpenACC cuFFT end-to-end warm-up failed");
        (void)gpu_suite_fft_bench::close_writer(writer);
        return EXIT_FAILURE;
      }
      for (int trial = 0; trial < options.trials; ++trial) {
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        double elapsed_total = 0.0;
        const char *message = "";
        const bool operation_ok =
            run_end_to_end_trial(options, input, output, count, start_timestamp,
                                 end_timestamp, elapsed_total, message);
        bool row_success = false;
        if (!gpu_suite_fft_bench::emit_measured(
                writer, options, trial, start_timestamp, end_timestamp,
                elapsed_total, output.data(), count, operation_ok, message,
                row_success)) {
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
    }
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
