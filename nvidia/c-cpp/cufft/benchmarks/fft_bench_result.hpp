#ifndef GPU_SUITE_FFT_BENCH_RESULT_HPP
#define GPU_SUITE_FFT_BENCH_RESULT_HPP

#include "gpu_suite/gpu_suite.h"
#if defined(GPU_SUITE_HAVE_CUDA_RUNTIME_METADATA)
#include "gpu_suite/cuda_metadata.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace gpu_suite_fft_bench {

struct Writer {
  FILE *stream = nullptr;
  bool must_close = false;
  bool header = true;
};

inline bool open_writer(const gpu_suite_options &options, Writer &writer) {
  char error[256] = {0};
  if (gpu_suite_output_open(options.output, &writer.stream, &writer.must_close,
                            error, sizeof(error)) != GPU_SUITE_OK) {
    std::fprintf(stderr, "could not open output: %s\n", error);
    return false;
  }
  return true;
}

inline bool close_writer(Writer &writer) {
  char error[256] = {0};
  if (writer.stream == nullptr) {
    return true;
  }
  if (gpu_suite_output_close(writer.stream, writer.must_close, error,
                             sizeof(error)) != GPU_SUITE_OK) {
    std::fprintf(stderr, "could not close output: %s\n", error);
    return false;
  }
  writer.stream = nullptr;
  return true;
}

inline bool write(Writer &writer, gpu_suite_output_format format,
                  gpu_suite_result &result) {
  char error[256] = {0};
  if (gpu_suite_result_write(writer.stream, format, writer.header, &result,
                             error, sizeof(error)) != GPU_SUITE_OK) {
    std::fprintf(stderr, "result output failed: %s\n", error);
    return false;
  }
  writer.header = false;
  return true;
}

inline int add_double(gpu_suite_json_value *object, const char *key,
                      double value) {
  gpu_suite_json_value *number = gpu_suite_json_double(value);
  if (number == nullptr) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  const int status = gpu_suite_json_object_set(object, key, number);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(number);
  }
  return status;
}

inline int add_string(gpu_suite_json_value *object, const char *key,
                      const char *value) {
  gpu_suite_json_value *string = gpu_suite_json_string(value);
  if (string == nullptr) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  const int status = gpu_suite_json_object_set(object, key, string);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(string);
  }
  return status;
}

template <typename Complex>
int verify(gpu_suite_result &result, const gpu_suite_options &options,
           const Complex *output, std::size_t count) {
  double dc_relative_error = 0.0;
  double non_dc_max_abs_error = 0.0;
  const std::size_t nfft = static_cast<std::size_t>(options.size);
  const std::size_t batch = static_cast<std::size_t>(options.batch);
  std::size_t expected_count = 0;

  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return GPU_SUITE_OK;
  }
  if (output == nullptr ||
      !gpu_suite_checked_mul_size(nfft, batch, &expected_count) ||
      count != expected_count || nfft == 0U) {
    return GPU_SUITE_ERROR_INVALID;
  }
  if (gpu_suite_verification_add_upper_bound_threshold(
          &result, "dc_relative_error", "relative-upper-bound",
          options.rel_tolerance) != GPU_SUITE_OK ||
      gpu_suite_verification_add_upper_bound_threshold(
          &result, "non_dc_max_abs_error", "absolute-upper-bound",
          options.abs_tolerance) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  bool finite = true;
  for (std::size_t transform = 0; transform < batch; ++transform) {
    const std::size_t base = transform * nfft;
    const double expected = static_cast<double>(nfft);
    const double dc_real = static_cast<double>(output[base].x);
    const double dc_imag = static_cast<double>(output[base].y);
    double real_error = 0.0;
    double imag_error = 0.0;
    double transform_error = 0.0;
    if (!gpu_suite_finite_absolute_error(dc_real, expected, &real_error) ||
        !gpu_suite_finite_absolute_error(dc_imag, 0.0, &imag_error) ||
        !gpu_suite_finite_max_update(real_error, &transform_error) ||
        !gpu_suite_finite_max_update(imag_error, &transform_error)) {
      finite = false;
    } else {
      transform_error /= expected;
      if (!gpu_suite_finite_max_update(transform_error, &dc_relative_error))
        finite = false;
    }
    for (std::size_t frequency = 1; frequency < nfft; ++frequency) {
      const Complex &value = output[base + frequency];
      double real_magnitude = 0.0;
      double imag_magnitude = 0.0;
      if (!gpu_suite_finite_absolute_error(static_cast<double>(value.x), 0.0,
                                           &real_magnitude) ||
          !gpu_suite_finite_absolute_error(static_cast<double>(value.y), 0.0,
                                           &imag_magnitude) ||
          !gpu_suite_finite_max_update(real_magnitude,
                                       &non_dc_max_abs_error) ||
          !gpu_suite_finite_max_update(imag_magnitude,
                                       &non_dc_max_abs_error))
        finite = false;
    }
  }
  if (!finite) {
    if (gpu_suite_json_add_null(result.verification_metrics,
                                "dc_relative_error") != GPU_SUITE_OK ||
        gpu_suite_json_add_null(result.verification_metrics,
                                "non_dc_max_abs_error") != GPU_SUITE_OK)
      return GPU_SUITE_ERROR_NOMEM;
    result.verification_primary_metric = "non_dc_max_abs_error";
    result.verification_status = "nonfinite";
    return GPU_SUITE_OK;
  }
  if (add_double(result.verification_metrics, "dc_relative_error",
                 dc_relative_error) != GPU_SUITE_OK ||
      add_double(result.verification_metrics, "non_dc_max_abs_error",
                 non_dc_max_abs_error) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  result.verification_primary_metric = "non_dc_max_abs_error";
  result.verification_status =
      dc_relative_error <= options.rel_tolerance &&
              non_dc_max_abs_error <= options.abs_tolerance
          ? "pass"
          : "failure";
  return GPU_SUITE_OK;
}

inline bool initialize_result(gpu_suite_result &result,
                              const gpu_suite_options &options, int trial) {
  char error[256] = {0};
  if (gpu_suite_result_init(&result) != GPU_SUITE_OK) {
    std::fprintf(stderr, "could not initialize result: %s\n", error);
    return false;
  }
  if (gpu_suite_result_apply_options(&result, &options, error, sizeof(error)) !=
      GPU_SUITE_OK) {
    gpu_suite_result_destroy(&result);
    std::fprintf(stderr, "could not initialize result: %s\n", error);
    return false;
  }
  result.trial = trial;
#if defined(GPU_SUITE_HAVE_CUDA_RUNTIME_METADATA)
  if (options.implementation != GPU_SUITE_IMPLEMENTATION_CPU &&
      !gpu_suite::apply_cuda_runtime_metadata(result, options.device)) {
    gpu_suite_result_destroy(&result);
    std::fprintf(stderr, "could not query CUDA device/cuFFT metadata\n");
    return false;
  }
#endif
  return true;
}

inline bool emit_unmeasured(Writer &writer, const gpu_suite_options &options,
                            int first_trial, bool first_attempted,
                            const char *first_origin, const char *message) {
  for (int trial = first_trial; trial < options.trials; ++trial) {
    gpu_suite_result result;
    if (!initialize_result(result, options, trial)) {
      return false;
    }
    result.attempted = trial == first_trial ? first_attempted : false;
    result.failure_origin =
        trial == first_trial ? first_origin : "prior-failure";
    result.verification_status = "skipped";
    result.exit_code = result.attempted ? gpu_suite_optional_int_value(1)
                                        : gpu_suite_optional_int_null();
    result.status = result.attempted ? "failure" : "skipped";
    result.message = message;
    const bool written = write(writer, options.format, result);
    gpu_suite_result_destroy(&result);
    if (!written) {
      return false;
    }
  }
  return true;
}

template <typename Complex>
bool emit_measured(Writer &writer, const gpu_suite_options &options, int trial,
                   const char *start_timestamp, const char *end_timestamp,
                   double elapsed_total, const Complex *output,
                   std::size_t count, bool operation_ok,
                   const char *operation_message, bool &row_success) {
  gpu_suite_result result;
  if (!initialize_result(result, options, trial)) {
    return false;
  }
  if (operation_ok) {
    result.measurement_start_timestamp = start_timestamp;
    result.measurement_end_timestamp = end_timestamp;
    result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed_total);
    result.elapsed_sec = gpu_suite_optional_double_value(
        elapsed_total / static_cast<double>(options.repeat));
    const int verification_status = verify(result, options, output, count);
    if (verification_status != GPU_SUITE_OK) {
      std::fprintf(stderr, "cuFFT verification result construction failed\n");
      gpu_suite_result_destroy(&result);
      row_success = false;
      return false;
    }
    if (std::strcmp(result.verification_status, "pass") == 0 ||
        std::strcmp(result.verification_status, "skipped") == 0) {
      result.attempted = true;
      result.failure_origin = nullptr;
      result.exit_code = gpu_suite_optional_int_value(0);
      result.status = "success";
      result.message = "";
      row_success = true;
    } else {
      result.attempted = true;
      result.failure_origin = "verification";
      result.exit_code = gpu_suite_optional_int_value(1);
      result.status = "failure";
      result.message = std::strcmp(result.verification_status, "nonfinite") == 0
                           ? "cuFFT verification produced nonfinite output"
                           : "cuFFT verification threshold exceeded";
      row_success = false;
    }
  } else {
    result.attempted = true;
    result.failure_origin = "benchmark";
    result.verification_status = "skipped";
    result.exit_code = gpu_suite_optional_int_value(1);
    result.status = "failure";
    result.message = operation_message;
    row_success = false;
  }
  const bool written = write(writer, options.format, result);
  gpu_suite_result_destroy(&result);
  return written;
}

} // namespace gpu_suite_fft_bench

#endif
