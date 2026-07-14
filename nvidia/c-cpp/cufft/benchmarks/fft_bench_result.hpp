#ifndef GPU_SUITE_FFT_BENCH_RESULT_HPP
#define GPU_SUITE_FFT_BENCH_RESULT_HPP

#include "gpu_suite/gpu_suite.h"

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

  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (result.verification_metrics == nullptr ||
      result.verification_thresholds == nullptr) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return GPU_SUITE_OK;
  }
  if (output == nullptr || count != nfft * batch) {
    return GPU_SUITE_ERROR_INVALID;
  }
  for (std::size_t transform = 0; transform < batch; ++transform) {
    const std::size_t base = transform * nfft;
    const double expected = static_cast<double>(nfft);
    dc_relative_error = std::max(
        dc_relative_error,
        std::max(std::fabs(static_cast<double>(output[base].x) - expected),
                 std::fabs(static_cast<double>(output[base].y))) /
            expected);
    for (std::size_t frequency = 1; frequency < nfft; ++frequency) {
      const Complex &value = output[base + frequency];
      non_dc_max_abs_error =
          std::max(non_dc_max_abs_error,
                   std::max(std::fabs(static_cast<double>(value.x)),
                            std::fabs(static_cast<double>(value.y))));
    }
  }
  if (!std::isfinite(dc_relative_error) ||
      !std::isfinite(non_dc_max_abs_error)) {
    result.verification_primary_metric = "non_dc_max_abs_error";
    result.verification_status = "nonfinite";
    return GPU_SUITE_ERROR_FORMAT;
  }
  if (add_double(result.verification_metrics, "dc_relative_error",
                 dc_relative_error) != GPU_SUITE_OK ||
      add_double(result.verification_metrics, "non_dc_max_abs_error",
                 non_dc_max_abs_error) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  gpu_suite_json_value *dc = gpu_suite_json_object();
  gpu_suite_json_value *non_dc = gpu_suite_json_object();
  if (dc == nullptr || non_dc == nullptr) {
    gpu_suite_json_free(dc);
    gpu_suite_json_free(non_dc);
    return GPU_SUITE_ERROR_NOMEM;
  }
  int status = add_string(dc, "method", "relative-upper-bound");
  if (status == GPU_SUITE_OK) {
    status = add_double(dc, "upper_bound", options.rel_tolerance);
  }
  if (status == GPU_SUITE_OK) {
    status = add_string(non_dc, "method", "absolute-upper-bound");
  }
  if (status == GPU_SUITE_OK) {
    status = add_double(non_dc, "upper_bound", options.abs_tolerance);
  }
  if (status == GPU_SUITE_OK) {
    status = gpu_suite_json_object_set(result.verification_thresholds,
                                       "dc_relative_error", dc);
  }
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(dc);
    gpu_suite_json_free(non_dc);
    return status;
  }
  status = gpu_suite_json_object_set(result.verification_thresholds,
                                     "non_dc_max_abs_error", non_dc);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(non_dc);
    return status;
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
  if (gpu_suite_result_init(&result) != GPU_SUITE_OK ||
      gpu_suite_result_apply_options(&result, &options, error, sizeof(error)) !=
          GPU_SUITE_OK) {
    std::fprintf(stderr, "could not initialize result: %s\n", error);
    return false;
  }
  result.trial = trial;
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
    if (verification_status == GPU_SUITE_OK &&
        (std::strcmp(result.verification_status, "pass") == 0 ||
         std::strcmp(result.verification_status, "skipped") == 0)) {
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
      result.message = verification_status == GPU_SUITE_OK
                           ? "cuFFT verification threshold exceeded"
                           : "cuFFT verification produced invalid metrics";
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
