#include "gpu_suite/benchmark.h"

#include <stdio.h>
#include <string.h>

int gpu_suite_benchmark_writer_open(gpu_suite_benchmark_writer *writer,
                                    const gpu_suite_options *options,
                                    char *error, size_t error_size) {
  if (writer == NULL || options == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  memset(writer, 0, sizeof(*writer));
  writer->include_header = true;
  writer->format = options->format;
  return gpu_suite_output_open(options->output, &writer->stream,
                               &writer->must_close, error, error_size);
}

int gpu_suite_benchmark_writer_write(gpu_suite_benchmark_writer *writer,
                                     gpu_suite_result *result, char *error,
                                     size_t error_size) {
  int status;
  if (writer == NULL || writer->stream == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  status =
      gpu_suite_result_write(writer->stream, writer->format,
                             writer->include_header, result, error, error_size);
  if (status == GPU_SUITE_OK) {
    writer->include_header = false;
  }
  return status;
}

int gpu_suite_benchmark_writer_close(gpu_suite_benchmark_writer *writer,
                                     char *error, size_t error_size) {
  int status;
  if (writer == NULL || writer->stream == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  status = gpu_suite_output_close(writer->stream, writer->must_close, error,
                                  error_size);
  writer->stream = NULL;
  return status;
}

int gpu_suite_benchmark_result_init(gpu_suite_result *result,
                                    const gpu_suite_options *options, int trial,
                                    char *error, size_t error_size) {
  int status = gpu_suite_result_init(result);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  status = gpu_suite_result_apply_options(result, options, error, error_size);
  if (status != GPU_SUITE_OK) {
    gpu_suite_result_destroy(result);
    return status;
  }
  result->trial = trial;
  return GPU_SUITE_OK;
}

int gpu_suite_benchmark_emit_unmeasured(gpu_suite_benchmark_writer *writer,
                                        const gpu_suite_options *options,
                                        int first_trial, bool first_attempted,
                                        const char *first_origin,
                                        const char *message, char *error,
                                        size_t error_size) {
  int trial;
  for (trial = first_trial; trial < options->trials; ++trial) {
    gpu_suite_result result;
    int status = gpu_suite_benchmark_result_init(&result, options, trial, error,
                                                 error_size);
    if (status != GPU_SUITE_OK) {
      return status;
    }
    result.attempted = trial == first_trial ? first_attempted : false;
    result.failure_origin =
        trial == first_trial ? first_origin : "prior-failure";
    result.verification_status = "skipped";
    result.exit_code = result.attempted ? gpu_suite_optional_int_value(1)
                                        : gpu_suite_optional_int_null();
    result.status = result.attempted ? "failure" : "skipped";
    result.message = message;
    status =
        gpu_suite_benchmark_writer_write(writer, &result, error, error_size);
    gpu_suite_result_destroy(&result);
    if (status != GPU_SUITE_OK) {
      return status;
    }
  }
  return GPU_SUITE_OK;
}

static int object_add(gpu_suite_json_value *object, const char *key,
                      gpu_suite_json_value *value) {
  int status;
  if (value == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  status = gpu_suite_json_object_set(object, key, value);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(value);
  }
  return status;
}

int gpu_suite_json_add_double(gpu_suite_json_value *object, const char *key,
                              double value) {
  return object_add(object, key, gpu_suite_json_double(value));
}

int gpu_suite_json_add_int(gpu_suite_json_value *object, const char *key,
                           int64_t value) {
  return object_add(object, key, gpu_suite_json_int(value));
}

int gpu_suite_json_add_string(gpu_suite_json_value *object, const char *key,
                              const char *value) {
  return object_add(object, key, gpu_suite_json_string(value));
}
