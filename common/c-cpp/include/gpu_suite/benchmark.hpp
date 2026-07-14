#ifndef GPU_SUITE_BENCHMARK_HPP
#define GPU_SUITE_BENCHMARK_HPP

#include "gpu_suite/gpu_suite.h"

#include <cstdio>

namespace gpu_suite {

class ResultWriter {
public:
  ResultWriter() = default;
  ResultWriter(const ResultWriter &) = delete;
  ResultWriter &operator=(const ResultWriter &) = delete;

  bool open(const gpu_suite_options &options) {
    char error[256] = {0};
    format_ = options.format;
    if (gpu_suite_output_open(options.output, &stream_, &must_close_, error,
                              sizeof(error)) != GPU_SUITE_OK) {
      std::fprintf(stderr, "could not open output: %s\n", error);
      return false;
    }
    return true;
  }

  bool write(gpu_suite_result &result) {
    char error[256] = {0};
    if (gpu_suite_result_write(stream_, format_, header_, &result, error,
                               sizeof(error)) != GPU_SUITE_OK) {
      std::fprintf(stderr, "result output failed: %s\n", error);
      return false;
    }
    header_ = false;
    return true;
  }

  bool close() {
    char error[256] = {0};
    if (stream_ == nullptr) {
      return true;
    }
    if (gpu_suite_output_close(stream_, must_close_, error, sizeof(error)) !=
        GPU_SUITE_OK) {
      std::fprintf(stderr, "could not close output: %s\n", error);
      stream_ = nullptr;
      return false;
    }
    stream_ = nullptr;
    return true;
  }

  ~ResultWriter() {
    if (stream_ != nullptr) {
      char error[1] = {0};
      (void)gpu_suite_output_close(stream_, must_close_, error, sizeof(error));
    }
  }

private:
  FILE *stream_ = nullptr;
  bool must_close_ = false;
  bool header_ = true;
  gpu_suite_output_format format_ = GPU_SUITE_FORMAT_JSONL;
};

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

inline bool emit_unmeasured(ResultWriter &writer,
                            const gpu_suite_options &options, int first_trial,
                            bool attempted, const char *origin,
                            const char *message) {
  for (int trial = first_trial; trial < options.trials; ++trial) {
    gpu_suite_result result;
    if (!initialize_result(result, options, trial)) {
      return false;
    }
    result.attempted = trial == first_trial ? attempted : false;
    result.failure_origin = trial == first_trial ? origin : "prior-failure";
    result.verification_status = "skipped";
    result.exit_code = result.attempted ? gpu_suite_optional_int_value(1)
                                        : gpu_suite_optional_int_null();
    result.status = result.attempted ? "failure" : "skipped";
    result.message = message;
    const bool success = writer.write(result);
    gpu_suite_result_destroy(&result);
    if (!success) {
      return false;
    }
  }
  return true;
}

inline int json_add_double(gpu_suite_json_value *object, const char *key,
                           double value) {
  gpu_suite_json_value *item = gpu_suite_json_double(value);
  if (item == nullptr) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  const int status = gpu_suite_json_object_set(object, key, item);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(item);
  }
  return status;
}

inline int json_add_string(gpu_suite_json_value *object, const char *key,
                           const char *value) {
  gpu_suite_json_value *item = gpu_suite_json_string(value);
  if (item == nullptr) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  const int status = gpu_suite_json_object_set(object, key, item);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(item);
  }
  return status;
}

} // namespace gpu_suite

#endif
