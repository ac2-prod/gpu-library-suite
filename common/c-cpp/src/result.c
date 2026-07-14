#include "gpu_suite/result.h"

#include "gpu_suite/checked.h"
#include "gpu_suite/clock.h"
#include "gpu_suite/metadata.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char ZERO_SHA256[] =
    "0000000000000000000000000000000000000000000000000000000000000000";

static const char CSV_HEADER[] =
    "result_schema_version,run_id,record_timestamp,measurement_start_timestamp,"
    "measurement_end_timestamp,system_label,wave,node_index,hostname,block_id,"
    "scheduler,scheduler_job_id,implementation_order,benchmark,implementation,"
    "scope,problem_size,secondary_size,parameters,precision,cpu_backend,"
    "cpu_backend_role,series_role,cpu_threads_requested,cpu_threads_effective,"
    "cpu_parallelism,warmup,repeat,trial,attempted,failure_origin,"
    "elapsed_total_sec,elapsed_sec,clock_id,clock_resolution_sec,"
    "verification_metrics,verification_thresholds,verification_primary_metric,"
    "verification_status,getrf_info,getrs_info,device_id,gpu_name,gpu_uuid,"
    "cuda_driver_version,compiler,compiler_version,global_configure_flags,"
    "library_name,library_version,cuda_runtime_version,"
    "git_metadata_available,git_commit,git_dirty,git_diff_sha256,"
    "source_snapshot_sha256,config_sha256,runtime_environment_sha256,"
    "binary_sha256,exit_code,status,message";

static void set_error(char *error, size_t error_size, const char *message) {
  if (error != NULL && error_size > 0U) {
    (void)snprintf(error, error_size, "%s", message);
  }
}

static const char *environment_or_null(const char *name) {
  const char *value = getenv(name);
  return value != NULL && value[0] != '\0' ? value : NULL;
}

static const char *environment_or_default(const char *name,
                                          const char *fallback) {
  const char *value = environment_or_null(name);
  return value == NULL ? fallback : value;
}

gpu_suite_optional_int gpu_suite_optional_int_null(void) {
  gpu_suite_optional_int result = {true, 0};
  return result;
}

gpu_suite_optional_int gpu_suite_optional_int_value(int64_t value) {
  gpu_suite_optional_int result = {false, value};
  return result;
}

gpu_suite_optional_double gpu_suite_optional_double_null(void) {
  gpu_suite_optional_double result = {true, 0.0};
  return result;
}

gpu_suite_optional_double gpu_suite_optional_double_value(double value) {
  gpu_suite_optional_double result = {false, value};
  return result;
}

static int u64_to_i64(uint64_t value, int64_t *result) {
  if (value > (uint64_t)INT64_MAX || result == NULL) {
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  *result = (int64_t)value;
  return GPU_SUITE_OK;
}

static int object_take(gpu_suite_json_value *object, const char *key,
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

static int object_int(gpu_suite_json_value *object, const char *key,
                      uint64_t value) {
  int64_t converted;
  if (u64_to_i64(value, &converted) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  return object_take(object, key, gpu_suite_json_int(converted));
}

static int build_implementation_order(const char *text,
                                      gpu_suite_json_value **output) {
  gpu_suite_json_value *array;
  char copy[GPU_SUITE_ORDER_CAPACITY];
  char *save = NULL;
  char *token;
  int status;

  if (!gpu_suite_implementation_order_validate(text) || output == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  (void)snprintf(copy, sizeof(copy), "%s", text);
  array = gpu_suite_json_array();
  if (array == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  token = strtok_r(copy, ",", &save);
  while (token != NULL) {
    gpu_suite_json_value *item = gpu_suite_json_string(token);
    if (item == NULL) {
      gpu_suite_json_free(array);
      return GPU_SUITE_ERROR_NOMEM;
    }
    status = gpu_suite_json_array_append(array, item);
    if (status != GPU_SUITE_OK) {
      gpu_suite_json_free(item);
      gpu_suite_json_free(array);
      return status;
    }
    token = strtok_r(NULL, ",", &save);
  }
  *output = array;
  return GPU_SUITE_OK;
}

static uint64_t integer_square_root(uint64_t value) {
  uint64_t low = 0U;
  uint64_t high = value < UINT64_C(4294967295) ? value : UINT64_C(4294967295);
  uint64_t answer = 0U;
  while (low <= high) {
    uint64_t middle = low + (high - low) / 2U;
    if (middle == 0U || middle <= value / middle) {
      answer = middle;
      low = middle + 1U;
    } else {
      high = middle - 1U;
    }
  }
  return answer;
}

static int sparse_sizes(const gpu_suite_options *options, uint64_t *nx,
                        uint64_t *ny, uint64_t *count, uint64_t *nnz) {
  uint64_t product;
  uint64_t root;
  uint64_t edges;

  if (options->size_set) {
    root = integer_square_root(options->size);
    *nx = root;
    *ny = root;
    product = options->size;
  } else {
    *nx = options->nx;
    *ny = options->ny;
    if (*nx != 0U && *ny > UINT64_MAX / *nx) {
      return GPU_SUITE_ERROR_OVERFLOW;
    }
    product = *nx * *ny;
  }
  if (product > UINT64_MAX / 5U || *nx > UINT64_MAX / 2U ||
      *ny > UINT64_MAX / 2U) {
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  edges = 2U * *nx + 2U * *ny;
  if (5U * product < edges) {
    return GPU_SUITE_ERROR_INVALID;
  }
  *count = product;
  *nnz = 5U * product - edges;
  return GPU_SUITE_OK;
}

static int build_parameters(const gpu_suite_options *options,
                            gpu_suite_json_value **output,
                            gpu_suite_optional_int *problem_size,
                            gpu_suite_optional_int *secondary_size) {
  gpu_suite_json_value *parameters = gpu_suite_json_object();
  int status = GPU_SUITE_OK;
  int64_t converted;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t nx;
  uint64_t ny;
  uint64_t count;
  uint64_t nnz;

  if (parameters == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
#define PARAMETER_TAKE(key, expression)                                        \
  do {                                                                         \
    status = object_take(parameters, (key), (expression));                     \
    if (status != GPU_SUITE_OK) {                                              \
      goto fail;                                                               \
    }                                                                          \
  } while (0)
#define PARAMETER_INT(key, value)                                              \
  do {                                                                         \
    status = object_int(parameters, (key), (value));                           \
    if (status != GPU_SUITE_OK) {                                              \
      goto fail;                                                               \
    }                                                                          \
  } while (0)

  *problem_size = gpu_suite_optional_int_null();
  *secondary_size = gpu_suite_optional_int_null();
  switch (options->benchmark) {
  case GPU_SUITE_BENCHMARK_CUFFT:
    PARAMETER_INT("nfft", options->size);
    PARAMETER_INT("batch", options->batch);
    PARAMETER_TAKE("transform", gpu_suite_json_string(options->transform));
    if (u64_to_i64(options->size, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *problem_size = gpu_suite_optional_int_value(converted);
    if (u64_to_i64(options->batch, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *secondary_size = gpu_suite_optional_int_value(converted);
    break;
  case GPU_SUITE_BENCHMARK_CUBLAS:
    m = options->size_set ? options->size : options->m;
    n = options->size_set ? options->size : options->n;
    k = options->size_set ? options->size : options->k;
    PARAMETER_INT("m", m);
    PARAMETER_INT("n", n);
    PARAMETER_INT("k", k);
    PARAMETER_TAKE("alpha", gpu_suite_json_double(options->alpha));
    PARAMETER_TAKE("beta", gpu_suite_json_double(options->beta));
    if (m == n && n == k) {
      if (u64_to_i64(m, &converted) != GPU_SUITE_OK) {
        status = GPU_SUITE_ERROR_OVERFLOW;
        goto fail;
      }
      *problem_size = gpu_suite_optional_int_value(converted);
    }
    break;
  case GPU_SUITE_BENCHMARK_CUSPARSE:
    status = sparse_sizes(options, &nx, &ny, &count, &nnz);
    if (status != GPU_SUITE_OK) {
      goto fail;
    }
    PARAMETER_INT("nx", nx);
    PARAMETER_INT("ny", ny);
    PARAMETER_TAKE("alpha", gpu_suite_json_double(options->alpha));
    PARAMETER_TAKE("beta", gpu_suite_json_double(options->beta));
    if (u64_to_i64(count, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *problem_size = gpu_suite_optional_int_value(converted);
    if (u64_to_i64(nnz, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *secondary_size = gpu_suite_optional_int_value(converted);
    break;
  case GPU_SUITE_BENCHMARK_CUSOLVER:
    PARAMETER_INT("n", options->size);
    PARAMETER_INT("nrhs", options->nrhs);
    if (u64_to_i64(options->size, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *problem_size = gpu_suite_optional_int_value(converted);
    if (u64_to_i64(options->nrhs, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *secondary_size = gpu_suite_optional_int_value(converted);
    break;
  case GPU_SUITE_BENCHMARK_CURAND:
    PARAMETER_INT("size", options->size);
    PARAMETER_TAKE("generator", gpu_suite_json_string(options->generator));
    PARAMETER_TAKE("distribution",
                   gpu_suite_json_string(options->distribution));
    PARAMETER_INT("seed", options->seed);
    PARAMETER_INT("offset", options->offset);
    PARAMETER_TAKE("order", gpu_suite_json_string(options->order));
    if (u64_to_i64(options->size, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *problem_size = gpu_suite_optional_int_value(converted);
    break;
  case GPU_SUITE_BENCHMARK_THRUST:
    PARAMETER_INT("size", options->size);
    PARAMETER_TAKE("operation", gpu_suite_json_string(options->operation));
    if (u64_to_i64(options->size, &converted) != GPU_SUITE_OK) {
      status = GPU_SUITE_ERROR_OVERFLOW;
      goto fail;
    }
    *problem_size = gpu_suite_optional_int_value(converted);
    break;
  }
#undef PARAMETER_TAKE
#undef PARAMETER_INT
  *output = parameters;
  return GPU_SUITE_OK;

fail:
#undef PARAMETER_TAKE
#undef PARAMETER_INT
  gpu_suite_json_free(parameters);
  return status;
}

int gpu_suite_result_init(gpu_suite_result *result) {
  char error[128];
  double resolution = 0.0;

  if (result == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  memset(result, 0, sizeof(*result));
  result->result_schema_version = 1;
  result->problem_size = gpu_suite_optional_int_null();
  result->secondary_size = gpu_suite_optional_int_null();
  result->cpu_threads_requested = gpu_suite_optional_int_null();
  result->cpu_threads_effective = gpu_suite_optional_int_null();
  result->elapsed_total_sec = gpu_suite_optional_double_null();
  result->elapsed_sec = gpu_suite_optional_double_null();
  result->getrf_info = gpu_suite_optional_int_null();
  result->getrs_info = gpu_suite_optional_int_null();
  result->device_id = gpu_suite_optional_int_null();
  result->exit_code = gpu_suite_optional_int_null();
  result->implementation_order = gpu_suite_json_array();
  result->parameters = gpu_suite_json_object();
  result->verification_metrics = gpu_suite_json_object();
  result->verification_thresholds = gpu_suite_json_object();
  if (result->implementation_order == NULL || result->parameters == NULL ||
      result->verification_metrics == NULL ||
      result->verification_thresholds == NULL) {
    gpu_suite_result_destroy(result);
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (gethostname(result->hostname_storage, sizeof(result->hostname_storage)) !=
      0) {
    (void)snprintf(result->hostname_storage, sizeof(result->hostname_storage),
                   "unknown");
  }
  result->hostname_storage[sizeof(result->hostname_storage) - 1U] = '\0';
  result->hostname = result->hostname_storage;
  if (gpu_suite_utc_timestamp(result->record_timestamp_storage, error,
                              sizeof(error)) != GPU_SUITE_OK) {
    gpu_suite_result_destroy(result);
    return GPU_SUITE_ERROR_IO;
  }
  result->record_timestamp = result->record_timestamp_storage;
  result->clock_id = "CLOCK_MONOTONIC";
  if (gpu_suite_clock_resolution(&resolution, error, sizeof(error)) !=
      GPU_SUITE_OK) {
    gpu_suite_result_destroy(result);
    return GPU_SUITE_ERROR_IO;
  }
  result->clock_resolution_sec = resolution;
  result->series_role = "primary";
  result->verification_status = "skipped";
  result->attempted = true;
  result->failure_origin = "benchmark";
  result->status = "failure";
  result->message = "result not finalized";
  result->config_sha256 = ZERO_SHA256;
  result->runtime_environment_sha256 = ZERO_SHA256;
  result->binary_sha256 = ZERO_SHA256;
  result->git_diff_sha256 = NULL;
  result->source_snapshot_sha256 = NULL;
  return GPU_SUITE_OK;
}

void gpu_suite_result_destroy(gpu_suite_result *result) {
  if (result == NULL) {
    return;
  }
  gpu_suite_json_free(result->implementation_order);
  gpu_suite_json_free(result->parameters);
  gpu_suite_json_free(result->verification_metrics);
  gpu_suite_json_free(result->verification_thresholds);
  result->implementation_order = NULL;
  result->parameters = NULL;
  result->verification_metrics = NULL;
  result->verification_thresholds = NULL;
}

int gpu_suite_result_apply_options(gpu_suite_result *result,
                                   const gpu_suite_options *options,
                                   char *error, size_t error_size) {
  gpu_suite_json_value *order = NULL;
  gpu_suite_json_value *parameters = NULL;
  gpu_suite_optional_int problem_size;
  gpu_suite_optional_int secondary_size;
  int status;
  int written;

  if (result == NULL || options == NULL) {
    set_error(error, error_size, "result or options are null");
    return GPU_SUITE_ERROR_INVALID;
  }
  status = gpu_suite_options_validate(options, error, error_size);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  status = build_implementation_order(options->implementation_order, &order);
  if (status != GPU_SUITE_OK) {
    set_error(error, error_size, "could not create implementation order");
    return status;
  }
  status =
      build_parameters(options, &parameters, &problem_size, &secondary_size);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(order);
    set_error(error, error_size, "could not create normalized parameters");
    return status;
  }
  gpu_suite_json_free(result->implementation_order);
  gpu_suite_json_free(result->parameters);
  result->implementation_order = order;
  result->parameters = parameters;
  result->run_id = options->run_id;
  result->system_label = options->system_label;
  result->wave = options->wave;
  result->node_index = options->node_index;
  written =
      snprintf(result->block_id_storage, sizeof(result->block_id_storage),
               "%s|%d|%s", result->run_id, result->wave, result->hostname);
  if (written < 0 || (size_t)written >= sizeof(result->block_id_storage)) {
    set_error(error, error_size, "block_id is too long");
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  result->block_id = result->block_id_storage;
  result->scheduler = environment_or_null("GPU_SUITE_SCHEDULER");
  result->scheduler_job_id = environment_or_null("GPU_SUITE_SCHEDULER_JOB_ID");
  result->benchmark = gpu_suite_benchmark_name(options->benchmark);
  result->implementation =
      gpu_suite_implementation_name(options->implementation);
  result->scope = gpu_suite_scope_name(options->scope);
  result->problem_size = problem_size;
  result->secondary_size = secondary_size;
  result->precision =
      options->benchmark == GPU_SUITE_BENCHMARK_CUFFT ? "fp32" : "fp64";
  result->warmup = options->warmup;
  result->repeat = options->repeat;
  result->trial = 0;

  if (options->implementation == GPU_SUITE_IMPLEMENTATION_CPU) {
    result->cpu_backend = options->cpu_backend;
    result->cpu_backend_role = options->cpu_backend_role;
    result->series_role = options->series_role;
    result->cpu_threads_requested =
        gpu_suite_optional_int_value(options->cpu_threads);
    if (options->cpu_threads_effective > 0) {
      result->cpu_threads_effective =
          gpu_suite_optional_int_value(options->cpu_threads_effective);
    } else {
      result->cpu_threads_effective = gpu_suite_optional_int_null();
    }
    result->cpu_parallelism = options->cpu_parallelism;
    result->device_id = gpu_suite_optional_int_null();
    result->library_name = options->cpu_backend;
  } else {
    result->cpu_backend = NULL;
    result->cpu_backend_role = NULL;
    result->series_role = options->series_role;
    result->cpu_threads_requested = gpu_suite_optional_int_null();
    result->cpu_threads_effective = gpu_suite_optional_int_null();
    result->cpu_parallelism = NULL;
    result->device_id = gpu_suite_optional_int_value(options->device);
    result->library_name = result->benchmark;
  }
  gpu_suite_result_apply_build_metadata(result, options->implementation);
  gpu_suite_result_apply_provenance_environment(result);
  return GPU_SUITE_OK;
}

void gpu_suite_result_apply_build_metadata(gpu_suite_result *result,
                                           gpu_suite_implementation impl) {
  if (result == NULL) {
    return;
  }
  result->compiler = gpu_suite_build_benchmark_compiler(impl, result->benchmark);
  result->compiler_version =
      gpu_suite_build_benchmark_compiler_version(impl, result->benchmark);
  result->global_configure_flags =
      gpu_suite_build_benchmark_global_configure_flags(impl,
                                                       result->benchmark);
  result->git_metadata_available =
      gpu_suite_build_git_metadata_available();
  result->git_commit = gpu_suite_build_git_commit();
  result->git_dirty = gpu_suite_build_git_dirty();
}

void gpu_suite_result_apply_provenance_environment(gpu_suite_result *result) {
  if (result == NULL) {
    return;
  }
  result->gpu_name = environment_or_null("GPU_SUITE_GPU_NAME");
  result->gpu_uuid = environment_or_null("GPU_SUITE_GPU_UUID");
  result->cuda_driver_version =
      environment_or_null("GPU_SUITE_CUDA_DRIVER_VERSION");
  result->library_version = environment_or_null("GPU_SUITE_LIBRARY_VERSION");
  result->cuda_runtime_version =
      environment_or_null("GPU_SUITE_CUDA_RUNTIME_VERSION");
  result->git_diff_sha256 = environment_or_null("GPU_SUITE_GIT_DIFF_SHA256");
  result->source_snapshot_sha256 =
      environment_or_null("GPU_SUITE_SOURCE_SNAPSHOT_SHA256");
  result->config_sha256 =
      environment_or_default("GPU_SUITE_CONFIG_SHA256", ZERO_SHA256);
  result->runtime_environment_sha256 = environment_or_default(
      "GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256", ZERO_SHA256);
  result->binary_sha256 =
      environment_or_default("GPU_SUITE_BINARY_SHA256", ZERO_SHA256);
}

static bool timestamp_valid(const char *value) {
  static const size_t digit_positions[] = {0U,  1U,  2U,  3U,  5U,  6U,
                                           8U,  9U,  11U, 12U, 14U, 15U,
                                           17U, 18U, 20U, 21U, 22U};
  size_t index;
  if (value == NULL || strlen(value) != 24U || value[4] != '-' ||
      value[7] != '-' || value[10] != 'T' || value[13] != ':' ||
      value[16] != ':' || value[19] != '.' || value[23] != 'Z') {
    return false;
  }
  for (index = 0U; index < sizeof(digit_positions) / sizeof(digit_positions[0]);
       ++index) {
    char character = value[digit_positions[index]];
    if (character < '0' || character > '9') {
      return false;
    }
  }
  return true;
}

static bool sha256_valid(const char *value) {
  size_t index;
  if (value == NULL || strlen(value) != 64U) {
    return false;
  }
  for (index = 0U; index < 64U; ++index) {
    if (!((value[index] >= '0' && value[index] <= '9') ||
          (value[index] >= 'a' && value[index] <= 'f'))) {
      return false;
    }
  }
  return true;
}

static bool string_in(const char *value, const char *const *allowed,
                      size_t count) {
  size_t index;
  if (value == NULL) {
    return false;
  }
  for (index = 0U; index < count; ++index) {
    if (strcmp(value, allowed[index]) == 0) {
      return true;
    }
  }
  return false;
}

int gpu_suite_result_validate(const gpu_suite_result *result, char *error,
                              size_t error_size) {
  static const char *const statuses[] = {"success", "failure", "skipped"};
  static const char *const origins[] = {"benchmark",     "verification",
                                        "subprocess",    "output-validation",
                                        "prior-failure", "prerequisite"};
  static const char *const verification_statuses[] = {"pass", "failure",
                                                      "skipped", "nonfinite"};
  static const char *const series_roles[] = {"primary", "auxiliary"};
  static const char *const backend_roles[] = {"production", "reference"};
  static const char *const parallelism_values[] = {"serial", "threaded",
                                                    "unknown"};
  char
      expected_block[GPU_SUITE_RUN_ID_CAPACITY + GPU_SUITE_LABEL_CAPACITY + 32];
  int written;

  if (result == NULL || result->result_schema_version != 1 ||
      !gpu_suite_run_id_validate(result->run_id) ||
      !timestamp_valid(result->record_timestamp) ||
      result->system_label == NULL || result->system_label[0] == '\0' ||
      result->wave < 0 || result->node_index < 0 || result->hostname == NULL ||
      result->hostname[0] == '\0' || result->block_id == NULL ||
      result->implementation_order == NULL || result->parameters == NULL ||
      result->verification_metrics == NULL ||
      result->verification_thresholds == NULL) {
    set_error(error, error_size, "missing or invalid required result field");
    return GPU_SUITE_ERROR_INVALID;
  }
  written = snprintf(expected_block, sizeof(expected_block), "%s|%d|%s",
                     result->run_id, result->wave, result->hostname);
  if (written < 0 || (size_t)written >= sizeof(expected_block) ||
      strcmp(expected_block, result->block_id) != 0) {
    set_error(error, error_size, "block_id does not match its components");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->benchmark == NULL || result->implementation == NULL ||
      result->scope == NULL || result->precision == NULL ||
      result->series_role == NULL || result->warmup < 0 ||
      result->repeat <= 0 || result->trial < 0 || result->clock_id == NULL ||
      strcmp(result->clock_id, "CLOCK_MONOTONIC") != 0 ||
      !isfinite(result->clock_resolution_sec) ||
      result->clock_resolution_sec <= 0.0 || result->compiler == NULL ||
      result->compiler_version == NULL ||
      result->global_configure_flags == NULL || result->library_name == NULL ||
      !sha256_valid(result->config_sha256) ||
      !sha256_valid(result->runtime_environment_sha256) ||
      !sha256_valid(result->binary_sha256) || result->message == NULL ||
      !string_in(result->status, statuses,
                 sizeof(statuses) / sizeof(statuses[0])) ||
      !string_in(result->verification_status, verification_statuses,
                 sizeof(verification_statuses) /
                     sizeof(verification_statuses[0])) ||
      !string_in(result->series_role, series_roles,
                 sizeof(series_roles) / sizeof(series_roles[0]))) {
    set_error(error, error_size, "invalid result value");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (strcmp(result->implementation, "cpu") == 0) {
    if (result->cpu_backend == NULL || result->cpu_backend[0] == '\0' ||
        !string_in(result->cpu_backend_role, backend_roles,
                   sizeof(backend_roles) / sizeof(backend_roles[0])) ||
        !string_in(result->cpu_parallelism, parallelism_values,
                   sizeof(parallelism_values) /
                       sizeof(parallelism_values[0])) ||
        result->cpu_threads_requested.is_null ||
        result->cpu_threads_requested.value <= 0 ||
        (!result->cpu_threads_effective.is_null &&
         result->cpu_threads_effective.value <= 0)) {
      set_error(error, error_size, "invalid explicit CPU series metadata");
      return GPU_SUITE_ERROR_INVALID;
    }
  } else if (result->cpu_backend != NULL || result->cpu_backend_role != NULL ||
             result->cpu_parallelism != NULL ||
             !result->cpu_threads_requested.is_null ||
             !result->cpu_threads_effective.is_null) {
    set_error(error, error_size, "GPU result contains CPU metadata");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (strcmp(result->implementation, "cpu") != 0 &&
      strcmp(result->status, "success") == 0 &&
      (result->gpu_name == NULL || result->gpu_name[0] == '\0' ||
       result->gpu_uuid == NULL || result->gpu_uuid[0] == '\0' ||
       result->cuda_driver_version == NULL ||
       result->cuda_driver_version[0] == '\0' ||
       result->cuda_runtime_version == NULL ||
       result->cuda_runtime_version[0] == '\0' ||
       result->library_version == NULL || result->library_version[0] == '\0')) {
    set_error(error, error_size,
              "successful GPU result lacks runtime identity metadata");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->git_diff_sha256 != NULL &&
      !sha256_valid(result->git_diff_sha256)) {
    set_error(error, error_size, "invalid git_diff_sha256");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->source_snapshot_sha256 != NULL &&
      !sha256_valid(result->source_snapshot_sha256)) {
    set_error(error, error_size, "invalid source_snapshot_sha256");
    return GPU_SUITE_ERROR_INVALID;
  }
  if ((result->git_metadata_available &&
       (result->git_commit == NULL || result->git_commit[0] == '\0')) ||
      (!result->git_metadata_available && result->git_commit != NULL)) {
    set_error(error, error_size, "inconsistent Git metadata availability");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->measurement_start_timestamp != NULL &&
      !timestamp_valid(result->measurement_start_timestamp)) {
    set_error(error, error_size, "invalid measurement start timestamp");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->measurement_end_timestamp != NULL &&
      (!timestamp_valid(result->measurement_end_timestamp) ||
       result->measurement_start_timestamp == NULL)) {
    set_error(error, error_size, "invalid measurement end timestamp");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (!result->elapsed_total_sec.is_null &&
      (!isfinite(result->elapsed_total_sec.value) ||
       result->elapsed_total_sec.value < 0.0)) {
    set_error(error, error_size, "invalid elapsed_total_sec");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (!result->elapsed_sec.is_null && (!isfinite(result->elapsed_sec.value) ||
                                       result->elapsed_sec.value < 0.0)) {
    set_error(error, error_size, "invalid elapsed_sec");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (result->elapsed_total_sec.is_null != result->elapsed_sec.is_null ||
      (!result->elapsed_total_sec.is_null &&
       (result->measurement_start_timestamp == NULL ||
        result->measurement_end_timestamp == NULL))) {
    set_error(error, error_size, "inconsistent timing fields");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (strcmp(result->status, "success") == 0) {
    if (!result->attempted || result->failure_origin != NULL ||
        result->elapsed_total_sec.is_null || result->exit_code.is_null ||
        result->exit_code.value != 0 || result->message[0] != '\0') {
      set_error(error, error_size, "invalid successful result");
      return GPU_SUITE_ERROR_INVALID;
    }
  } else if (strcmp(result->status, "failure") == 0) {
    if (!result->attempted ||
        !string_in(result->failure_origin, origins,
                   sizeof(origins) / sizeof(origins[0])) ||
        strcmp(result->failure_origin, "prior-failure") == 0 ||
        strcmp(result->failure_origin, "prerequisite") == 0) {
      set_error(error, error_size, "invalid failed result");
      return GPU_SUITE_ERROR_INVALID;
    }
  } else {
    if (result->attempted ||
        !(result->failure_origin != NULL &&
          (strcmp(result->failure_origin, "prior-failure") == 0 ||
           strcmp(result->failure_origin, "prerequisite") == 0)) ||
        result->measurement_start_timestamp != NULL ||
        result->measurement_end_timestamp != NULL ||
        !result->elapsed_total_sec.is_null || !result->elapsed_sec.is_null ||
        strcmp(result->verification_status, "skipped") != 0 ||
        !result->exit_code.is_null) {
      set_error(error, error_size, "invalid skipped result");
      return GPU_SUITE_ERROR_INVALID;
    }
  }
  if (strcmp(result->benchmark, "cusolver") == 0 && result->repeat != 1) {
    set_error(error, error_size, "cuSOLVER result repeat must be one");
    return GPU_SUITE_ERROR_INVALID;
  }
  return GPU_SUITE_OK;
}

static gpu_suite_json_value *nullable_string(const char *value) {
  return value == NULL ? gpu_suite_json_null() : gpu_suite_json_string(value);
}

static gpu_suite_json_value *optional_int_value(gpu_suite_optional_int value) {
  return value.is_null ? gpu_suite_json_null()
                       : gpu_suite_json_int(value.value);
}

static gpu_suite_json_value *
optional_double_value(gpu_suite_optional_double value) {
  return value.is_null ? gpu_suite_json_null()
                       : gpu_suite_json_double(value.value);
}

static int result_json(const gpu_suite_result *result,
                       gpu_suite_json_value **output) {
  gpu_suite_json_value *root = gpu_suite_json_object();
  int status = GPU_SUITE_OK;
  if (root == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
#define ADD(key, expression)                                                   \
  do {                                                                         \
    status = object_take(root, (key), (expression));                           \
    if (status != GPU_SUITE_OK) {                                              \
      goto fail;                                                               \
    }                                                                          \
  } while (0)
  ADD("result_schema_version",
      gpu_suite_json_int(result->result_schema_version));
  ADD("run_id", gpu_suite_json_string(result->run_id));
  ADD("record_timestamp", gpu_suite_json_string(result->record_timestamp));
  ADD("measurement_start_timestamp",
      nullable_string(result->measurement_start_timestamp));
  ADD("measurement_end_timestamp",
      nullable_string(result->measurement_end_timestamp));
  ADD("system_label", gpu_suite_json_string(result->system_label));
  ADD("wave", gpu_suite_json_int(result->wave));
  ADD("node_index", gpu_suite_json_int(result->node_index));
  ADD("hostname", gpu_suite_json_string(result->hostname));
  ADD("block_id", gpu_suite_json_string(result->block_id));
  ADD("scheduler", nullable_string(result->scheduler));
  ADD("scheduler_job_id", nullable_string(result->scheduler_job_id));
  ADD("implementation_order",
      gpu_suite_json_clone(result->implementation_order));
  ADD("benchmark", gpu_suite_json_string(result->benchmark));
  ADD("implementation", gpu_suite_json_string(result->implementation));
  ADD("scope", gpu_suite_json_string(result->scope));
  ADD("problem_size", optional_int_value(result->problem_size));
  ADD("secondary_size", optional_int_value(result->secondary_size));
  ADD("parameters", gpu_suite_json_clone(result->parameters));
  ADD("precision", gpu_suite_json_string(result->precision));
  ADD("cpu_backend", nullable_string(result->cpu_backend));
  ADD("cpu_backend_role", nullable_string(result->cpu_backend_role));
  ADD("series_role", gpu_suite_json_string(result->series_role));
  ADD("cpu_threads_requested",
      optional_int_value(result->cpu_threads_requested));
  ADD("cpu_threads_effective",
      optional_int_value(result->cpu_threads_effective));
  ADD("cpu_parallelism", nullable_string(result->cpu_parallelism));
  ADD("warmup", gpu_suite_json_int(result->warmup));
  ADD("repeat", gpu_suite_json_int(result->repeat));
  ADD("trial", gpu_suite_json_int(result->trial));
  ADD("attempted", gpu_suite_json_bool(result->attempted));
  ADD("failure_origin", nullable_string(result->failure_origin));
  ADD("elapsed_total_sec", optional_double_value(result->elapsed_total_sec));
  ADD("elapsed_sec", optional_double_value(result->elapsed_sec));
  ADD("clock_id", gpu_suite_json_string(result->clock_id));
  ADD("clock_resolution_sec",
      gpu_suite_json_double(result->clock_resolution_sec));
  ADD("verification_metrics",
      gpu_suite_json_clone(result->verification_metrics));
  ADD("verification_thresholds",
      gpu_suite_json_clone(result->verification_thresholds));
  ADD("verification_primary_metric",
      nullable_string(result->verification_primary_metric));
  ADD("verification_status",
      gpu_suite_json_string(result->verification_status));
  ADD("getrf_info", optional_int_value(result->getrf_info));
  ADD("getrs_info", optional_int_value(result->getrs_info));
  ADD("device_id", optional_int_value(result->device_id));
  ADD("gpu_name", nullable_string(result->gpu_name));
  ADD("gpu_uuid", nullable_string(result->gpu_uuid));
  ADD("cuda_driver_version", nullable_string(result->cuda_driver_version));
  ADD("compiler", gpu_suite_json_string(result->compiler));
  ADD("compiler_version", gpu_suite_json_string(result->compiler_version));
  ADD("global_configure_flags",
      gpu_suite_json_string(result->global_configure_flags));
  ADD("library_name", gpu_suite_json_string(result->library_name));
  ADD("library_version", nullable_string(result->library_version));
  ADD("cuda_runtime_version", nullable_string(result->cuda_runtime_version));
  ADD("git_metadata_available",
      gpu_suite_json_bool(result->git_metadata_available));
  ADD("git_commit", nullable_string(result->git_commit));
  ADD("git_dirty", result->git_metadata_available
                       ? gpu_suite_json_bool(result->git_dirty)
                       : gpu_suite_json_null());
  ADD("git_diff_sha256", nullable_string(result->git_diff_sha256));
  ADD("source_snapshot_sha256",
      nullable_string(result->source_snapshot_sha256));
  ADD("config_sha256", gpu_suite_json_string(result->config_sha256));
  ADD("runtime_environment_sha256",
      gpu_suite_json_string(result->runtime_environment_sha256));
  ADD("binary_sha256", gpu_suite_json_string(result->binary_sha256));
  ADD("exit_code", optional_int_value(result->exit_code));
  ADD("status", gpu_suite_json_string(result->status));
  ADD("message", gpu_suite_json_string(result->message));
#undef ADD
  *output = root;
  return GPU_SUITE_OK;
fail:
#undef ADD
  gpu_suite_json_free(root);
  return status;
}

const char *gpu_suite_result_csv_header(void) { return CSV_HEADER; }

static int csv_cell(FILE *stream, const char *value) {
  const char *cursor;
  bool quote = false;
  if (value == NULL) {
    return GPU_SUITE_OK;
  }
  for (cursor = value; *cursor != '\0'; ++cursor) {
    if (*cursor == ',' || *cursor == '"' || *cursor == '\r' ||
        *cursor == '\n') {
      quote = true;
      break;
    }
  }
  if (!quote) {
    return fputs(value, stream) == EOF ? GPU_SUITE_ERROR_IO : GPU_SUITE_OK;
  }
  if (fputc('"', stream) == EOF) {
    return GPU_SUITE_ERROR_IO;
  }
  for (cursor = value; *cursor != '\0'; ++cursor) {
    if (*cursor == '"' && fputc('"', stream) == EOF) {
      return GPU_SUITE_ERROR_IO;
    }
    if (fputc((unsigned char)*cursor, stream) == EOF) {
      return GPU_SUITE_ERROR_IO;
    }
  }
  return fputc('"', stream) == EOF ? GPU_SUITE_ERROR_IO : GPU_SUITE_OK;
}

static int csv_separator(FILE *stream, bool *first) {
  if (*first) {
    *first = false;
    return GPU_SUITE_OK;
  }
  return fputc(',', stream) == EOF ? GPU_SUITE_ERROR_IO : GPU_SUITE_OK;
}

static int csv_text(FILE *stream, bool *first, const char *value) {
  int status = csv_separator(stream, first);
  return status == GPU_SUITE_OK ? csv_cell(stream, value) : status;
}

static int csv_int(FILE *stream, bool *first, int64_t value) {
  char text[32];
  (void)snprintf(text, sizeof(text), "%lld", (long long)value);
  return csv_text(stream, first, text);
}

static int csv_bool(FILE *stream, bool *first, bool value) {
  return csv_text(stream, first, value ? "true" : "false");
}

static int csv_json(FILE *stream, bool *first,
                    const gpu_suite_json_value *value) {
  char *text = NULL;
  size_t length = 0U;
  char error[128];
  int status =
      gpu_suite_json_serialize(value, &text, &length, error, sizeof(error));
  (void)length;
  if (status == GPU_SUITE_OK) {
    status = csv_text(stream, first, text);
  }
  free(text);
  return status;
}

static int csv_optional_int(FILE *stream, bool *first,
                            gpu_suite_optional_int value) {
  return value.is_null ? csv_text(stream, first, NULL)
                       : csv_int(stream, first, value.value);
}

static int csv_optional_double(FILE *stream, bool *first,
                               gpu_suite_optional_double value) {
  gpu_suite_json_value *number;
  char *text = NULL;
  size_t length = 0U;
  char error[128];
  int status;
  if (value.is_null) {
    return csv_text(stream, first, NULL);
  }
  number = gpu_suite_json_double(value.value);
  if (number == NULL) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  status =
      gpu_suite_json_serialize(number, &text, &length, error, sizeof(error));
  (void)length;
  gpu_suite_json_free(number);
  if (status == GPU_SUITE_OK) {
    status = csv_text(stream, first, text);
  }
  free(text);
  return status;
}

static int write_csv_row(FILE *stream, const gpu_suite_result *result) {
  bool first = true;
  int status;
#define CSV(call)                                                              \
  do {                                                                         \
    status = (call);                                                           \
    if (status != GPU_SUITE_OK) {                                              \
      return status;                                                           \
    }                                                                          \
  } while (0)
  CSV(csv_int(stream, &first, result->result_schema_version));
  CSV(csv_text(stream, &first, result->run_id));
  CSV(csv_text(stream, &first, result->record_timestamp));
  CSV(csv_text(stream, &first, result->measurement_start_timestamp));
  CSV(csv_text(stream, &first, result->measurement_end_timestamp));
  CSV(csv_text(stream, &first, result->system_label));
  CSV(csv_int(stream, &first, result->wave));
  CSV(csv_int(stream, &first, result->node_index));
  CSV(csv_text(stream, &first, result->hostname));
  CSV(csv_text(stream, &first, result->block_id));
  CSV(csv_text(stream, &first, result->scheduler));
  CSV(csv_text(stream, &first, result->scheduler_job_id));
  CSV(csv_json(stream, &first, result->implementation_order));
  CSV(csv_text(stream, &first, result->benchmark));
  CSV(csv_text(stream, &first, result->implementation));
  CSV(csv_text(stream, &first, result->scope));
  CSV(csv_optional_int(stream, &first, result->problem_size));
  CSV(csv_optional_int(stream, &first, result->secondary_size));
  CSV(csv_json(stream, &first, result->parameters));
  CSV(csv_text(stream, &first, result->precision));
  CSV(csv_text(stream, &first, result->cpu_backend));
  CSV(csv_text(stream, &first, result->cpu_backend_role));
  CSV(csv_text(stream, &first, result->series_role));
  CSV(csv_optional_int(stream, &first, result->cpu_threads_requested));
  CSV(csv_optional_int(stream, &first, result->cpu_threads_effective));
  CSV(csv_text(stream, &first, result->cpu_parallelism));
  CSV(csv_int(stream, &first, result->warmup));
  CSV(csv_int(stream, &first, result->repeat));
  CSV(csv_int(stream, &first, result->trial));
  CSV(csv_bool(stream, &first, result->attempted));
  CSV(csv_text(stream, &first, result->failure_origin));
  CSV(csv_optional_double(stream, &first, result->elapsed_total_sec));
  CSV(csv_optional_double(stream, &first, result->elapsed_sec));
  CSV(csv_text(stream, &first, result->clock_id));
  CSV(csv_optional_double(
      stream, &first,
      gpu_suite_optional_double_value(result->clock_resolution_sec)));
  CSV(csv_json(stream, &first, result->verification_metrics));
  CSV(csv_json(stream, &first, result->verification_thresholds));
  CSV(csv_text(stream, &first, result->verification_primary_metric));
  CSV(csv_text(stream, &first, result->verification_status));
  CSV(csv_optional_int(stream, &first, result->getrf_info));
  CSV(csv_optional_int(stream, &first, result->getrs_info));
  CSV(csv_optional_int(stream, &first, result->device_id));
  CSV(csv_text(stream, &first, result->gpu_name));
  CSV(csv_text(stream, &first, result->gpu_uuid));
  CSV(csv_text(stream, &first, result->cuda_driver_version));
  CSV(csv_text(stream, &first, result->compiler));
  CSV(csv_text(stream, &first, result->compiler_version));
  CSV(csv_text(stream, &first, result->global_configure_flags));
  CSV(csv_text(stream, &first, result->library_name));
  CSV(csv_text(stream, &first, result->library_version));
  CSV(csv_text(stream, &first, result->cuda_runtime_version));
  CSV(csv_bool(stream, &first, result->git_metadata_available));
  CSV(csv_text(stream, &first, result->git_commit));
  if (result->git_metadata_available) {
    CSV(csv_bool(stream, &first, result->git_dirty));
  } else {
    CSV(csv_text(stream, &first, NULL));
  }
  CSV(csv_text(stream, &first, result->git_diff_sha256));
  CSV(csv_text(stream, &first, result->source_snapshot_sha256));
  CSV(csv_text(stream, &first, result->config_sha256));
  CSV(csv_text(stream, &first, result->runtime_environment_sha256));
  CSV(csv_text(stream, &first, result->binary_sha256));
  CSV(csv_optional_int(stream, &first, result->exit_code));
  CSV(csv_text(stream, &first, result->status));
  CSV(csv_text(stream, &first, result->message));
#undef CSV
  return fputc('\n', stream) == EOF ? GPU_SUITE_ERROR_IO : GPU_SUITE_OK;
}

int gpu_suite_result_write(FILE *stream, gpu_suite_output_format format,
                           bool include_csv_header,
                           const gpu_suite_result *result, char *error,
                           size_t error_size) {
  gpu_suite_result snapshot;
  int status;
  if (stream == NULL || result == NULL) {
    set_error(error, error_size, "result stream or record is null");
    return GPU_SUITE_ERROR_INVALID;
  }
  snapshot = *result;
  if (gpu_suite_utc_timestamp(snapshot.record_timestamp_storage, error,
                              error_size) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_IO;
  }
  snapshot.record_timestamp = snapshot.record_timestamp_storage;
  result = &snapshot;
  status = gpu_suite_result_validate(result, error, error_size);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  if (format == GPU_SUITE_FORMAT_JSONL) {
    gpu_suite_json_value *json = NULL;
    char *serialized = NULL;
    size_t length = 0U;
    status = result_json(result, &json);
    if (status == GPU_SUITE_OK) {
      status = gpu_suite_json_serialize(json, &serialized, &length, error,
                                        error_size);
    }
    if (status == GPU_SUITE_OK &&
        (fwrite(serialized, 1U, length, stream) != length ||
         fputc('\n', stream) == EOF)) {
      status = GPU_SUITE_ERROR_IO;
    }
    free(serialized);
    gpu_suite_json_free(json);
  } else if (format == GPU_SUITE_FORMAT_CSV) {
    if (include_csv_header &&
        (fputs(CSV_HEADER, stream) == EOF || fputc('\n', stream) == EOF)) {
      status = GPU_SUITE_ERROR_IO;
    } else {
      status = write_csv_row(stream, result);
    }
  } else {
    status = GPU_SUITE_ERROR_INVALID;
  }
  if (status == GPU_SUITE_OK && fflush(stream) != 0) {
    status = GPU_SUITE_ERROR_IO;
  }
  if (status != GPU_SUITE_OK) {
    set_error(error, error_size, "could not write result record");
  }
  return status;
}

int gpu_suite_output_open(const char *path, FILE **stream, bool *must_close,
                          char *error, size_t error_size) {
  int descriptor;
  FILE *opened;
  if (path == NULL || stream == NULL || must_close == NULL || path[0] == '\0') {
    set_error(error, error_size, "invalid output path");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (strcmp(path, "-") == 0) {
    *stream = stdout;
    *must_close = false;
    return GPU_SUITE_OK;
  }
  descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0666);
  if (descriptor < 0) {
    set_error(error, error_size, strerror(errno));
    return errno == EEXIST ? GPU_SUITE_ERROR_EXISTS : GPU_SUITE_ERROR_IO;
  }
  opened = fdopen(descriptor, "w");
  if (opened == NULL) {
    int saved_errno = errno;
    (void)close(descriptor);
    set_error(error, error_size, strerror(saved_errno));
    return GPU_SUITE_ERROR_IO;
  }
  *stream = opened;
  *must_close = true;
  return GPU_SUITE_OK;
}

int gpu_suite_output_close(FILE *stream, bool must_close, char *error,
                           size_t error_size) {
  int status;
  if (stream == NULL) {
    set_error(error, error_size, "output stream is null");
    return GPU_SUITE_ERROR_INVALID;
  }
  status = must_close ? fclose(stream) : fflush(stream);
  if (status != 0) {
    set_error(error, error_size, strerror(errno));
    return GPU_SUITE_ERROR_IO;
  }
  return GPU_SUITE_OK;
}
