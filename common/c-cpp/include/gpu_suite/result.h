#ifndef GPU_SUITE_RESULT_H
#define GPU_SUITE_RESULT_H

#include "gpu_suite/cli.h"
#include "gpu_suite/json.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  bool is_null;
  int64_t value;
} gpu_suite_optional_int;

typedef struct {
  bool is_null;
  double value;
} gpu_suite_optional_double;

typedef struct {
  int result_schema_version;
  const char *run_id;
  const char *record_timestamp;
  const char *measurement_start_timestamp;
  const char *measurement_end_timestamp;
  const char *system_label;
  int wave;
  int node_index;
  const char *hostname;
  const char *block_id;
  const char *scheduler;
  const char *scheduler_job_id;
  gpu_suite_json_value *implementation_order;
  const char *benchmark;
  const char *implementation;
  const char *scope;
  gpu_suite_optional_int problem_size;
  gpu_suite_optional_int secondary_size;
  gpu_suite_json_value *parameters;
  const char *precision;
  const char *cpu_backend;
  const char *cpu_backend_role;
  const char *series_role;
  gpu_suite_optional_int cpu_threads_requested;
  gpu_suite_optional_int cpu_threads_effective;
  const char *cpu_parallelism;
  int warmup;
  int repeat;
  int trial;
  bool attempted;
  const char *failure_origin;
  gpu_suite_optional_double elapsed_total_sec;
  gpu_suite_optional_double elapsed_sec;
  const char *clock_id;
  double clock_resolution_sec;
  gpu_suite_json_value *verification_metrics;
  gpu_suite_json_value *verification_thresholds;
  const char *verification_primary_metric;
  const char *verification_status;
  gpu_suite_optional_int getrf_info;
  gpu_suite_optional_int getrs_info;
  gpu_suite_optional_int device_id;
  const char *gpu_name;
  const char *gpu_uuid;
  const char *cuda_driver_version;
  const char *compiler;
  const char *compiler_version;
  const char *global_configure_flags;
  const char *library_name;
  const char *library_version;
  const char *cuda_runtime_version;
  bool git_metadata_available;
  const char *git_commit;
  bool git_dirty;
  const char *git_diff_sha256;
  const char *source_snapshot_sha256;
  const char *config_sha256;
  const char *runtime_environment_sha256;
  const char *binary_sha256;
  gpu_suite_optional_int exit_code;
  const char *status;
  const char *message;

  char hostname_storage[GPU_SUITE_LABEL_CAPACITY];
  char block_id_storage[GPU_SUITE_RUN_ID_CAPACITY + GPU_SUITE_LABEL_CAPACITY +
                        32];
  char record_timestamp_storage[GPU_SUITE_TIMESTAMP_CAPACITY];
  char gpu_name_storage[GPU_SUITE_LABEL_CAPACITY];
  char gpu_uuid_storage[64];
  char cuda_driver_version_storage[32];
  char cuda_runtime_version_storage[32];
  char library_version_storage[32];
} gpu_suite_result;

int gpu_suite_result_init(gpu_suite_result *result);
void gpu_suite_result_destroy(gpu_suite_result *result);
int gpu_suite_result_apply_options(gpu_suite_result *result,
                                   const gpu_suite_options *options,
                                   char *error, size_t error_size);
void gpu_suite_result_apply_build_metadata(gpu_suite_result *result,
                                           gpu_suite_implementation impl);
void gpu_suite_result_apply_provenance_environment(gpu_suite_result *result);
int gpu_suite_result_validate(const gpu_suite_result *result, char *error,
                              size_t error_size);

const char *gpu_suite_result_csv_header(void);
int gpu_suite_result_write(FILE *stream, gpu_suite_output_format format,
                           bool include_csv_header,
                           const gpu_suite_result *result, char *error,
                           size_t error_size);
int gpu_suite_output_open(const char *path, FILE **stream, bool *must_close,
                          char *error, size_t error_size);
int gpu_suite_output_close(FILE *stream, bool must_close, char *error,
                           size_t error_size);

gpu_suite_optional_int gpu_suite_optional_int_null(void);
gpu_suite_optional_int gpu_suite_optional_int_value(int64_t value);
gpu_suite_optional_double gpu_suite_optional_double_null(void);
gpu_suite_optional_double gpu_suite_optional_double_value(double value);

#ifdef __cplusplus
}
#endif

#endif
