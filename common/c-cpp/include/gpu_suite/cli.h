#ifndef GPU_SUITE_CLI_H
#define GPU_SUITE_CLI_H

#include "gpu_suite/common.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GPU_SUITE_BENCHMARK_CUFFT,
  GPU_SUITE_BENCHMARK_CUBLAS,
  GPU_SUITE_BENCHMARK_CUSPARSE,
  GPU_SUITE_BENCHMARK_CUSOLVER,
  GPU_SUITE_BENCHMARK_CURAND,
  GPU_SUITE_BENCHMARK_THRUST
} gpu_suite_benchmark_kind;

typedef enum {
  GPU_SUITE_IMPLEMENTATION_CPU,
  GPU_SUITE_IMPLEMENTATION_CUDA,
  GPU_SUITE_IMPLEMENTATION_OPENACC
} gpu_suite_implementation;

typedef enum {
  GPU_SUITE_SCOPE_COMPUTE,
  GPU_SUITE_SCOPE_END_TO_END
} gpu_suite_scope;

typedef enum {
  GPU_SUITE_FORMAT_CSV,
  GPU_SUITE_FORMAT_JSONL
} gpu_suite_output_format;

typedef enum {
  GPU_SUITE_PARSE_ERROR = -1,
  GPU_SUITE_PARSE_OK = 0,
  GPU_SUITE_PARSE_HELP = 1
} gpu_suite_parse_result;

typedef struct {
  gpu_suite_benchmark_kind benchmark;
  gpu_suite_implementation implementation;
  uint64_t size;
  bool size_set;
  int warmup;
  int repeat;
  int trials;
  gpu_suite_scope scope;
  bool verify;
  char output[GPU_SUITE_PATH_CAPACITY];
  gpu_suite_output_format format;
  int device;
  char run_id[GPU_SUITE_RUN_ID_CAPACITY];
  char system_label[GPU_SUITE_LABEL_CAPACITY];
  int node_index;
  int wave;
  uint64_t seed;
  int cpu_threads;
  int cpu_threads_effective;
  char cpu_backend_role[GPU_SUITE_NAME_CAPACITY];
  char series_role[GPU_SUITE_NAME_CAPACITY];
  char cpu_parallelism[GPU_SUITE_NAME_CAPACITY];
  char implementation_order[GPU_SUITE_ORDER_CAPACITY];
  double abs_tolerance;
  double rel_tolerance;
  double sigma_multiplier;
  double expected_mean;
  double expected_second_central_moment;

  uint64_t batch;
  char transform[GPU_SUITE_NAME_CAPACITY];
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bool m_set;
  bool n_set;
  bool k_set;
  double alpha;
  double beta;
  uint64_t nx;
  uint64_t ny;
  bool nx_set;
  bool ny_set;
  uint64_t nrhs;
  char cpu_backend[GPU_SUITE_NAME_CAPACITY];
  char generator[GPU_SUITE_NAME_CAPACITY];
  char distribution[GPU_SUITE_NAME_CAPACITY];
  uint64_t offset;
  char order[GPU_SUITE_NAME_CAPACITY];
  char operation[GPU_SUITE_NAME_CAPACITY];
} gpu_suite_options;

void gpu_suite_options_init(gpu_suite_options *options,
                            gpu_suite_benchmark_kind benchmark,
                            gpu_suite_implementation implementation);
gpu_suite_parse_result gpu_suite_options_parse(gpu_suite_options *options,
                                               int argc, char **argv,
                                               char *error, size_t error_size);
int gpu_suite_options_validate(const gpu_suite_options *options, char *error,
                               size_t error_size);
void gpu_suite_options_usage(FILE *stream, const char *program,
                             gpu_suite_benchmark_kind benchmark);

const char *gpu_suite_benchmark_name(gpu_suite_benchmark_kind value);
const char *gpu_suite_implementation_name(gpu_suite_implementation value);
const char *gpu_suite_scope_name(gpu_suite_scope value);
const char *gpu_suite_format_name(gpu_suite_output_format value);
bool gpu_suite_run_id_validate(const char *value);
bool gpu_suite_implementation_order_validate(const char *value);

#ifdef __cplusplus
}
#endif

#endif
