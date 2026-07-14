#ifndef GPU_SUITE_BENCHMARK_H
#define GPU_SUITE_BENCHMARK_H

#include "gpu_suite/result.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FILE *stream;
  bool must_close;
  bool include_header;
  gpu_suite_output_format format;
} gpu_suite_benchmark_writer;

int gpu_suite_benchmark_writer_open(gpu_suite_benchmark_writer *writer,
                                    const gpu_suite_options *options,
                                    char *error, size_t error_size);
int gpu_suite_benchmark_writer_write(gpu_suite_benchmark_writer *writer,
                                     gpu_suite_result *result, char *error,
                                     size_t error_size);
int gpu_suite_benchmark_writer_close(gpu_suite_benchmark_writer *writer,
                                     char *error, size_t error_size);
int gpu_suite_benchmark_result_init(gpu_suite_result *result,
                                    const gpu_suite_options *options, int trial,
                                    char *error, size_t error_size);
int gpu_suite_benchmark_emit_unmeasured(gpu_suite_benchmark_writer *writer,
                                        const gpu_suite_options *options,
                                        int first_trial, bool first_attempted,
                                        const char *first_origin,
                                        const char *message, char *error,
                                        size_t error_size);

int gpu_suite_json_add_double(gpu_suite_json_value *object, const char *key,
                              double value);
int gpu_suite_json_add_int(gpu_suite_json_value *object, const char *key,
                           int64_t value);
int gpu_suite_json_add_string(gpu_suite_json_value *object, const char *key,
                              const char *value);

#ifdef __cplusplus
}
#endif

#endif
