#include "gpu_suite/result.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void finalize_success(gpu_suite_result *result) {
  result->measurement_start_timestamp = "2026-07-14T00:00:00.000Z";
  result->measurement_end_timestamp = "2026-07-14T00:00:00.010Z";
  result->elapsed_total_sec = gpu_suite_optional_double_value(0.01);
  result->elapsed_sec = gpu_suite_optional_double_value(0.005);
  result->verification_status = "pass";
  result->attempted = true;
  result->failure_origin = NULL;
  result->exit_code = gpu_suite_optional_int_value(0);
  result->status = "success";
  result->message = "";
}

static char *read_stream(FILE *stream) {
  long length;
  char *content;
  assert(fflush(stream) == 0);
  assert(fseek(stream, 0L, SEEK_END) == 0);
  length = ftell(stream);
  assert(length >= 0L);
  assert(fseek(stream, 0L, SEEK_SET) == 0);
  content = (char *)malloc((size_t)length + 1U);
  assert(content != NULL);
  assert(fread(content, 1U, (size_t)length, stream) == (size_t)length);
  content[length] = '\0';
  return content;
}

int main(void) {
  gpu_suite_options options;
  gpu_suite_result result;
  gpu_suite_result skipped;
  char error[256] = {0};
  FILE *stream;
  char *content;
  char path[] = "/tmp/gpu-suite-result-XXXXXX";
  int descriptor;
  bool must_close = false;

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.size = 256U;
  options.size_set = true;
  options.batch = 8U;
  options.repeat = 2;
  assert(gpu_suite_result_init(&result) == GPU_SUITE_OK);
  assert(gpu_suite_result_apply_options(&result, &options, error,
                                        sizeof(error)) == GPU_SUITE_OK);
  finalize_success(&result);
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) ==
         GPU_SUITE_OK);
  result.record_timestamp = "2000-01-01T00:00:00.000Z";

  stream = tmpfile();
  assert(stream != NULL);
  assert(gpu_suite_result_write(stream, GPU_SUITE_FORMAT_JSONL, false, &result,
                                error, sizeof(error)) == GPU_SUITE_OK);
  content = read_stream(stream);
  assert(strstr(content, "\"attempted\":true") != NULL);
  assert(strstr(content, "\"elapsed_total_sec\":0.01") != NULL);
  assert(strstr(content, "2000-01-01T00:00:00.000Z") == NULL);
  assert(strstr(content, "NaN") == NULL && strstr(content, "Infinity") == NULL);
  assert(content[strlen(content) - 1U] == '\n');
  free(content);
  assert(fclose(stream) == 0);

  stream = tmpfile();
  assert(stream != NULL);
  assert(gpu_suite_result_write(stream, GPU_SUITE_FORMAT_CSV, true, &result,
                                error, sizeof(error)) == GPU_SUITE_OK);
  content = read_stream(stream);
  assert(strncmp(content, gpu_suite_result_csv_header(),
                 strlen(gpu_suite_result_csv_header())) == 0);
  assert(strstr(content, "\"\"nfft\"\":256") != NULL);
  free(content);
  assert(fclose(stream) == 0);

  assert(gpu_suite_result_init(&skipped) == GPU_SUITE_OK);
  assert(gpu_suite_result_apply_options(&skipped, &options, error,
                                        sizeof(error)) == GPU_SUITE_OK);
  skipped.attempted = false;
  skipped.failure_origin = "prior-failure";
  skipped.verification_status = "skipped";
  skipped.status = "skipped";
  skipped.message = "not started after prior failure";
  assert(gpu_suite_result_validate(&skipped, error, sizeof(error)) ==
         GPU_SUITE_OK);
  skipped.attempted = true;
  assert(gpu_suite_result_validate(&skipped, error, sizeof(error)) !=
         GPU_SUITE_OK);

  descriptor = mkstemp(path);
  assert(descriptor >= 0);
  assert(close(descriptor) == 0);
  assert(gpu_suite_output_open(path, &stream, &must_close, error,
                               sizeof(error)) == GPU_SUITE_ERROR_EXISTS);
  assert(unlink(path) == 0);
  assert(gpu_suite_output_open(path, &stream, &must_close, error,
                               sizeof(error)) == GPU_SUITE_OK);
  assert(must_close);
  assert(gpu_suite_output_close(stream, must_close, error, sizeof(error)) ==
         GPU_SUITE_OK);
  assert(unlink(path) == 0);

  gpu_suite_result_destroy(&skipped);
  gpu_suite_result_destroy(&result);
  return 0;
}
