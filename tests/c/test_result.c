#include "gpu_suite/benchmark.h"

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
  gpu_suite_result gpu;
  char error[256] = {0};
  FILE *stream;
  char *content;
  char path[] = "/tmp/gpu-suite-result-XXXXXX";
  int descriptor;
  bool must_close = false;
  const char *original_hostname;
  const char *original_block_id;
  char invalid_block[GPU_SUITE_RUN_ID_CAPACITY + GPU_SUITE_LABEL_CAPACITY +
                     32];

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

  result.verification_status = "failure";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  result.verification_status = "pass";
  result.status = "failure";
  result.failure_origin = "verification";
  result.exit_code = gpu_suite_optional_int_value(1);
  result.message = "verification failed";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  finalize_success(&result);

  result.problem_size = gpu_suite_optional_int_value(-1);
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  result.problem_size = gpu_suite_optional_int_value(256);
  result.secondary_size = gpu_suite_optional_int_value(-1);
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  result.secondary_size = gpu_suite_optional_int_value(8);

  result.verification_primary_metric = "missing_metric";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  result.verification_primary_metric = NULL;

  assert(gpu_suite_json_add_double(result.verification_metrics,
                                   "finite_metric", 0.0) == GPU_SUITE_OK);
  result.verification_primary_metric = "finite_metric";
  result.verification_status = "nonfinite";
  result.status = "failure";
  result.failure_origin = "verification";
  result.exit_code = gpu_suite_optional_int_value(1);
  result.message = "nonfinite output";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  assert(gpu_suite_json_add_null(result.verification_metrics,
                                 "null_metric") == GPU_SUITE_OK);
  result.verification_primary_metric = "null_metric";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) ==
         GPU_SUITE_OK);
  finalize_success(&result);

  original_hostname = result.hostname;
  original_block_id = result.block_id;
  for (size_t index = 0U; index < 4U; ++index) {
    static const char *const invalid_hostnames[] = {".", "..", "bad/name",
                                                     "bad\\name"};
    result.hostname = invalid_hostnames[index];
    assert(snprintf(invalid_block, sizeof(invalid_block), "%s|%d|%s",
                    result.run_id, result.wave, result.hostname) > 0);
    result.block_id = invalid_block;
    assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
           GPU_SUITE_OK);
  }
  result.hostname = original_hostname;
  result.block_id = "inconsistent-block";
  assert(gpu_suite_result_validate(&result, error, sizeof(error)) !=
         GPU_SUITE_OK);
  result.block_id = original_block_id;
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

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
  options.size = 256U;
  options.size_set = true;
  options.batch = 8U;
  options.repeat = 2;
  assert(gpu_suite_result_init(&gpu) == GPU_SUITE_OK);
  assert(gpu_suite_result_apply_options(&gpu, &options, error,
                                        sizeof(error)) == GPU_SUITE_OK);
  finalize_success(&gpu);
  assert(gpu_suite_result_validate(&gpu, error, sizeof(error)) != GPU_SUITE_OK);
  gpu.gpu_name = "Test GPU";
  gpu.gpu_uuid = "GPU-11111111-1111-1111-1111-111111111111";
  gpu.cuda_driver_version = "12.8.0";
  gpu.cuda_runtime_version = "12.8.0";
  gpu.library_version = "12080";
  assert(gpu_suite_result_validate(&gpu, error, sizeof(error)) == GPU_SUITE_OK);
  gpu.device_id = gpu_suite_optional_int_value(-1);
  assert(gpu_suite_result_validate(&gpu, error, sizeof(error)) != GPU_SUITE_OK);
  gpu.device_id = gpu_suite_optional_int_value(0);

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
  gpu_suite_result_destroy(&gpu);
  gpu_suite_result_destroy(&result);
  return 0;
}
