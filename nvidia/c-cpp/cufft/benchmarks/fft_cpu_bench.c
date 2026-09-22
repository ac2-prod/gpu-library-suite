#include "gpu_suite/gpu_suite.h"

#include <fftw3.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  fftwf_complex *input;
  fftwf_complex *output;
  size_t count;
  int nfft;
  int batch;
} fft_buffers;

static void initialize_buffers(fft_buffers *buffers) {
  size_t index;
  for (index = 0U; index < buffers->count; ++index) {
    buffers->input[index][0] = 1.0F;
    buffers->input[index][1] = 0.0F;
    buffers->output[index][0] = 0.0F;
    buffers->output[index][1] = 0.0F;
  }
}

static fftwf_plan make_plan(fft_buffers *buffers) {
  int length[1] = {buffers->nfft};
  return fftwf_plan_many_dft(1, length, buffers->batch, buffers->input, NULL, 1,
                             buffers->nfft, buffers->output, NULL, 1,
                             buffers->nfft, FFTW_FORWARD, FFTW_ESTIMATE);
}

static int add_double(gpu_suite_json_value *object, const char *key,
                      double value) {
  gpu_suite_json_value *number = gpu_suite_json_double(value);
  int status;
  if (number == NULL) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  status = gpu_suite_json_object_set(object, key, number);
  if (status != GPU_SUITE_OK) {
    gpu_suite_json_free(number);
  }
  return status;
}

static int make_verification(gpu_suite_result *result,
                             const gpu_suite_options *options,
                             const fft_buffers *buffers) {
  double dc_relative_error = 0.0;
  double non_dc_max_abs_error = 0.0;
  size_t transform;
  int status;
  bool finite = true;

  if (gpu_suite_verification_reset(result) != GPU_SUITE_OK) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (!options->verify) {
    result->verification_primary_metric = NULL;
    result->verification_status = "skipped";
    return GPU_SUITE_OK;
  }

  for (transform = 0U; transform < (size_t)buffers->batch; ++transform) {
    const size_t base = transform * (size_t)buffers->nfft;
    size_t frequency;
    const double dc_real = (double)buffers->output[base][0];
    const double dc_imag = (double)buffers->output[base][1];
    const double expected = (double)buffers->nfft;
    double real_error = 0.0;
    double imag_error = 0.0;
    double transform_dc_error = 0.0;
    if (!gpu_suite_finite_absolute_error(dc_real, expected, &real_error) ||
        !gpu_suite_finite_absolute_error(dc_imag, 0.0, &imag_error) ||
        !gpu_suite_finite_max_update(real_error, &transform_dc_error) ||
        !gpu_suite_finite_max_update(imag_error, &transform_dc_error)) {
      finite = false;
    } else {
      transform_dc_error /= expected;
      if (!gpu_suite_finite_max_update(transform_dc_error,
                                       &dc_relative_error))
        finite = false;
    }
    for (frequency = 1U; frequency < (size_t)buffers->nfft; ++frequency) {
      const size_t index = base + frequency;
      double real_magnitude = 0.0;
      double imag_magnitude = 0.0;
      if (!gpu_suite_finite_absolute_error(
              (double)buffers->output[index][0], 0.0, &real_magnitude) ||
          !gpu_suite_finite_absolute_error(
              (double)buffers->output[index][1], 0.0, &imag_magnitude) ||
          !gpu_suite_finite_max_update(real_magnitude,
                                       &non_dc_max_abs_error) ||
          !gpu_suite_finite_max_update(imag_magnitude,
                                       &non_dc_max_abs_error))
        finite = false;
    }
  }
  status = gpu_suite_verification_add_upper_bound_threshold(
      result, "dc_relative_error", "relative-upper-bound",
      options->rel_tolerance);
  if (status == GPU_SUITE_OK)
    status = gpu_suite_verification_add_upper_bound_threshold(
        result, "non_dc_max_abs_error", "absolute-upper-bound",
        options->abs_tolerance);
  if (status != GPU_SUITE_OK)
    return status;
  if (!finite) {
    if (gpu_suite_json_add_null(result->verification_metrics,
                                "dc_relative_error") != GPU_SUITE_OK ||
        gpu_suite_json_add_null(result->verification_metrics,
                                "non_dc_max_abs_error") != GPU_SUITE_OK)
      return GPU_SUITE_ERROR_NOMEM;
    result->verification_primary_metric = "non_dc_max_abs_error";
    result->verification_status = "nonfinite";
    return GPU_SUITE_OK;
  }
  status = add_double(result->verification_metrics, "dc_relative_error",
                      dc_relative_error);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  status = add_double(result->verification_metrics, "non_dc_max_abs_error",
                      non_dc_max_abs_error);
  if (status != GPU_SUITE_OK) {
    return status;
  }

  result->verification_primary_metric = "non_dc_max_abs_error";
  result->verification_status =
      dc_relative_error <= options->rel_tolerance &&
              non_dc_max_abs_error <= options->abs_tolerance
          ? "pass"
          : "failure";
  return GPU_SUITE_OK;
}

static int write_result(FILE *stream, bool *write_header,
                        gpu_suite_output_format format,
                        gpu_suite_result *result) {
  char error[256] = {0};
  int status = gpu_suite_result_write(stream, format, *write_header, result,
                                      error, sizeof(error));
  if (status != GPU_SUITE_OK) {
    fprintf(stderr, "result output failed: %s\n", error);
    return status;
  }
  *write_header = false;
  return GPU_SUITE_OK;
}

static int emit_unmeasured_rows(FILE *stream, bool *write_header,
                                const gpu_suite_options *options,
                                int first_trial, bool first_attempted,
                                const char *origin, const char *message) {
  int trial;
  for (trial = first_trial; trial < options->trials; ++trial) {
    gpu_suite_result result;
    char error[256] = {0};
    if (gpu_suite_result_init(&result) != GPU_SUITE_OK) {
      fprintf(stderr, "could not initialize failure result: %s\n", error);
      return GPU_SUITE_ERROR_FORMAT;
    }
    if (gpu_suite_result_apply_options(&result, options, error,
                                       sizeof(error)) != GPU_SUITE_OK) {
      gpu_suite_result_destroy(&result);
      fprintf(stderr, "could not initialize failure result: %s\n", error);
      return GPU_SUITE_ERROR_FORMAT;
    }
    result.trial = trial;
    result.attempted = trial == first_trial ? first_attempted : false;
    result.failure_origin = trial == first_trial ? origin : "prior-failure";
    result.verification_status = "skipped";
    result.exit_code = result.attempted ? gpu_suite_optional_int_value(1)
                                        : gpu_suite_optional_int_null();
    result.status = result.attempted ? "failure" : "skipped";
    result.message = message;
    if (write_result(stream, write_header, options->format, &result) !=
        GPU_SUITE_OK) {
      gpu_suite_result_destroy(&result);
      return GPU_SUITE_ERROR_IO;
    }
    gpu_suite_result_destroy(&result);
  }
  return GPU_SUITE_OK;
}

int main(int argc, char **argv) {
  gpu_suite_options options;
  gpu_suite_parse_result parse_status;
  fft_buffers buffers = {0};
  fftwf_plan compute_plan = NULL;
  FILE *stream = NULL;
  bool must_close = false;
  bool write_header = true;
  bool threads_initialized = false;
  size_t bytes;
  char error[256] = {0};
  int trial;
  int exit_status = EXIT_FAILURE;

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  parse_status =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parse_status == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUFFT);
    return EXIT_SUCCESS;
  }
  if (parse_status != GPU_SUITE_PARSE_OK) {
    fprintf(stderr, "%s\n", error);
    gpu_suite_options_usage(stderr, argv[0], GPU_SUITE_BENCHMARK_CUFFT);
    return EXIT_FAILURE;
  }
  if (strcmp(options.cpu_backend, "cpu-fftw-serial") != 0 &&
      strcmp(options.cpu_backend, "cpu-fftw-threaded") != 0) {
    fprintf(stderr, "unsupported cuFFT CPU backend: %s\n", options.cpu_backend);
    return EXIT_FAILURE;
  }
  if (gpu_suite_output_open(options.output, &stream, &must_close, error,
                            sizeof(error)) != GPU_SUITE_OK) {
    fprintf(stderr, "could not open output: %s\n", error);
    return EXIT_FAILURE;
  }

#ifndef GPU_SUITE_HAVE_FFTW_THREADS
  if (strcmp(options.cpu_backend, "cpu-fftw-threaded") == 0) {
    fprintf(stderr, "threaded FFTW backend is not compiled into this binary\n");
    (void)emit_unmeasured_rows(stream, &write_header, &options, 0, false,
                               "prerequisite",
                               "threaded FFTW backend is unavailable");
    goto cleanup;
  }
#else
  if (strcmp(options.cpu_backend, "cpu-fftw-threaded") == 0) {
    if (fftwf_init_threads() == 0) {
      fprintf(stderr, "fftwf_init_threads failed\n");
      (void)emit_unmeasured_rows(stream, &write_header, &options, 0, true,
                                 "benchmark", "fftwf_init_threads failed");
      goto cleanup;
    }
    threads_initialized = true;
    fftwf_plan_with_nthreads(options.cpu_threads);
  }
#endif

  if (!gpu_suite_checked_u64_to_int(options.size, &buffers.nfft) ||
      !gpu_suite_checked_u64_to_int(options.batch, &buffers.batch) ||
      !gpu_suite_checked_u64_to_size(options.size, &buffers.count) ||
      !gpu_suite_checked_mul_size(buffers.count, (size_t)buffers.batch,
                                  &buffers.count) ||
      !gpu_suite_checked_bytes(buffers.count, sizeof(*buffers.input), &bytes)) {
    fprintf(stderr, "cuFFT dimensions overflow host representation\n");
    (void)emit_unmeasured_rows(stream, &write_header, &options, 0, false,
                               "prerequisite", "problem dimensions overflow");
    goto cleanup;
  }
  buffers.input = (fftwf_complex *)fftwf_malloc(bytes);
  buffers.output = (fftwf_complex *)fftwf_malloc(bytes);
  if (buffers.input == NULL || buffers.output == NULL) {
    fprintf(stderr, "FFTW allocation failed for %zu complex values\n",
            buffers.count);
    (void)emit_unmeasured_rows(stream, &write_header, &options, 0, true,
                               "benchmark", "FFTW allocation failed");
    goto cleanup;
  }
  initialize_buffers(&buffers);

  if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
    int warmup;
    compute_plan = make_plan(&buffers);
    if (compute_plan == NULL) {
      fprintf(stderr, "fftwf_plan_many_dft failed\n");
      (void)emit_unmeasured_rows(stream, &write_header, &options, 0, true,
                                 "benchmark", "FFTW plan creation failed");
      goto cleanup;
    }
    for (warmup = 0; warmup < options.warmup; ++warmup) {
      fftwf_execute(compute_plan);
    }
    initialize_buffers(&buffers);
  } else {
    int warmup;
    for (warmup = 0; warmup < options.warmup; ++warmup) {
      fftwf_plan plan;
      initialize_buffers(&buffers);
      plan = make_plan(&buffers);
      if (plan == NULL) {
        fprintf(stderr, "FFTW warm-up plan creation failed\n");
        (void)emit_unmeasured_rows(stream, &write_header, &options, 0, true,
                                   "benchmark",
                                   "FFTW warm-up plan creation failed");
        goto cleanup;
      }
      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
    }
    initialize_buffers(&buffers);
  }

  for (trial = 0; trial < options.trials; ++trial) {
    gpu_suite_result result;
    struct timespec start;
    struct timespec end;
    char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    double elapsed_total = 0.0;
    int repeat;
    bool trial_failed = false;
    const char *failure_message = "";

    initialize_buffers(&buffers);
    if (gpu_suite_result_init(&result) != GPU_SUITE_OK) {
      fprintf(stderr, "could not initialize result: %s\n", error);
      (void)emit_unmeasured_rows(stream, &write_header, &options, trial, true,
                                 "benchmark", "result initialization failed");
      goto cleanup;
    }
    if (gpu_suite_result_apply_options(&result, &options, error,
                                       sizeof(error)) != GPU_SUITE_OK) {
      gpu_suite_result_destroy(&result);
      fprintf(stderr, "could not initialize result: %s\n", error);
      (void)emit_unmeasured_rows(stream, &write_header, &options, trial, true,
                                 "benchmark", "result initialization failed");
      goto cleanup;
    }
    result.trial = trial;
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      if (gpu_suite_measurement_start(start_timestamp, &start, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
        trial_failed = true;
        failure_message = "could not read measurement start clock";
      } else {
        for (repeat = 0; repeat < options.repeat; ++repeat) {
          fftwf_execute(compute_plan);
        }
        if (gpu_suite_measurement_end(&end, end_timestamp, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
          trial_failed = true;
          failure_message = "could not read measurement end clock";
        } else {
          elapsed_total = gpu_suite_clock_elapsed(&start, &end);
        }
      }
    } else {
      for (repeat = 0; repeat < options.repeat; ++repeat) {
        fftwf_plan plan;
        initialize_buffers(&buffers);
        const int start_status =
            repeat == 0
                ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                              sizeof(error))
                : gpu_suite_clock_now(&start, error, sizeof(error));
        if (start_status != GPU_SUITE_OK) {
          trial_failed = true;
          failure_message = "could not read measurement start clock";
          break;
        }
        plan = make_plan(&buffers);
        if (plan == NULL) {
          trial_failed = true;
          failure_message = "FFTW plan creation failed";
          break;
        }
        fftwf_execute(plan);
        const int end_status =
            repeat + 1 == options.repeat
                ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                            sizeof(error))
                : gpu_suite_clock_now(&end, error, sizeof(error));
        if (end_status != GPU_SUITE_OK) {
          fftwf_destroy_plan(plan);
          trial_failed = true;
          failure_message = "could not read measurement end clock";
          break;
        }
        fftwf_destroy_plan(plan);
        elapsed_total += gpu_suite_clock_elapsed(&start, &end);
      }
    }

    if (!trial_failed) {
      result.measurement_start_timestamp = start_timestamp;
      result.measurement_end_timestamp = end_timestamp;
      result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed_total);
      result.elapsed_sec = gpu_suite_optional_double_value(
          elapsed_total / (double)options.repeat);
      if (make_verification(&result, &options, &buffers) != GPU_SUITE_OK) {
        fprintf(stderr, "verification result construction failed\n");
        gpu_suite_result_destroy(&result);
        (void)emit_unmeasured_rows(
            stream, &write_header, &options, trial, true, "benchmark",
            "verification result construction failed");
        goto cleanup;
      } else if (strcmp(result.verification_status, "pass") == 0 ||
                 strcmp(result.verification_status, "skipped") == 0) {
        result.attempted = true;
        result.failure_origin = NULL;
        result.exit_code = gpu_suite_optional_int_value(0);
        result.status = "success";
        result.message = "";
      } else {
        result.attempted = true;
        result.failure_origin = "verification";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "cuFFT verification threshold exceeded";
      }
    } else {
      result.attempted = true;
      result.failure_origin = "benchmark";
      result.verification_status = "skipped";
      result.exit_code = gpu_suite_optional_int_value(1);
      result.status = "failure";
      result.message = failure_message;
    }

    if (write_result(stream, &write_header, options.format, &result) !=
        GPU_SUITE_OK) {
      gpu_suite_result_destroy(&result);
      goto cleanup;
    }
    if (strcmp(result.status, "failure") == 0 && trial_failed) {
      gpu_suite_result_destroy(&result);
      if (trial + 1 < options.trials) {
        (void)emit_unmeasured_rows(stream, &write_header, &options, trial + 1,
                                   false, "prior-failure", failure_message);
      }
      goto cleanup;
    }
    if (strcmp(result.status, "failure") == 0) {
      exit_status = EXIT_FAILURE;
    } else if (exit_status != EXIT_FAILURE || trial == 0) {
      exit_status = EXIT_SUCCESS;
    }
    gpu_suite_result_destroy(&result);
  }

cleanup:
  if (compute_plan != NULL) {
    fftwf_destroy_plan(compute_plan);
  }
  fftwf_free(buffers.output);
  fftwf_free(buffers.input);
#ifdef GPU_SUITE_HAVE_FFTW_THREADS
  if (threads_initialized) {
    fftwf_cleanup_threads();
  }
#else
  (void)threads_initialized;
#endif
  if (stream != NULL && gpu_suite_output_close(stream, must_close, error,
                                               sizeof(error)) != GPU_SUITE_OK) {
    fprintf(stderr, "could not close output: %s\n", error);
    exit_status = EXIT_FAILURE;
  }
  return exit_status;
}
