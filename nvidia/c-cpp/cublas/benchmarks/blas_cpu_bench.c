#include "gpu_suite/gpu_suite.h"

#include GPU_SUITE_CBLAS_HEADER

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill(double *values, size_t count, double value) {
  for (size_t i = 0; i < count; ++i)
    values[i] = value;
}

static int set_verification(gpu_suite_result *result,
                            const gpu_suite_options *options, const double *c,
                            size_t count, int k) {
  gpu_suite_json_free(result->verification_metrics);
  gpu_suite_json_free(result->verification_thresholds);
  result->verification_metrics = gpu_suite_json_object();
  result->verification_thresholds = gpu_suite_json_object();
  if (!result->verification_metrics || !result->verification_thresholds)
    return GPU_SUITE_ERROR_NOMEM;
  if (!options->verify) {
    result->verification_primary_metric = NULL;
    result->verification_status = "skipped";
    return GPU_SUITE_OK;
  }
  int updates = options->scope == GPU_SUITE_SCOPE_COMPUTE ? options->repeat : 1;
  double expected = 1.0;
  for (int r = 0; r < updates; ++r)
    expected = options->alpha * k + options->beta * expected;
  double maximum = 0.0;
  for (size_t i = 0; i < count; ++i)
    maximum = fmax(maximum, fabs(c[i] - expected));
  if (!isfinite(maximum)) {
    result->verification_status = "nonfinite";
    return GPU_SUITE_ERROR_FORMAT;
  }
  gpu_suite_json_add_double(result->verification_metrics, "max_abs_error",
                            maximum);
  gpu_suite_json_value *threshold = gpu_suite_json_object();
  gpu_suite_json_add_string(threshold, "method", "absolute-plus-relative");
  gpu_suite_json_add_double(threshold, "reference_scale", fabs(expected));
  gpu_suite_json_add_double(threshold, "abs_tolerance", options->abs_tolerance);
  gpu_suite_json_add_double(threshold, "rel_tolerance", options->rel_tolerance);
  if (gpu_suite_json_object_set(result->verification_thresholds,
                                "max_abs_error", threshold) != GPU_SUITE_OK) {
    gpu_suite_json_free(threshold);
    return GPU_SUITE_ERROR_NOMEM;
  }
  result->verification_primary_metric = "max_abs_error";
  result->verification_status =
      maximum <=
              options->abs_tolerance + options->rel_tolerance * fabs(expected)
          ? "pass"
          : "failure";
  return GPU_SUITE_OK;
}

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUBLAS);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK ||
      strcmp(options.cpu_backend, GPU_SUITE_COMPILED_CPU_BACKEND) != 0) {
    fprintf(stderr, "%s%s\n", error,
            parsed == GPU_SUITE_PARSE_OK
                ? "CPU backend does not match this binary"
                : "");
    return EXIT_FAILURE;
  }
  uint64_t mu = options.size_set ? options.size : options.m,
           nu = options.size_set ? options.size : options.n,
           ku = options.size_set ? options.size : options.k;
  if (mu > INT_MAX || nu > INT_MAX || ku > INT_MAX) {
    fprintf(stderr, "DGEMM dimension exceeds CBLAS int range\n");
    return EXIT_FAILURE;
  }
  int m = (int)mu, n = (int)nu, k = (int)ku;
  size_t ac, bc, cc;
  if (!gpu_suite_checked_mul_size((size_t)m, (size_t)k, &ac) ||
      !gpu_suite_checked_mul_size((size_t)k, (size_t)n, &bc) ||
      !gpu_suite_checked_mul_size((size_t)m, (size_t)n, &cc)) {
    return EXIT_FAILURE;
  }
  gpu_suite_benchmark_writer writer;
  if (gpu_suite_benchmark_writer_open(&writer, &options, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
    fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }
  double *a = malloc(ac * sizeof(*a)), *b = malloc(bc * sizeof(*b)),
         *c = malloc(cc * sizeof(*c));
  if (!a || !b || !c) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "matrix allocation failed", error,
                                        sizeof(error));
    free(c);
    free(b);
    free(a);
    gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
    return EXIT_FAILURE;
  }
  fill(a, ac, 1);
  fill(b, bc, 1);
  fill(c, cc, 1);
  for (int warmup = 0; warmup < options.warmup; ++warmup)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                options.alpha, a, m, b, k, options.beta, c, m);
  fill(c, cc, 1);
  int any_failure = 0;
  for (int trial = 0; trial < options.trials; ++trial) {
    gpu_suite_result result;
    if (gpu_suite_benchmark_result_init(&result, &options, trial, error,
                                        sizeof(error)) != GPU_SUITE_OK) {
      any_failure = 1;
      break;
    }
    char start_ts[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
         end_ts[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    struct timespec start, end;
    double elapsed = 0;
    int ok =
        gpu_suite_utc_timestamp(start_ts, error, sizeof(error)) == GPU_SUITE_OK;
    fill(c, cc, 1);
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE && ok) {
      ok = gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK;
      for (int r = 0; r < options.repeat && ok; ++r)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    options.alpha, a, m, b, k, options.beta, c, m);
      ok =
          ok && gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
      if (ok)
        elapsed = gpu_suite_clock_elapsed(&start, &end);
    } else if (ok) {
      for (int r = 0; r < options.repeat && ok; ++r) {
        fill(c, cc, 1);
        ok = gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK;
        if (!ok)
          break;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    options.alpha, a, m, b, k, options.beta, c, m);
        ok = gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
        if (ok)
          elapsed += gpu_suite_clock_elapsed(&start, &end);
      }
    }
    ok = ok &&
         gpu_suite_utc_timestamp(end_ts, error, sizeof(error)) == GPU_SUITE_OK;
    if (ok) {
      result.measurement_start_timestamp = start_ts;
      result.measurement_end_timestamp = end_ts;
      result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      result.elapsed_sec =
          gpu_suite_optional_double_value(elapsed / options.repeat);
      int verified = set_verification(&result, &options, c, cc, k);
      int pass = verified == GPU_SUITE_OK &&
                 (strcmp(result.verification_status, "pass") == 0 ||
                  strcmp(result.verification_status, "skipped") == 0);
      result.attempted = true;
      result.failure_origin = pass ? NULL : "verification";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.status = pass ? "success" : "failure";
      result.message = pass ? "" : "DGEMM verification failed";
      any_failure |= !pass;
    } else {
      result.attempted = true;
      result.failure_origin = "benchmark";
      result.verification_status = "skipped";
      result.exit_code = gpu_suite_optional_int_value(1);
      result.status = "failure";
      result.message = "timing failed";
      any_failure = 1;
    }
    if (gpu_suite_benchmark_writer_write(&writer, &result, error,
                                         sizeof(error)) != GPU_SUITE_OK) {
      gpu_suite_result_destroy(&result);
      any_failure = 1;
      break;
    }
    gpu_suite_result_destroy(&result);
    if (!ok) {
      gpu_suite_benchmark_emit_unmeasured(
          &writer, &options, trial + 1, false, "prior-failure",
          "prior timing failure", error, sizeof(error));
      break;
    }
  }
  free(c);
  free(b);
  free(a);
  if (gpu_suite_benchmark_writer_close(&writer, error, sizeof(error)) !=
      GPU_SUITE_OK)
    any_failure = 1;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
