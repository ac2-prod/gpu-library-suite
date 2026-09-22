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
  if (gpu_suite_verification_reset(result) != GPU_SUITE_OK)
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
  double reference_scale = isfinite(expected) ? fabs(expected) : 0.0;
  if (gpu_suite_verification_add_absolute_relative_threshold(
          result, "max_abs_error", reference_scale, options->abs_tolerance,
          options->rel_tolerance) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  double maximum = 0.0;
  int finite = isfinite(expected);
  for (size_t i = 0; i < count; ++i) {
    double error = 0.0;
    if (!gpu_suite_finite_absolute_error(c[i], expected, &error) ||
        !gpu_suite_finite_max_update(error, &maximum))
      finite = 0;
  }
  result->verification_primary_metric = "max_abs_error";
  if (!finite) {
    if (gpu_suite_json_add_null(result->verification_metrics,
                                "max_abs_error") != GPU_SUITE_OK)
      return GPU_SUITE_ERROR_NOMEM;
    result->verification_status = "nonfinite";
    return GPU_SUITE_OK;
  }
  if (gpu_suite_json_add_double(result->verification_metrics, "max_abs_error",
                                maximum) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  result->verification_status =
      maximum <= options->abs_tolerance +
                     options->rel_tolerance * reference_scale
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
  gpu_suite_benchmark_writer writer;
  if (gpu_suite_benchmark_writer_open(&writer, &options, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
    fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }
  int m = 0, n = 0, k = 0;
  size_t ac = 0U, bc = 0U, cc = 0U;
  size_t ab = 0U, bb = 0U, cb = 0U;
  if (!gpu_suite_checked_u64_to_int(mu, &m) ||
      !gpu_suite_checked_u64_to_int(nu, &n) ||
      !gpu_suite_checked_u64_to_int(ku, &k) ||
      !gpu_suite_checked_mul_size((size_t)m, (size_t)k, &ac) ||
      !gpu_suite_checked_mul_size((size_t)k, (size_t)n, &bc) ||
      !gpu_suite_checked_mul_size((size_t)m, (size_t)n, &cc) ||
      !gpu_suite_checked_bytes(ac, sizeof(double), &ab) ||
      !gpu_suite_checked_bytes(bc, sizeof(double), &bb) ||
      !gpu_suite_checked_bytes(cc, sizeof(double), &cb)) {
    fprintf(stderr, "DGEMM dimensions or byte counts exceed supported range\n");
    (void)gpu_suite_benchmark_emit_unmeasured(
        &writer, &options, 0, false, "prerequisite",
        "DGEMM dimensions exceed supported integer or byte range", error,
        sizeof(error));
    (void)gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
    return EXIT_FAILURE;
  }
  double *a = malloc(ab), *b = malloc(bb), *c = malloc(cb);
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
    fill(c, cc, 1);
    int ok = 1;
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      ok = gpu_suite_measurement_start(start_ts, &start, error,
                                       sizeof(error)) == GPU_SUITE_OK;
      for (int r = 0; r < options.repeat && ok; ++r)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    options.alpha, a, m, b, k, options.beta, c, m);
      ok = ok && gpu_suite_measurement_end(&end, end_ts, error,
                                            sizeof(error)) == GPU_SUITE_OK;
      if (ok)
        elapsed = gpu_suite_clock_elapsed(&start, &end);
    } else {
      for (int r = 0; r < options.repeat && ok; ++r) {
        fill(c, cc, 1);
        ok = (r == 0 ? gpu_suite_measurement_start(start_ts, &start, error,
                                                   sizeof(error))
                     : gpu_suite_clock_now(&start, error, sizeof(error))) ==
             GPU_SUITE_OK;
        if (!ok)
          break;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    options.alpha, a, m, b, k, options.beta, c, m);
        ok = (r + 1 == options.repeat
                  ? gpu_suite_measurement_end(&end, end_ts, error,
                                              sizeof(error))
                  : gpu_suite_clock_now(&end, error, sizeof(error))) ==
             GPU_SUITE_OK;
        if (ok)
          elapsed += gpu_suite_clock_elapsed(&start, &end);
      }
    }
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
      result.failure_origin =
          pass ? NULL : (verified == GPU_SUITE_OK ? "verification" : "benchmark");
      if (verified != GPU_SUITE_OK)
        result.verification_status = "skipped";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.status = pass ? "success" : "failure";
      result.message =
          pass ? ""
               : (verified == GPU_SUITE_OK
                      ? "DGEMM verification failed"
                      : "verification result construction failed");
      any_failure |= !pass;
      ok = ok && verified == GPU_SUITE_OK;
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
