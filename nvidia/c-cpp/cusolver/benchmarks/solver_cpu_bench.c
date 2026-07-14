#include "gpu_suite/gpu_suite.h"

#include GPU_SUITE_LAPACKE_HEADER

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void make_dense_system(int n, int nrhs, double *a, double *b) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? n + 1.0 : 1.0;
  for (int r = 0; r < nrhs; ++r)
    for (int i = 0; i < n; ++i)
      b[i + (size_t)r * n] = 2.0 * n;
}
static void restore(double *a, double *b, lapack_int *piv,
                    lapack_int *getrf_info, lapack_int *getrs_info,
                    const double *a0, const double *b0, size_t matrix_bytes,
                    size_t rhs_bytes, size_t pivot_bytes) {
  memcpy(a, a0, matrix_bytes);
  memcpy(b, b0, rhs_bytes);
  memset(piv, 0, pivot_bytes);
  *getrf_info = 0;
  *getrs_info = 0;
}
static void report_lapack_info(const char *api, lapack_int info) {
  if (info != 0)
    fprintf(stderr, "%s: LAPACKE info %lld\n", api, (long long)info);
}
static int set_verification(gpu_suite_result *result,
                            const gpu_suite_options *options,
                            const double *solution, const double *a0,
                            const double *b0, int n, int nrhs) {
  if (gpu_suite_verification_reset(result) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  if (!options->verify) {
    result->verification_primary_metric = NULL;
    result->verification_status = "skipped";
    return GPU_SUITE_OK;
  }
  if (gpu_suite_verification_add_absolute_relative_threshold(
          result, "solution_relative_error", 1.0, options->abs_tolerance,
          options->rel_tolerance) != GPU_SUITE_OK ||
      gpu_suite_verification_add_absolute_relative_threshold(
          result, "relative_residual", 1.0, options->abs_tolerance,
          options->rel_tolerance) != GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;

  double solution_error = 0, residual = 0, a_norm = 0, b_norm = 0;
  int finite = 1;
  for (int i = 0; i < n; ++i) {
    double row_sum = 0;
    for (int j = 0; j < n; ++j) {
      double entry = a0[i + (size_t)j * n];
      if (!isfinite(entry) || !isfinite(row_sum + fabs(entry))) {
        finite = 0;
        continue;
      }
      row_sum += fabs(entry);
    }
    if (!gpu_suite_finite_max_update(row_sum, &a_norm))
      finite = 0;
  }
  for (int r = 0; r < nrhs; ++r)
    for (int i = 0; i < n; ++i) {
      size_t index = i + (size_t)r * n;
      double error = 0.0;
      if (!gpu_suite_finite_absolute_error(solution[index], 1.0, &error) ||
          !gpu_suite_finite_max_update(error, &solution_error))
        finite = 0;
      if (!isfinite(b0[index]) ||
          !gpu_suite_finite_max_update(fabs(b0[index]), &b_norm))
        finite = 0;
      double ax = 0;
      int product_finite = 1;
      for (int j = 0; j < n; ++j) {
        double matrix_value = a0[i + (size_t)j * n];
        double solution_value = solution[j + (size_t)r * n];
        double term = matrix_value * solution_value;
        if (!isfinite(matrix_value) || !isfinite(solution_value) ||
            !isfinite(term) || !isfinite(ax + term)) {
          finite = 0;
          product_finite = 0;
          continue;
        }
        ax += term;
      }
      if (product_finite &&
          (!gpu_suite_finite_absolute_error(ax, b0[index], &error) ||
           !gpu_suite_finite_max_update(error, &residual)))
        finite = 0;
    }
  double denominator = a_norm + b_norm;
  double relative_residual = residual / denominator;
  finite = finite && isfinite(denominator) && denominator > 0.0 &&
           isfinite(relative_residual);
  result->verification_primary_metric = "relative_residual";
  if (!finite) {
    if (gpu_suite_json_add_null(result->verification_metrics,
                                "solution_relative_error") != GPU_SUITE_OK ||
        gpu_suite_json_add_null(result->verification_metrics,
                                "relative_residual") != GPU_SUITE_OK)
      return GPU_SUITE_ERROR_NOMEM;
    result->verification_status = "nonfinite";
    return GPU_SUITE_OK;
  }
  if (gpu_suite_json_add_double(result->verification_metrics,
                                "solution_relative_error", solution_error) !=
          GPU_SUITE_OK ||
      gpu_suite_json_add_double(result->verification_metrics,
                                "relative_residual", relative_residual) !=
          GPU_SUITE_OK)
    return GPU_SUITE_ERROR_NOMEM;
  double bound = options->abs_tolerance + options->rel_tolerance;
  result->verification_status =
      solution_error <= bound && relative_residual <= bound
          ? "pass"
          : "failure";
  return GPU_SUITE_OK;
}

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSOLVER,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUSOLVER);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK ||
      strcmp(options.cpu_backend, GPU_SUITE_COMPILED_CPU_BACKEND) != 0) {
    fprintf(stderr, "invalid solver options or CPU backend\n");
    return EXIT_FAILURE;
  }
  gpu_suite_benchmark_writer writer;
  if (gpu_suite_benchmark_writer_open(&writer, &options, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
    return EXIT_FAILURE;
  }
  int n = 0, nrhs = 0;
  size_t ac = 0U, bc = 0U, matrix_bytes = 0U, rhs_bytes = 0U,
         pivot_bytes = 0U;
  if (!gpu_suite_checked_u64_to_int(options.size, &n) ||
      !gpu_suite_checked_u64_to_int(options.nrhs, &nrhs) ||
      !gpu_suite_checked_mul_size((size_t)n, (size_t)n, &ac) ||
      !gpu_suite_checked_mul_size((size_t)n, (size_t)nrhs, &bc) ||
      !gpu_suite_checked_bytes(ac, sizeof(double), &matrix_bytes) ||
      !gpu_suite_checked_bytes(bc, sizeof(double), &rhs_bytes) ||
      !gpu_suite_checked_bytes((size_t)n, sizeof(lapack_int), &pivot_bytes)) {
    fprintf(stderr, "solver dimensions or byte counts exceed supported range\n");
    (void)gpu_suite_benchmark_emit_unmeasured(
        &writer, &options, 0, false, "prerequisite",
        "solver dimensions exceed supported integer or byte range", error,
        sizeof(error));
    (void)gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
    return EXIT_FAILURE;
  }
  double *a = malloc(matrix_bytes), *b = malloc(rhs_bytes),
         *a0 = malloc(matrix_bytes), *b0 = malloc(rhs_bytes);
  lapack_int *piv = malloc(pivot_bytes);
  if (!a || !b || !a0 || !b0 || !piv) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "dense-system allocation failed", error,
                                        sizeof(error));
    goto failed;
  }
  make_dense_system(n, nrhs, a0, b0);
  lapack_int info1 = 0, info2 = 0;
  for (int w = 0; w < options.warmup; ++w) {
    restore(a, b, piv, &info1, &info2, a0, b0, matrix_bytes, rhs_bytes,
            pivot_bytes);
    info1 = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, piv);
    if (info1 == 0)
      info2 = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, a, n, piv, b, n);
    report_lapack_info("warmup LAPACKE_dgetrf", info1);
    report_lapack_info("warmup LAPACKE_dgetrs", info2);
    if (info1 != 0 || info2 != 0) {
      (void)gpu_suite_benchmark_emit_unmeasured(
          &writer, &options, 0, true, "benchmark",
          "LAPACKE warmup failed", error, sizeof(error));
      goto failed;
    }
  }
  restore(a, b, piv, &info1, &info2, a0, b0, matrix_bytes, rhs_bytes,
          pivot_bytes);
  int any_failure = 0;
  for (int trial = 0; trial < options.trials; ++trial) {
    gpu_suite_result result;
    if (gpu_suite_benchmark_result_init(&result, &options, trial, error,
                                        sizeof(error)) != GPU_SUITE_OK) {
      fprintf(stderr, "gpu_suite_benchmark_result_init: %s\n", error);
      (void)gpu_suite_benchmark_emit_unmeasured(
          &writer, &options, trial, true, "benchmark",
          "gpu_suite_benchmark_result_init failed", error, sizeof(error));
      any_failure = 1;
      break;
    }
    restore(a, b, piv, &info1, &info2, a0, b0, matrix_bytes, rhs_bytes,
            pivot_bytes);
    char sts[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
         ets[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    struct timespec start, end;
    int ok = gpu_suite_measurement_start(sts, &start, error, sizeof(error)) ==
             GPU_SUITE_OK;
    if (ok) {
      info1 = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, piv);
      if (info1 == 0)
        info2 = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, a, n, piv, b, n);
      report_lapack_info("LAPACKE_dgetrf", info1);
      report_lapack_info("LAPACKE_dgetrs", info2);
      ok = gpu_suite_measurement_end(&end, ets, error, sizeof(error)) ==
           GPU_SUITE_OK;
    }
    if (ok) {
      double elapsed = gpu_suite_clock_elapsed(&start, &end);
      result.measurement_start_timestamp = sts;
      result.measurement_end_timestamp = ets;
      result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
      result.getrf_info = gpu_suite_optional_int_value(info1);
      result.getrs_info = gpu_suite_optional_int_value(info2);
      int verified =
          set_verification(&result, &options, b, a0, b0, n, nrhs);
      if (verified == GPU_SUITE_OK && (info1 != 0 || info2 != 0))
        result.verification_status = "failure";
      int pass = info1 == 0 && info2 == 0 && verified == GPU_SUITE_OK &&
                 (!options.verify ||
                  strcmp(result.verification_status, "pass") == 0);
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
                      ? "LU solve verification/info failure"
                      : "verification result construction failed");
      any_failure |= !pass;
      ok = ok && verified == GPU_SUITE_OK;
    } else {
      result.attempted = true;
      result.failure_origin = "benchmark";
      result.verification_status = "skipped";
      result.exit_code = gpu_suite_optional_int_value(1);
      result.status = "failure";
      result.message = "solver timing failed";
      any_failure = 1;
    }
    if (gpu_suite_benchmark_writer_write(&writer, &result, error,
                                         sizeof(error)) != GPU_SUITE_OK) {
      fprintf(stderr, "gpu_suite_benchmark_writer_write: %s\n", error);
      any_failure = 1;
      gpu_suite_result_destroy(&result);
      break;
    }
    gpu_suite_result_destroy(&result);
    if (!ok) {
      gpu_suite_benchmark_emit_unmeasured(
          &writer, &options, trial + 1, false, "prior-failure",
          "prior solver failure", error, sizeof(error));
      break;
    }
  }
  free(piv);
  free(b0);
  free(a0);
  free(b);
  free(a);
  if (gpu_suite_benchmark_writer_close(&writer, error, sizeof(error)) !=
      GPU_SUITE_OK) {
    fprintf(stderr, "gpu_suite_benchmark_writer_close: %s\n", error);
    any_failure = 1;
  }
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
failed:
  free(piv);
  free(b0);
  free(a0);
  free(b);
  free(a);
  gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
  return EXIT_FAILURE;
}
