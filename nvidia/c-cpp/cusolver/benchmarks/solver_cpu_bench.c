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
static void restore(int n, int nrhs, double *a, double *b, lapack_int *piv,
                    lapack_int *getrf_info, lapack_int *getrs_info,
                    const double *a0, const double *b0) {
  memcpy(a, a0, (size_t)n * n * sizeof(*a));
  memcpy(b, b0, (size_t)n * nrhs * sizeof(*b));
  memset(piv, 0, (size_t)n * sizeof(*piv));
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
  double solution_error = 0, residual = 0, a_norm = 0, b_norm = 0;
  for (int i = 0; i < n; ++i) {
    double row_sum = 0;
    for (int j = 0; j < n; ++j)
      row_sum += fabs(a0[i + (size_t)j * n]);
    a_norm = fmax(a_norm, row_sum);
  }
  for (int r = 0; r < nrhs; ++r)
    for (int i = 0; i < n; ++i) {
      solution_error =
          fmax(solution_error, fabs(solution[i + (size_t)r * n] - 1.0));
      b_norm = fmax(b_norm, fabs(b0[i + (size_t)r * n]));
      double ax = 0;
      for (int j = 0; j < n; ++j)
        ax += a0[i + (size_t)j * n] * solution[j + (size_t)r * n];
      residual = fmax(residual, fabs(ax - b0[i + (size_t)r * n]));
    }
  double relative_residual = residual / (a_norm + b_norm);
  gpu_suite_json_free(result->verification_metrics);
  gpu_suite_json_free(result->verification_thresholds);
  result->verification_metrics = gpu_suite_json_object();
  result->verification_thresholds = gpu_suite_json_object();
  if (!options->verify) {
    result->verification_primary_metric = NULL;
    result->verification_status = "skipped";
    return 1;
  }
  gpu_suite_json_add_double(result->verification_metrics,
                            "solution_relative_error", solution_error);
  gpu_suite_json_add_double(result->verification_metrics, "relative_residual",
                            relative_residual);
  gpu_suite_json_value *s = gpu_suite_json_object(),
                       *r = gpu_suite_json_object();
  gpu_suite_json_add_string(s, "method", "absolute-plus-relative");
  gpu_suite_json_add_double(s, "reference_scale", 1);
  gpu_suite_json_add_double(s, "abs_tolerance", options->abs_tolerance);
  gpu_suite_json_add_double(s, "rel_tolerance", options->rel_tolerance);
  gpu_suite_json_add_string(r, "method", "absolute-plus-relative");
  gpu_suite_json_add_double(r, "reference_scale", 1);
  gpu_suite_json_add_double(r, "abs_tolerance", options->abs_tolerance);
  gpu_suite_json_add_double(r, "rel_tolerance", options->rel_tolerance);
  gpu_suite_json_object_set(result->verification_thresholds,
                            "solution_relative_error", s);
  gpu_suite_json_object_set(result->verification_thresholds,
                            "relative_residual", r);
  result->verification_primary_metric = "relative_residual";
  double bound = options->abs_tolerance + options->rel_tolerance;
  result->verification_status =
      isfinite(solution_error) && isfinite(relative_residual) &&
              solution_error <= bound && relative_residual <= bound
          ? "pass"
          : "failure";
  return 1;
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
      strcmp(options.cpu_backend, GPU_SUITE_COMPILED_CPU_BACKEND) != 0 ||
      options.size > INT_MAX || options.nrhs > INT_MAX) {
    fprintf(stderr, "invalid solver options or CPU backend\n");
    return EXIT_FAILURE;
  }
  int n = (int)options.size, nrhs = (int)options.nrhs;
  size_t ac = (size_t)n * n, bc = (size_t)n * nrhs;
  double *a = malloc(ac * sizeof(*a)), *b = malloc(bc * sizeof(*b)),
         *a0 = malloc(ac * sizeof(*a0)), *b0 = malloc(bc * sizeof(*b0));
  lapack_int *piv = malloc((size_t)n * sizeof(*piv));
  gpu_suite_benchmark_writer writer;
  if (gpu_suite_benchmark_writer_open(&writer, &options, error,
                                      sizeof(error)) != GPU_SUITE_OK) {
    free(piv);
    free(b0);
    free(a0);
    free(b);
    free(a);
    return EXIT_FAILURE;
  }
  if (!a || !b || !a0 || !b0 || !piv) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "dense-system allocation failed", error,
                                        sizeof(error));
    goto failed;
  }
  make_dense_system(n, nrhs, a0, b0);
  lapack_int info1 = 0, info2 = 0;
  for (int w = 0; w < options.warmup; ++w) {
    restore(n, nrhs, a, b, piv, &info1, &info2, a0, b0);
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
  restore(n, nrhs, a, b, piv, &info1, &info2, a0, b0);
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
    restore(n, nrhs, a, b, piv, &info1, &info2, a0, b0);
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
      set_verification(&result, &options, b, a0, b0, n, nrhs);
      int pass =
          info1 == 0 && info2 == 0 &&
          (!options.verify || strcmp(result.verification_status, "pass") == 0);
      result.attempted = true;
      result.failure_origin = pass ? NULL : "verification";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.status = pass ? "success" : "failure";
      result.message = pass ? "" : "LU solve verification/info failure";
      any_failure |= !pass;
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
