#include "gpu_suite/gpu_suite.h"

#ifdef GPU_SUITE_USE_MKL_SPARSE
#include <mkl_spblas.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t integer_root(uint64_t value) {
  uint64_t lo = 0, hi = value < 4294967295ULL ? value : 4294967295ULL, ans = 0;
  while (lo <= hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    if (mid == 0 || mid <= value / mid) {
      ans = mid;
      lo = mid + 1;
    } else
      hi = mid - 1;
  }
  return ans;
}
static int make_poisson2d_csr(int nx, int ny, int *row, int *col, double *val) {
  int off = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int p = iy * nx + ix;
      row[p] = off;
      if (iy > 0) {
        col[off] = p - nx;
        val[off++] = -1;
      }
      if (ix > 0) {
        col[off] = p - 1;
        val[off++] = -1;
      }
      col[off] = p;
      val[off++] = 4;
      if (ix + 1 < nx) {
        col[off] = p + 1;
        val[off++] = -1;
      }
      if (iy + 1 < ny) {
        col[off] = p + nx;
        val[off++] = -1;
      }
    }
  row[nx * ny] = off;
  return off;
}
static int validate_csr(int n, int nnz, const int *row, const int *col,
                        const double *val) {
  if (row[0] != 0 || row[n] != nnz)
    return 0;
  for (int i = 0; i < n; ++i) {
    if (row[i] > row[i + 1])
      return 0;
    for (int j = row[i]; j < row[i + 1]; ++j) {
      if (col[j] < 0 || col[j] >= n || !isfinite(val[j]) ||
          (j > row[i] && col[j - 1] >= col[j]))
        return 0;
    }
  }
  return 1;
}
static void fill(double *x, int n, double value) {
  for (int i = 0; i < n; ++i)
    x[i] = value;
}
#ifdef GPU_SUITE_USE_MKL_SPARSE
typedef struct {
  sparse_matrix_t matrix;
  struct matrix_descr descriptor;
} sparse_context;
static int context_create(sparse_context *ctx, int n, int *row, int *col,
                          double *val) {
  ctx->matrix = NULL;
  ctx->descriptor.type = SPARSE_MATRIX_TYPE_GENERAL;
  ctx->descriptor.mode = SPARSE_FILL_MODE_FULL;
  ctx->descriptor.diag = SPARSE_DIAG_NON_UNIT;
  return mkl_sparse_d_create_csr(&ctx->matrix, SPARSE_INDEX_BASE_ZERO, n, n,
                                 row, row + 1, col,
                                 val) == SPARSE_STATUS_SUCCESS &&
         mkl_sparse_set_mv_hint(ctx->matrix, SPARSE_OPERATION_NON_TRANSPOSE,
                                ctx->descriptor, 1) == SPARSE_STATUS_SUCCESS &&
         mkl_sparse_optimize(ctx->matrix) == SPARSE_STATUS_SUCCESS;
}
static int apply(sparse_context *ctx, double alpha, const double *x,
                 double beta, double *y) {
  return mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, ctx->matrix,
                         ctx->descriptor, x, beta, y) == SPARSE_STATUS_SUCCESS;
}
static void context_destroy(sparse_context *ctx) {
  if (ctx->matrix)
    mkl_sparse_destroy(ctx->matrix);
  ctx->matrix = NULL;
}
#else
typedef struct {
  int n;
  const int *row, *col;
  const double *val;
} sparse_context;
static int context_create(sparse_context *ctx, int n, int *row, int *col,
                          double *val) {
  ctx->n = n;
  ctx->row = row;
  ctx->col = col;
  ctx->val = val;
  return 1;
}
static int apply(sparse_context *ctx, double alpha, const double *x,
                 double beta, double *y) {
  for (int i = 0; i < ctx->n; ++i) {
    double sum = 0;
    for (int j = ctx->row[i]; j < ctx->row[i + 1]; ++j)
      sum += ctx->val[j] * x[ctx->col[j]];
    y[i] = alpha * sum + beta * y[i];
  }
  return 1;
}
static void context_destroy(sparse_context *ctx) { (void)ctx; }
#endif

static int verify_result(gpu_suite_result *result,
                         const gpu_suite_options *options, const double *y,
                         int nx, int ny) {
  gpu_suite_json_free(result->verification_metrics);
  gpu_suite_json_free(result->verification_thresholds);
  result->verification_metrics = gpu_suite_json_object();
  result->verification_thresholds = gpu_suite_json_object();
  if (!result->verification_metrics || !result->verification_thresholds)
    return 0;
  if (!options->verify) {
    result->verification_primary_metric = NULL;
    result->verification_status = "skipped";
    return 1;
  }
  int updates = options->scope == GPU_SUITE_SCOPE_COMPUTE ? options->repeat : 1;
  double maximum = 0, scale = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int neighbors =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      double base = 4.0 - neighbors, expected = 1.0;
      for (int r = 0; r < updates; ++r)
        expected = options->alpha * base + options->beta * expected;
      maximum = fmax(maximum, fabs(y[iy * nx + ix] - expected));
      scale = fmax(scale, fabs(expected));
    }
  gpu_suite_json_add_double(result->verification_metrics, "max_abs_error",
                            maximum);
  gpu_suite_json_value *t = gpu_suite_json_object();
  gpu_suite_json_add_string(t, "method", "absolute-plus-relative");
  gpu_suite_json_add_double(t, "reference_scale", scale);
  gpu_suite_json_add_double(t, "abs_tolerance", options->abs_tolerance);
  gpu_suite_json_add_double(t, "rel_tolerance", options->rel_tolerance);
  gpu_suite_json_object_set(result->verification_thresholds, "max_abs_error",
                            t);
  result->verification_primary_metric = "max_abs_error";
  result->verification_status =
      isfinite(maximum) &&
              maximum <= options->abs_tolerance + options->rel_tolerance * scale
          ? "pass"
          : "failure";
  return 1;
}

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSPARSE,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUSPARSE);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK ||
      strcmp(options.cpu_backend, GPU_SUITE_COMPILED_CPU_BACKEND) != 0) {
    fprintf(stderr, "%s%s\n", error,
            parsed == GPU_SUITE_PARSE_OK ? "CPU backend mismatch" : "");
    return EXIT_FAILURE;
  }
  uint64_t nxu = options.size_set ? integer_root(options.size) : options.nx,
           nyu = options.size_set ? integer_root(options.size) : options.ny;
  if (nxu > 2147483647ULL || nyu > 2147483647ULL || nxu * nyu > 2147483647ULL)
    return EXIT_FAILURE;
  int nx = (int)nxu, ny = (int)nyu, n = nx * ny, nnz = 5 * n - 2 * nx - 2 * ny;
  int *row = malloc((size_t)(n + 1) * sizeof(*row)),
      *col = malloc((size_t)nnz * sizeof(*col));
  double *val = malloc((size_t)nnz * sizeof(*val)),
         *x = malloc((size_t)n * sizeof(*x)),
         *y = malloc((size_t)n * sizeof(*y));
  gpu_suite_benchmark_writer writer;
  if (gpu_suite_benchmark_writer_open(&writer, &options, error,
                                      sizeof(error)) != GPU_SUITE_OK)
    return EXIT_FAILURE;
  if (!row || !col || !val || !x || !y) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "CSR allocation failed", error,
                                        sizeof(error));
    goto failed;
  }
  if (make_poisson2d_csr(nx, ny, row, col, val) != nnz ||
      !validate_csr(n, nnz, row, col, val)) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "CSR validation failed", error,
                                        sizeof(error));
    goto failed;
  }
  fill(x, n, 1);
  fill(y, n, 1);
  sparse_context compute;
  if (!context_create(&compute, n, row, col, val)) {
    gpu_suite_benchmark_emit_unmeasured(&writer, &options, 0, true, "benchmark",
                                        "sparse descriptor setup failed", error,
                                        sizeof(error));
    goto failed;
  }
  for (int w = 0; w < options.warmup; ++w)
    if (!apply(&compute, options.alpha, x, options.beta, y)) {
      context_destroy(&compute);
      gpu_suite_benchmark_emit_unmeasured(
          &writer, &options, 0, true, "benchmark", "SpMV warmup failed", error,
          sizeof(error));
      goto failed;
    }
  fill(y, n, 1);
  int any_failure = 0;
  for (int trial = 0; trial < options.trials; ++trial) {
    gpu_suite_result result;
    gpu_suite_benchmark_result_init(&result, &options, trial, error,
                                    sizeof(error));
    char sts[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
         ets[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    struct timespec start, end;
    double elapsed = 0;
    int ok = gpu_suite_utc_timestamp(sts, error, sizeof(error)) == GPU_SUITE_OK;
    fill(y, n, 1);
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE && ok) {
      ok = gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK;
      for (int r = 0; r < options.repeat && ok; ++r)
        ok = apply(&compute, options.alpha, x, options.beta, y);
      ok =
          ok && gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
      if (ok)
        elapsed = gpu_suite_clock_elapsed(&start, &end);
    } else if (ok) {
      for (int r = 0; r < options.repeat && ok; ++r) {
        fill(y, n, 1);
        sparse_context temporary = {0};
        ok =
            gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK &&
            context_create(&temporary, n, row, col, val);
        if (ok)
          ok = apply(&temporary, options.alpha, x, options.beta, y);
        if (ok)
          ok = gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
        context_destroy(&temporary);
        if (ok)
          elapsed += gpu_suite_clock_elapsed(&start, &end);
      }
    }
    ok = ok &&
         gpu_suite_utc_timestamp(ets, error, sizeof(error)) == GPU_SUITE_OK;
    if (ok) {
      result.measurement_start_timestamp = sts;
      result.measurement_end_timestamp = ets;
      result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      result.elapsed_sec =
          gpu_suite_optional_double_value(elapsed / options.repeat);
      verify_result(&result, &options, y, nx, ny);
      int pass =
          !options.verify || strcmp(result.verification_status, "pass") == 0;
      result.attempted = true;
      result.failure_origin = pass ? NULL : "verification";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.status = pass ? "success" : "failure";
      result.message = pass ? "" : "SpMV verification failed";
      any_failure |= !pass;
    } else {
      result.attempted = true;
      result.failure_origin = "benchmark";
      result.verification_status = "skipped";
      result.exit_code = gpu_suite_optional_int_value(1);
      result.status = "failure";
      result.message = "SpMV/timing failed";
      any_failure = 1;
    }
    gpu_suite_benchmark_writer_write(&writer, &result, error, sizeof(error));
    gpu_suite_result_destroy(&result);
    if (!ok) {
      gpu_suite_benchmark_emit_unmeasured(&writer, &options, trial + 1, false,
                                          "prior-failure", "prior SpMV failure",
                                          error, sizeof(error));
      break;
    }
  }
  context_destroy(&compute);
  free(y);
  free(x);
  free(val);
  free(col);
  free(row);
  gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
failed:
  free(y);
  free(x);
  free(val);
  free(col);
  free(row);
  gpu_suite_benchmark_writer_close(&writer, error, sizeof(error));
  return EXIT_FAILURE;
}
