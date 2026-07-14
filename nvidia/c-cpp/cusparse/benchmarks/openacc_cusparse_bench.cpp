#include "gpu_suite/benchmark.hpp"
#include "sparse_bench_common.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint64_t root(uint64_t v) {
  uint64_t r = (uint64_t)std::sqrt((long double)v);
  return r;
}
static int make_poisson2d_csr(int nx, int ny, int *r, int *c, double *v) {
  int o = 0;
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      int p = y * nx + x;
      r[p] = o;
      if (y > 0) {
        c[o] = p - nx;
        v[o++] = -1;
      }
      if (x > 0) {
        c[o] = p - 1;
        v[o++] = -1;
      }
      c[o] = p;
      v[o++] = 4;
      if (x + 1 < nx) {
        c[o] = p + 1;
        v[o++] = -1;
      }
      if (y + 1 < ny) {
        c[o] = p + nx;
        v[o++] = -1;
      }
    }
  r[nx * ny] = o;
  return o;
}

static bool run_end_to_end_once(int n, int nnz, int *row_offsets,
                                size_t row_count, int *column_indices,
                                size_t column_count, double *matrix_values,
                                size_t value_count, double *x, size_t x_count,
                                double *y, size_t y_count, double alpha,
                                double beta, double *elapsed, char *error,
                                size_t error_size) {
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  cusparseDnVecDescr_t vector_x = nullptr;
  cusparseDnVecDescr_t vector_y = nullptr;
  void *workspace = nullptr;
  size_t workspace_size = 0;
  struct timespec start;
  struct timespec end;
  bool ok = gpu_suite_clock_now(&start, error, error_size) == GPU_SUITE_OK &&
            cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS;

#pragma acc data copyin(                                                       \
    row_offsets[0 : row_count], column_indices[0 : column_count],              \
    matrix_values[0 : value_count], x[0 : x_count]) copy(y[0 : y_count])
  {
#pragma acc host_data use_device(row_offsets, column_indices, matrix_values,   \
                                 x, y)
    {
      if (ok) {
        ok = cusparseCreateCsr(&matrix, n, n, nnz, row_offsets, column_indices,
                               matrix_values, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                               CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
             cusparseCreateDnVec(&vector_x, n, x, CUDA_R_64F) ==
                 CUSPARSE_STATUS_SUCCESS &&
             cusparseCreateDnVec(&vector_y, n, y, CUDA_R_64F) ==
                 CUSPARSE_STATUS_SUCCESS;
      }
    }
    if (ok) {
      ok = cusparseSpMV_bufferSize(
               handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix,
               vector_x, &beta, vector_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
               &workspace_size) == CUSPARSE_STATUS_SUCCESS &&
           (workspace_size == 0 ||
            cudaMalloc(&workspace, workspace_size) == cudaSuccess) &&
           cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                        matrix, vector_x, &beta, vector_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        workspace) == CUSPARSE_STATUS_SUCCESS &&
           cudaDeviceSynchronize() == cudaSuccess;
    }
  }
  ok = ok && gpu_suite_clock_now(&end, error, error_size) == GPU_SUITE_OK;
  if (ok)
    *elapsed = gpu_suite_clock_elapsed(&start, &end);

  if (workspace)
    cudaFree(workspace);
  if (vector_y)
    cusparseDestroyDnVec(vector_y);
  if (vector_x)
    cusparseDestroyDnVec(vector_x);
  if (matrix)
    cusparseDestroySpMat(matrix);
  if (handle)
    cusparseDestroy(handle);
  return ok;
}
// Descriptor creation follows data entry and device-address translation; only
// workspace uses cudaMalloc.
int main(int argc, char **argv) {
  gpu_suite_options o;
  char e[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_CUSPARSE,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
  auto p = gpu_suite_options_parse(&o, argc, argv, e, sizeof(e));
  if (p == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUSPARSE);
    return 0;
  }
  if (p != GPU_SUITE_PARSE_OK)
    return 1;
  int nx = (int)(o.size_set ? root(o.size) : o.nx),
      ny = (int)(o.size_set ? root(o.size) : o.ny), n = nx * ny,
      nnz = 5 * n - 2 * nx - 2 * ny;
  std::vector<int> row(n + 1), col(nnz);
  std::vector<double> val(nnz), x(n, 1), y(n, 1);
  make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data());
  int *rp = row.data(), *cp = col.data();
  double *vp = val.data(), *xp = x.data(), *yp = y.data();
  size_t rs = row.size(), cs = col.size(), vs = val.size(), xs = x.size(),
         ys = y.size();
  gpu_suite::ResultWriter w;
  if (!w.open(o))
    return 1;
  bool failed = false;

  if (cudaSetDevice(o.device) != cudaSuccess) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "cudaSetDevice failed");
    w.close();
    return 1;
  }

  if (o.scope == GPU_SUITE_SCOPE_END_TO_END) {
    for (int warmup = 0; warmup < o.warmup && !failed; ++warmup) {
      std::fill(y.begin(), y.end(), 1.0);
      double ignored_elapsed = 0.0;
      failed =
          !run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                               o.alpha, o.beta, &ignored_elapsed, e, sizeof(e));
    }
    if (failed) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "OpenACC cuSPARSE warmup failed");
    }
    for (int trial = 0; trial < o.trials && !failed; ++trial) {
      gpu_suite_result result;
      gpu_suite::initialize_result(result, o, trial);
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      double elapsed_total = 0.0;
      bool ok = gpu_suite_utc_timestamp(start_timestamp, e, sizeof(e)) ==
                GPU_SUITE_OK;
      for (int repeat = 0; repeat < o.repeat && ok; ++repeat) {
        std::fill(y.begin(), y.end(), 1.0);
        double elapsed = 0.0;
        ok = run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                                 o.alpha, o.beta, &elapsed, e, sizeof(e));
        if (ok)
          elapsed_total += elapsed;
      }
      ok = ok &&
           gpu_suite_utc_timestamp(end_timestamp, e, sizeof(e)) == GPU_SUITE_OK;
      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed_total / o.repeat);
        const bool pass =
            gpu_suite_cusparse::set_spmv_verification(result, o, y, nx, ny, 1);
        result.attempted = true;
        result.failure_origin = pass ? nullptr : "verification";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message = pass ? "" : "OpenACC SpMV verification failed";
        failed = !pass;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "OpenACC end-to-end SpMV failed";
        failed = true;
      }
      w.write(result);
      gpu_suite_result_destroy(&result);
      if (failed) {
        gpu_suite::emit_unmeasured(w, o, trial + 1, false, "prior-failure",
                                   "prior OpenACC cuSPARSE failure");
      }
    }
    if (!w.close())
      failed = true;
    return failed ? 1 : 0;
  }

  cusparseHandle_t h = nullptr;
  failed = cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS;
  int next_trial = 0;
#pragma acc data copyin(rp[0 : rs], cp[0 : cs], vp[0 : vs], xp[0 : xs])        \
    copy(yp[0 : ys])
  {
    cusparseSpMatDescr_t a = nullptr;
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    void *work = nullptr;
    size_t ws = 0;
#pragma acc host_data use_device(rp, cp, vp, xp, yp)
    {
      if (!failed) {
        failed =
            cusparseCreateCsr(&a, n, n, nnz, rp, cp, vp, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
            cusparseCreateDnVec(&vx, n, xp, CUDA_R_64F) !=
                CUSPARSE_STATUS_SUCCESS ||
            cusparseCreateDnVec(&vy, n, yp, CUDA_R_64F) !=
                CUSPARSE_STATUS_SUCCESS;
      }
    }
    if (!failed)
      failed = cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &o.alpha, a, vx, &o.beta, vy, CUDA_R_64F,
                                       CUSPARSE_SPMV_ALG_DEFAULT,
                                       &ws) != CUSPARSE_STATUS_SUCCESS ||
               (ws != 0 && cudaMalloc(&work, ws) != cudaSuccess);
    for (int warm = 0; warm < o.warmup && !failed; ++warm)
      failed =
          cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &o.alpha, a, vx,
                       &o.beta, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                       work) != CUSPARSE_STATUS_SUCCESS;
    cudaDeviceSynchronize();
    for (int t = 0; t < o.trials && !failed; ++t) {
      std::fill(y.begin(), y.end(), 1);
#pragma acc update device(yp[0 : ys])
      gpu_suite_result r;
      gpu_suite::initialize_result(r, o, t);
      char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
           et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec ts, te;
      gpu_suite_utc_timestamp(st, e, sizeof(e));
      cudaDeviceSynchronize();
      gpu_suite_clock_now(&ts, e, sizeof(e));
      bool ok = true;
      for (int rep = 0; rep < o.repeat && ok; ++rep)
        ok = cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &o.alpha, a, vx,
                          &o.beta, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                          work) == CUSPARSE_STATUS_SUCCESS;
      ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
           gpu_suite_clock_now(&te, e, sizeof(e)) == GPU_SUITE_OK;
      gpu_suite_utc_timestamp(et, e, sizeof(e));
#pragma acc update self(yp[0 : ys])
      double elapsed = gpu_suite_clock_elapsed(&ts, &te);
      r.measurement_start_timestamp = st;
      r.measurement_end_timestamp = et;
      r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
      bool pass = ok && gpu_suite_cusparse::set_spmv_verification(r, o, y, nx,
                                                                  ny, o.repeat);
      r.attempted = true;
      r.failure_origin = pass ? nullptr : (ok ? "verification" : "benchmark");
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? "" : "OpenACC SpMV failed";
      w.write(r);
      gpu_suite_result_destroy(&r);
      next_trial = t + 1;
      failed |= !pass;
      if (failed) {
        gpu_suite::emit_unmeasured(w, o, next_trial, false, "prior-failure",
                                   "prior OpenACC cuSPARSE failure");
        break;
      }
    }
    if (work)
      cudaFree(work);
    if (vy)
      cusparseDestroyDnVec(vy);
    if (vx)
      cusparseDestroyDnVec(vx);
    if (a)
      cusparseDestroySpMat(a);
  }
  if (failed && next_trial == 0) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "OpenACC cuSPARSE setup/warmup failed");
  }
  if (h)
    cusparseDestroy(h);
  if (!w.close())
    failed = true;
  return failed ? 1 : 0;
}
