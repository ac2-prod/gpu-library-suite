#include "gpu_suite/benchmark.hpp"
#include "sparse_bench_common.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool cuda_success(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}

static bool cusparse_success(cusparseStatus_t status, const char *api) {
  if (status == CUSPARSE_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuSPARSE status %d\n", api,
               static_cast<int>(status));
  return false;
}

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
                                size_t error_size, bool first_repeat,
                                bool last_repeat,
                                char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                                char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY]) {
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  cusparseDnVecDescr_t vector_x = nullptr;
  cusparseDnVecDescr_t vector_y = nullptr;
  void *workspace = nullptr;
  size_t workspace_size = 0;
  struct timespec start;
  struct timespec end;
  bool ok = (first_repeat
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK &&
            cusparse_success(cusparseCreate(&handle), "cusparseCreate");

#pragma acc data copyin(                                                       \
    row_offsets[0 : row_count], column_indices[0 : column_count],              \
    matrix_values[0 : value_count], x[0 : x_count]) copy(y[0 : y_count])
  {
#pragma acc host_data use_device(row_offsets, column_indices, matrix_values,   \
                                 x, y)
    {
      if (ok) {
        ok = cusparse_success(
                 cusparseCreateCsr(&matrix, n, n, nnz, row_offsets,
                                   column_indices, matrix_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                 "cusparseCreateCsr") &&
             cusparse_success(cusparseCreateDnVec(&vector_x, n, x, CUDA_R_64F),
                              "cusparseCreateDnVec x") &&
             cusparse_success(cusparseCreateDnVec(&vector_y, n, y, CUDA_R_64F),
                              "cusparseCreateDnVec y");
      }
    }
    if (ok) {
      ok = cusparse_success(
               cusparseSpMV_bufferSize(
                   handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix,
                   vector_x, &beta, vector_y, CUDA_R_64F,
                   CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size),
               "cusparseSpMV_bufferSize") &&
           (workspace_size == 0 ||
            cuda_success(cudaMalloc(&workspace, workspace_size),
                         "cudaMalloc cuSPARSE workspace")) &&
           cusparse_success(
               cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matrix, vector_x, &beta, vector_y, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, workspace),
               "cusparseSpMV") &&
           cuda_success(cudaDeviceSynchronize(),
                        "end-to-end synchronize");
    }
  }
  ok = ok &&
       (last_repeat
            ? gpu_suite_measurement_end(&end, end_timestamp, error, error_size)
            : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
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
  bool fatal_failure = false;
  bool any_failure = false;

  if (!cuda_success(cudaSetDevice(o.device), "cudaSetDevice")) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "cudaSetDevice failed");
    w.close();
    return 1;
  }

  if (o.scope == GPU_SUITE_SCOPE_END_TO_END) {
    for (int warmup = 0; warmup < o.warmup && !fatal_failure; ++warmup) {
      std::fill(y.begin(), y.end(), 1.0);
      double ignored_elapsed = 0.0;
      char ignored_start[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char ignored_end[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      fatal_failure =
          !run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                               o.alpha, o.beta, &ignored_elapsed, e, sizeof(e),
                               false, false, ignored_start, ignored_end);
    }
    if (fatal_failure) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "OpenACC cuSPARSE warmup failed");
      any_failure = true;
    }
    for (int trial = 0; trial < o.trials && !fatal_failure; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, o, trial)) {
        fatal_failure = true;
        any_failure = true;
        break;
      }
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      double elapsed_total = 0.0;
      bool ok = true;
      for (int repeat = 0; repeat < o.repeat && ok; ++repeat) {
        std::fill(y.begin(), y.end(), 1.0);
        double elapsed = 0.0;
        ok = run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                                 o.alpha, o.beta, &elapsed, e, sizeof(e),
                                 repeat == 0, repeat + 1 == o.repeat,
                                 start_timestamp, end_timestamp);
        if (ok)
          elapsed_total += elapsed;
      }
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
        any_failure = any_failure || !pass;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "OpenACC end-to-end SpMV failed";
        fatal_failure = true;
        any_failure = true;
      }
      if (!w.write(result)) {
        fatal_failure = true;
        any_failure = true;
      }
      gpu_suite_result_destroy(&result);
      if (fatal_failure) {
        gpu_suite::emit_unmeasured(w, o, trial + 1, false, "prior-failure",
                                   "prior OpenACC cuSPARSE failure");
      }
    }
    if (!w.close())
      any_failure = true;
    return any_failure ? 1 : 0;
  }

  cusparseHandle_t h = nullptr;
  fatal_failure = !cusparse_success(cusparseCreate(&h), "cusparseCreate");
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
      if (!fatal_failure) {
        fatal_failure =
            !cusparse_success(
                cusparseCreateCsr(&a, n, n, nnz, rp, cp, vp,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr") ||
            !cusparse_success(cusparseCreateDnVec(&vx, n, xp, CUDA_R_64F),
                              "cusparseCreateDnVec x") ||
            !cusparse_success(cusparseCreateDnVec(&vy, n, yp, CUDA_R_64F),
                              "cusparseCreateDnVec y");
      }
    }
    if (!fatal_failure)
      fatal_failure =
          !cusparse_success(
              cusparseSpMV_bufferSize(
                  h, CUSPARSE_OPERATION_NON_TRANSPOSE, &o.alpha, a, vx,
                  &o.beta, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &ws),
              "cusparseSpMV_bufferSize") ||
          (ws != 0 && !cuda_success(cudaMalloc(&work, ws),
                                    "cudaMalloc cuSPARSE workspace"));
    for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm)
      fatal_failure = !cusparse_success(
          cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &o.alpha, a, vx,
                       &o.beta, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                       work),
          "warmup cusparseSpMV");
    if (!fatal_failure)
      fatal_failure = !cuda_success(cudaDeviceSynchronize(),
                                    "warmup synchronize");
    for (int t = 0; t < o.trials && !fatal_failure; ++t) {
      std::fill(y.begin(), y.end(), 1);
#pragma acc update device(yp[0 : ys])
      gpu_suite_result r;
      if (!gpu_suite::initialize_result(r, o, t)) {
        fatal_failure = true;
        any_failure = true;
        break;
      }
      char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
           et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec ts, te;
      bool ok = cuda_success(cudaDeviceSynchronize(),
                             "OpenACC update device y") &&
                gpu_suite_measurement_start(st, &ts, e, sizeof(e)) ==
                    GPU_SUITE_OK;
      for (int rep = 0; rep < o.repeat && ok; ++rep)
        ok = cusparse_success(
            cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &o.alpha, a, vx,
                         &o.beta, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                         work),
            "cusparseSpMV");
      ok = ok && cuda_success(cudaDeviceSynchronize(),
                              "post-timing synchronize") &&
           gpu_suite_measurement_end(&te, et, e, sizeof(e)) == GPU_SUITE_OK;
#pragma acc update self(yp[0 : ys])
      ok = ok && cuda_success(cudaDeviceSynchronize(),
                              "OpenACC update self y");
      bool pass = false;
      if (ok) {
        const double elapsed = gpu_suite_clock_elapsed(&ts, &te);
        r.measurement_start_timestamp = st;
        r.measurement_end_timestamp = et;
        r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
        pass = gpu_suite_cusparse::set_spmv_verification(r, o, y, nx, ny,
                                                         o.repeat);
      } else {
        r.verification_status = "skipped";
      }
      r.attempted = true;
      r.failure_origin = pass ? nullptr : (ok ? "verification" : "benchmark");
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? "" : "OpenACC SpMV failed";
      if (!w.write(r)) {
        fatal_failure = true;
        any_failure = true;
      }
      gpu_suite_result_destroy(&r);
      next_trial = t + 1;
      any_failure = any_failure || !pass;
      if (!ok)
        fatal_failure = true;
      if (fatal_failure) {
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
  if (fatal_failure && next_trial == 0) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "OpenACC cuSPARSE setup/warmup failed");
  }
  if (h)
    cusparseDestroy(h);
  if (!w.close())
    any_failure = true;
  if (fatal_failure)
    any_failure = true;
  return any_failure ? 1 : 0;
}
