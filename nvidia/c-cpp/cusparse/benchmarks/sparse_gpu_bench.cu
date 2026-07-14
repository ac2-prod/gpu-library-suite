#include "gpu_suite/benchmark.hpp"
#include "sparse_bench_common.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
static uint64_t root(uint64_t v) {
  uint64_t r = (uint64_t)std::sqrt((long double)v);
  while ((r + 1) <= v / (r + 1))
    ++r;
  while (r > v / r)
    --r;
  return r;
}
struct Context {
  cusparseHandle_t h = nullptr;
  cusparseSpMatDescr_t a = nullptr;
  cusparseDnVecDescr_t x = nullptr, y = nullptr;
  int *dr = nullptr, *dc = nullptr;
  double *dv = nullptr, *dx = nullptr, *dy = nullptr;
  void *work = nullptr;
  size_t ws = 0;
};
static bool cuda_ok(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}
static bool sparse_ok(cusparseStatus_t status, const char *api) {
  if (status == CUSPARSE_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuSPARSE status %d\n", api, (int)status);
  return false;
}
static void destroy(Context &q) {
  if (q.work)
    cudaFree(q.work);
  if (q.y)
    cusparseDestroyDnVec(q.y);
  if (q.x)
    cusparseDestroyDnVec(q.x);
  if (q.a)
    cusparseDestroySpMat(q.a);
  if (q.h)
    cusparseDestroy(q.h);
  if (q.dy)
    cudaFree(q.dy);
  if (q.dx)
    cudaFree(q.dx);
  if (q.dv)
    cudaFree(q.dv);
  if (q.dc)
    cudaFree(q.dc);
  if (q.dr)
    cudaFree(q.dr);
  q = Context{};
}
static bool create(Context &q, int n, int nnz, const std::vector<int> &r,
                   const std::vector<int> &c, const std::vector<double> &v,
                   const std::vector<double> &x, const std::vector<double> &y,
                   double alpha, double beta) {
  if (!cuda_ok(cudaMalloc((void **)&q.dr, r.size() * sizeof(int)),
               "cudaMalloc CSR row") ||
      !cuda_ok(cudaMalloc((void **)&q.dc, c.size() * sizeof(int)),
               "cudaMalloc CSR column") ||
      !cuda_ok(cudaMalloc((void **)&q.dv, v.size() * sizeof(double)),
               "cudaMalloc CSR value") ||
      !cuda_ok(cudaMalloc((void **)&q.dx, x.size() * sizeof(double)),
               "cudaMalloc x") ||
      !cuda_ok(cudaMalloc((void **)&q.dy, y.size() * sizeof(double)),
               "cudaMalloc y") ||
      !cuda_ok(cudaMemcpy(q.dr, r.data(), r.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy CSR row") ||
      !cuda_ok(cudaMemcpy(q.dc, c.data(), c.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy CSR column") ||
      !cuda_ok(cudaMemcpy(q.dv, v.data(), v.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy CSR value") ||
      !cuda_ok(cudaMemcpy(q.dx, x.data(), x.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy x") ||
      !cuda_ok(cudaMemcpy(q.dy, y.data(), y.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy y") ||
      !sparse_ok(cusparseCreate(&q.h), "cusparseCreate") ||
      !sparse_ok(cusparseCreateCsr(&q.a, n, n, nnz, q.dr, q.dc, q.dv,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                 "cusparseCreateCsr") ||
      !sparse_ok(cusparseCreateDnVec(&q.x, n, q.dx, CUDA_R_64F),
                 "cusparseCreateDnVec x") ||
      !sparse_ok(cusparseCreateDnVec(&q.y, n, q.dy, CUDA_R_64F),
                 "cusparseCreateDnVec y") ||
      !sparse_ok(cusparseSpMV_bufferSize(
                     q.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, q.a, q.x,
                     &beta, q.y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &q.ws),
                 "cusparseSpMV_bufferSize"))
    return false;
  return q.ws == 0 || cuda_ok(cudaMalloc(&q.work, q.ws),
                              "cudaMalloc cuSPARSE workspace");
}
static bool apply(Context &q, double a, double b) {
  return sparse_ok(cusparseSpMV(q.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &a, q.a,
                                q.x, &b, q.y, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, q.work),
                   "cusparseSpMV");
}
int main(int argc, char **argv) {
  gpu_suite_options o;
  char e[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_CUSPARSE,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
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
  gpu_suite::ResultWriter w;
  if (!w.open(o))
    return 1;
  Context q;
  bool fatal_failure =
      !cuda_ok(cudaSetDevice(o.device), "cudaSetDevice") ||
      !create(q, n, nnz, row, col, val, x, y, o.alpha, o.beta);
  bool any_failure = fatal_failure;
  for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm)
    fatal_failure = !apply(q, o.alpha, o.beta);
  if (!fatal_failure)
    fatal_failure = !cuda_ok(cudaDeviceSynchronize(), "warmup synchronize");
  if (fatal_failure) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "cuSPARSE setup/warmup failed");
    any_failure = true;
  }
  for (int t = 0; t < o.trials && !fatal_failure; ++t) {
    gpu_suite_result r;
    if (!gpu_suite::initialize_result(r, o, t)) {
      any_failure = true;
      fatal_failure = true;
      break;
    }
    std::fill(y.begin(), y.end(), 1);
    char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
         et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    struct timespec ts, te;
    double elapsed = 0;
    bool ok = true;
    if (o.scope == GPU_SUITE_SCOPE_COMPUTE) {
      ok = cuda_ok(cudaMemcpy(q.dy, y.data(), y.size() * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "restore y") &&
           cuda_ok(cudaDeviceSynchronize(), "pre-timing synchronize") &&
           gpu_suite_measurement_start(st, &ts, e, sizeof(e)) == GPU_SUITE_OK;
      for (int rep = 0; rep < o.repeat && ok; ++rep)
        ok = apply(q, o.alpha, o.beta);
      ok = ok && cuda_ok(cudaDeviceSynchronize(), "post-timing synchronize") &&
           gpu_suite_measurement_end(&te, et, e, sizeof(e)) == GPU_SUITE_OK;
      if (ok)
        elapsed = gpu_suite_clock_elapsed(&ts, &te);
      if (ok)
        ok = cuda_ok(cudaMemcpy(y.data(), q.dy, y.size() * sizeof(double),
                               cudaMemcpyDeviceToHost),
                     "copy result y");
    } else {
      for (int rep = 0; rep < o.repeat && ok; ++rep) {
        std::fill(y.begin(), y.end(), 1);
        Context temp;
        ok = (rep == 0 ? gpu_suite_measurement_start(st, &ts, e, sizeof(e))
                       : gpu_suite_clock_now(&ts, e, sizeof(e))) ==
                 GPU_SUITE_OK &&
             create(temp, n, nnz, row, col, val, x, y, o.alpha, o.beta) &&
             apply(temp, o.alpha, o.beta) &&
             cuda_ok(cudaDeviceSynchronize(), "end-to-end synchronize") &&
             cuda_ok(cudaMemcpy(y.data(), temp.dy, y.size() * sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "copy end-to-end y");
        if (ok)
          ok = (rep + 1 == o.repeat
                    ? gpu_suite_measurement_end(&te, et, e, sizeof(e))
                    : gpu_suite_clock_now(&te, e, sizeof(e))) == GPU_SUITE_OK;
        if (ok)
          elapsed += gpu_suite_clock_elapsed(&ts, &te);
        destroy(temp);
      }
    }
    int updates = o.scope == GPU_SUITE_SCOPE_COMPUTE ? o.repeat : 1;
    if (ok) {
      r.measurement_start_timestamp = st;
      r.measurement_end_timestamp = et;
      r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
      bool pass =
          gpu_suite_cusparse::set_spmv_verification(r, o, y, nx, ny, updates);
      r.attempted = true;
      r.failure_origin = pass ? nullptr : "verification";
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? "" : "SpMV verification failed";
      any_failure = any_failure || !pass;
    } else {
      r.attempted = true;
      r.failure_origin = "benchmark";
      r.verification_status = "skipped";
      r.exit_code = gpu_suite_optional_int_value(1);
      r.status = "failure";
      r.message = "cuSPARSE pipeline failed";
      any_failure = true;
      fatal_failure = true;
    }
    if (!w.write(r)) {
      any_failure = true;
      fatal_failure = true;
    }
    gpu_suite_result_destroy(&r);
    if (fatal_failure) {
      gpu_suite::emit_unmeasured(w, o, t + 1, false, "prior-failure",
                                 "prior cuSPARSE failure");
      break;
    }
  }
  destroy(q);
  if (!w.close())
    any_failure = true;
  return any_failure ? 1 : 0;
}
