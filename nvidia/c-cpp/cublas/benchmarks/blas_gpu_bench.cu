#include "gpu_suite/benchmark.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static bool cuda_ok(cudaError_t s, const char *m) {
  if (s == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(s));
  return false;
}
static bool blas_ok(cublasStatus_t s, const char *m) {
  if (s == CUBLAS_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cublas status %d\n", m, (int)s);
  return false;
}
static void fill(std::vector<double> &v, double x) {
  std::fill(v.begin(), v.end(), x);
}
static bool verify(gpu_suite_result &r, const gpu_suite_options &o,
                   const std::vector<double> &c, int k) {
  gpu_suite_json_free(r.verification_metrics);
  gpu_suite_json_free(r.verification_thresholds);
  r.verification_metrics = gpu_suite_json_object();
  r.verification_thresholds = gpu_suite_json_object();
  if (!o.verify) {
    r.verification_status = "skipped";
    return true;
  }
  int updates = o.scope == GPU_SUITE_SCOPE_COMPUTE ? o.repeat : 1;
  double expected = 1;
  for (int i = 0; i < updates; ++i)
    expected = o.alpha * k + o.beta * expected;
  double e = 0;
  for (double x : c)
    e = std::max(e, std::fabs(x - expected));
  gpu_suite::json_add_double(r.verification_metrics, "max_abs_error", e);
  auto *t = gpu_suite_json_object();
  gpu_suite::json_add_string(t, "method", "absolute-plus-relative");
  gpu_suite::json_add_double(t, "reference_scale", std::fabs(expected));
  gpu_suite::json_add_double(t, "abs_tolerance", o.abs_tolerance);
  gpu_suite::json_add_double(t, "rel_tolerance", o.rel_tolerance);
  gpu_suite_json_object_set(r.verification_thresholds, "max_abs_error", t);
  r.verification_primary_metric = "max_abs_error";
  bool pass = std::isfinite(e) &&
              e <= o.abs_tolerance + o.rel_tolerance * std::fabs(expected);
  r.verification_status = pass ? "pass" : "failure";
  return pass;
}

int main(int argc, char **argv) {
  gpu_suite_options o;
  char error[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
  auto p = gpu_suite_options_parse(&o, argc, argv, error, sizeof(error));
  if (p == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUBLAS);
    return 0;
  }
  if (p != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    return 1;
  }
  int m = (int)(o.size_set ? o.size : o.m),
      n = (int)(o.size_set ? o.size : o.n),
      k = (int)(o.size_set ? o.size : o.k);
  gpu_suite::ResultWriter w;
  if (!w.open(o))
    return 1;
  bool failed = false;
  try {
    std::vector<double> a((size_t)m * k, 1), b((size_t)k * n, 1),
        c((size_t)m * n, 1);
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cublasHandle_t h = nullptr;
    size_t ab = a.size() * sizeof(double), bb = b.size() * sizeof(double),
           cb = c.size() * sizeof(double);
    if (!cuda_ok(cudaSetDevice(o.device), "cudaSetDevice") ||
        !cuda_ok(cudaMalloc((void **)&da, ab), "cudaMalloc A") ||
        !cuda_ok(cudaMalloc((void **)&db, bb), "cudaMalloc B") ||
        !cuda_ok(cudaMalloc((void **)&dc, cb), "cudaMalloc C") ||
        !cuda_ok(cudaMemcpy(da, a.data(), ab, cudaMemcpyHostToDevice),
                 "copy A") ||
        !cuda_ok(cudaMemcpy(db, b.data(), bb, cudaMemcpyHostToDevice),
                 "copy B") ||
        !blas_ok(cublasCreate(&h), "cublasCreate")) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "cuBLAS setup failed");
      failed = true;
    } else {
      for (int warm = 0; warm < o.warmup; ++warm) {
        cudaMemcpy(dc, c.data(), cb, cudaMemcpyHostToDevice);
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &o.alpha, da, m, db,
                    k, &o.beta, dc, m);
      }
      cudaDeviceSynchronize();
      for (int trial = 0; trial < o.trials && !failed; ++trial) {
        gpu_suite_result r;
        gpu_suite::initialize_result(r, o, trial);
        fill(c, 1);
        char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
             et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec ts, te;
        double elapsed = 0;
        bool ok =
            gpu_suite_utc_timestamp(st, error, sizeof(error)) == GPU_SUITE_OK;
        if (o.scope == GPU_SUITE_SCOPE_COMPUTE && ok) {
          ok = cuda_ok(cudaMemcpy(dc, c.data(), cb, cudaMemcpyHostToDevice),
                       "restore C") &&
               cuda_ok(cudaDeviceSynchronize(), "sync before") &&
               gpu_suite_clock_now(&ts, error, sizeof(error)) == GPU_SUITE_OK;
          for (int rep = 0; rep < o.repeat && ok; ++rep)
            ok = blas_ok(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &o.alpha, da, m, db, k, &o.beta, dc, m),
                         "cublasDgemm");
          ok = ok && cuda_ok(cudaDeviceSynchronize(), "sync after") &&
               gpu_suite_clock_now(&te, error, sizeof(error)) == GPU_SUITE_OK;
          if (ok)
            elapsed = gpu_suite_clock_elapsed(&ts, &te);
        } else if (ok) {
          for (int rep = 0; rep < o.repeat && ok; ++rep) {
            fill(c, 1);
            double *ta = nullptr, *tb = nullptr, *tc = nullptr;
            cublasHandle_t th = nullptr;
            ok = gpu_suite_clock_now(&ts, error, sizeof(error)) ==
                     GPU_SUITE_OK &&
                 cuda_ok(cudaMalloc((void **)&ta, ab), "alloc A") &&
                 cuda_ok(cudaMalloc((void **)&tb, bb), "alloc B") &&
                 cuda_ok(cudaMalloc((void **)&tc, cb), "alloc C") &&
                 cuda_ok(cudaMemcpy(ta, a.data(), ab, cudaMemcpyHostToDevice),
                         "copy A") &&
                 cuda_ok(cudaMemcpy(tb, b.data(), bb, cudaMemcpyHostToDevice),
                         "copy B") &&
                 cuda_ok(cudaMemcpy(tc, c.data(), cb, cudaMemcpyHostToDevice),
                         "copy C") &&
                 blas_ok(cublasCreate(&th), "create") &&
                 blas_ok(cublasDgemm(th, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &o.alpha, ta, m, tb, k, &o.beta, tc, m),
                         "gemm") &&
                 cuda_ok(cudaDeviceSynchronize(), "sync") &&
                 cuda_ok(cudaMemcpy(c.data(), tc, cb, cudaMemcpyDeviceToHost),
                         "copy out") &&
                 gpu_suite_clock_now(&te, error, sizeof(error)) == GPU_SUITE_OK;
            if (ok)
              elapsed += gpu_suite_clock_elapsed(&ts, &te);
            if (th)
              cublasDestroy(th);
            if (tc)
              cudaFree(tc);
            if (tb)
              cudaFree(tb);
            if (ta)
              cudaFree(ta);
          }
        }
        if (ok && o.scope == GPU_SUITE_SCOPE_COMPUTE)
          ok = cuda_ok(cudaMemcpy(c.data(), dc, cb, cudaMemcpyDeviceToHost),
                       "copy result");
        ok = ok &&
             gpu_suite_utc_timestamp(et, error, sizeof(error)) == GPU_SUITE_OK;
        if (ok) {
          r.measurement_start_timestamp = st;
          r.measurement_end_timestamp = et;
          r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
          r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
          bool pass = verify(r, o, c, k);
          r.attempted = true;
          r.failure_origin = pass ? nullptr : "verification";
          r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          r.status = pass ? "success" : "failure";
          r.message = pass ? "" : "DGEMM verification failed";
          failed |= !pass;
        } else {
          r.attempted = true;
          r.failure_origin = "benchmark";
          r.verification_status = "skipped";
          r.exit_code = gpu_suite_optional_int_value(1);
          r.status = "failure";
          r.message = "cuBLAS pipeline failed";
          failed = true;
        }
        w.write(r);
        gpu_suite_result_destroy(&r);
        if (failed) {
          gpu_suite::emit_unmeasured(w, o, trial + 1, false, "prior-failure",
                                     "prior cuBLAS failure");
          break;
        }
      }
    }
    if (h)
      cublasDestroy(h);
    if (dc)
      cudaFree(dc);
    if (db)
      cudaFree(db);
    if (da)
      cudaFree(da);
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation failed");
    failed = true;
  }
  if (!w.close())
    failed = true;
  return failed ? 1 : 0;
}
