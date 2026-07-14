#include <cstdint>
#include "gpu_suite/benchmark.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
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
static gpu_suite::VerificationOutcome
verify(gpu_suite_result &r, const gpu_suite_options &o,
       const std::vector<double> &c, int k) {
  if (gpu_suite_verification_reset(&r) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!o.verify) {
    r.verification_primary_metric = nullptr;
    r.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }
  int updates = o.scope == GPU_SUITE_SCOPE_COMPUTE ? o.repeat : 1;
  double expected = 1;
  for (int i = 0; i < updates; ++i)
    expected = o.alpha * k + o.beta * expected;
  const double reference_scale =
      std::isfinite(expected) ? std::fabs(expected) : 0.0;
  if (gpu_suite_verification_add_absolute_relative_threshold(
          &r, "max_abs_error", reference_scale, o.abs_tolerance,
          o.rel_tolerance) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  double e = 0;
  bool finite = std::isfinite(expected);
  for (double x : c) {
    double error = 0.0;
    if (!gpu_suite_finite_absolute_error(x, expected, &error) ||
        !gpu_suite_finite_max_update(error, &e))
      finite = false;
  }
  r.verification_primary_metric = "max_abs_error";
  if (!finite) {
    if (gpu_suite_json_add_null(r.verification_metrics, "max_abs_error") !=
        GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    r.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }
  if (gpu_suite::json_add_double(r.verification_metrics, "max_abs_error", e) !=
      GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  const bool pass =
      e <= o.abs_tolerance + o.rel_tolerance * reference_scale;
  r.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

static bool run_end_to_end_pipeline(
    const gpu_suite_options &options, const std::vector<double> &a,
    const std::vector<double> &b, std::vector<double> &c, int m, int n, int k,
    std::size_t a_bytes, std::size_t b_bytes, std::size_t c_bytes,
    int repeat_index, bool measure,
    char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY], char *error,
    std::size_t error_size, double *elapsed_total) {
  double *device_a = nullptr;
  double *device_b = nullptr;
  double *device_c = nullptr;
  cublasHandle_t handle = nullptr;
  struct timespec start;
  struct timespec end;
  fill(c, 1.0);
  bool ok = !measure ||
            (repeat_index == 0
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK;
  ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_a), a_bytes),
                     "cudaMalloc A") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_b), b_bytes),
               "cudaMalloc B") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_c), c_bytes),
               "cudaMalloc C") &&
       cuda_ok(cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice),
               "copy A") &&
       cuda_ok(cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice),
               "copy B") &&
       cuda_ok(cudaMemcpy(device_c, c.data(), c_bytes, cudaMemcpyHostToDevice),
               "copy C") &&
       blas_ok(cublasCreate(&handle), "cublasCreate") &&
       blas_ok(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &options.alpha, device_a, m, device_b, k,
                           &options.beta, device_c, m),
               "cublasDgemm") &&
       cuda_ok(cudaDeviceSynchronize(), "end-to-end synchronize") &&
       cuda_ok(cudaMemcpy(c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost),
               "copy C to host");
  if (ok && measure) {
    ok = (repeat_index + 1 == options.repeat
              ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                          error_size)
              : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
    if (ok)
      *elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  if (handle != nullptr)
    ok = blas_ok(cublasDestroy(handle), "cublasDestroy") && ok;
  if (device_c != nullptr)
    ok = cuda_ok(cudaFree(device_c), "cudaFree C") && ok;
  if (device_b != nullptr)
    ok = cuda_ok(cudaFree(device_b), "cudaFree B") && ok;
  if (device_a != nullptr)
    ok = cuda_ok(cudaFree(device_a), "cudaFree A") && ok;
  return ok;
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
  gpu_suite::ResultWriter w;
  if (!w.open(o))
    return 1;
  const std::uint64_t m_value = o.size_set ? o.size : o.m;
  const std::uint64_t n_value = o.size_set ? o.size : o.n;
  const std::uint64_t k_value = o.size_set ? o.size : o.k;
  int m = 0;
  int n = 0;
  int k = 0;
  std::size_t a_count = 0;
  std::size_t b_count = 0;
  std::size_t c_count = 0;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
  if (!gpu_suite_checked_u64_to_int(m_value, &m) ||
      !gpu_suite_checked_u64_to_int(n_value, &n) ||
      !gpu_suite_checked_u64_to_int(k_value, &k) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(m),
                                  static_cast<std::size_t>(k), &a_count) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(k),
                                  static_cast<std::size_t>(n), &b_count) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(m),
                                  static_cast<std::size_t>(n), &c_count) ||
      !gpu_suite_checked_bytes(a_count, sizeof(double), &a_bytes) ||
      !gpu_suite_checked_bytes(b_count, sizeof(double), &b_bytes) ||
      !gpu_suite_checked_bytes(c_count, sizeof(double), &c_bytes)) {
    std::fprintf(stderr,
                 "DGEMM dimensions or byte counts exceed supported range\n");
    gpu_suite::emit_unmeasured(
        w, o, 0, false, "prerequisite",
        "DGEMM dimensions exceed supported integer or byte range");
    (void)w.close();
    return 1;
  }
  bool any_failure = false;
  bool fatal_failure = false;
  try {
    std::vector<double> a(a_count, 1.0);
    std::vector<double> b(b_count, 1.0);
    std::vector<double> c(c_count, 1.0);
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cublasHandle_t h = nullptr;
    if (!cuda_ok(cudaSetDevice(o.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "cuBLAS setup failed");
      any_failure = true;
      fatal_failure = true;
    } else if (o.scope == GPU_SUITE_SCOPE_COMPUTE &&
               (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&da), a_bytes),
                         "cudaMalloc A") ||
                !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&db), b_bytes),
                         "cudaMalloc B") ||
                !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&dc), c_bytes),
                         "cudaMalloc C") ||
                !cuda_ok(cudaMemcpy(da, a.data(), a_bytes,
                                    cudaMemcpyHostToDevice),
                         "copy A") ||
                !cuda_ok(cudaMemcpy(db, b.data(), b_bytes,
                                    cudaMemcpyHostToDevice),
                         "copy B") ||
                !blas_ok(cublasCreate(&h), "cublasCreate"))) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "cuBLAS compute setup failed");
      any_failure = true;
      fatal_failure = true;
    } else if (o.scope == GPU_SUITE_SCOPE_COMPUTE) {
      for (int warm = 0; warm < o.warmup; ++warm) {
        if (!cuda_ok(cudaMemcpy(dc, c.data(), c_bytes, cudaMemcpyHostToDevice),
                     "warmup restore C") ||
            !blas_ok(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &o.alpha, da, m, db, k, &o.beta, dc, m),
                     "warmup cublasDgemm")) {
          fatal_failure = true;
          break;
        }
      }
      if (!fatal_failure &&
          !cuda_ok(cudaDeviceSynchronize(), "warmup synchronize"))
        fatal_failure = true;
      if (fatal_failure) {
        gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                   "cuBLAS warmup failed");
        any_failure = true;
      }
    } else {
      for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm) {
        fatal_failure = !run_end_to_end_pipeline(
            o, a, b, c, m, n, k, a_bytes, b_bytes, c_bytes, 0, false,
            nullptr, nullptr, error, sizeof(error), nullptr);
      }
      if (fatal_failure) {
        gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                   "cuBLAS end-to-end warmup failed");
        any_failure = true;
      }
    }
    if (!fatal_failure) {
      for (int trial = 0; trial < o.trials && !fatal_failure; ++trial) {
        gpu_suite_result r;
        if (!gpu_suite::initialize_result(r, o, trial)) {
          fatal_failure = true;
          any_failure = true;
          break;
        }
        fill(c, 1);
        char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
             et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec ts, te;
        double elapsed = 0;
        bool ok = true;
        if (o.scope == GPU_SUITE_SCOPE_COMPUTE) {
          ok = cuda_ok(cudaMemcpy(dc, c.data(), c_bytes,
                                  cudaMemcpyHostToDevice),
                       "restore C") &&
               cuda_ok(cudaDeviceSynchronize(), "sync before") &&
               gpu_suite_measurement_start(st, &ts, error, sizeof(error)) ==
                   GPU_SUITE_OK;
          for (int rep = 0; rep < o.repeat && ok; ++rep)
            ok = blas_ok(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &o.alpha, da, m, db, k, &o.beta, dc, m),
                         "cublasDgemm");
          ok = ok && cuda_ok(cudaDeviceSynchronize(), "sync after") &&
               gpu_suite_measurement_end(&te, et, error, sizeof(error)) ==
                   GPU_SUITE_OK;
          if (ok)
            elapsed = gpu_suite_clock_elapsed(&ts, &te);
        } else {
          for (int rep = 0; rep < o.repeat && ok; ++rep) {
            ok = run_end_to_end_pipeline(
                o, a, b, c, m, n, k, a_bytes, b_bytes, c_bytes, rep, true,
                st, et, error, sizeof(error), &elapsed);
          }
        }
        if (ok && o.scope == GPU_SUITE_SCOPE_COMPUTE)
          ok = cuda_ok(cudaMemcpy(c.data(), dc, c_bytes,
                                  cudaMemcpyDeviceToHost),
                       "copy result");
        if (ok) {
          r.measurement_start_timestamp = st;
          r.measurement_end_timestamp = et;
          r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
          r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
          const gpu_suite::VerificationOutcome outcome = verify(r, o, c, k);
          const bool constructed =
              outcome != gpu_suite::VerificationOutcome::construction_error;
          const bool pass = outcome == gpu_suite::VerificationOutcome::pass;
          r.attempted = true;
          r.failure_origin =
              pass ? nullptr : (constructed ? "verification" : "benchmark");
          if (!constructed)
            r.verification_status = "skipped";
          r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          r.status = pass ? "success" : "failure";
          r.message =
              pass ? ""
                   : (constructed ? "DGEMM verification failed"
                                  : "verification result construction failed");
          any_failure = any_failure || !pass;
          fatal_failure = fatal_failure || !constructed;
        } else {
          r.attempted = true;
          r.failure_origin = "benchmark";
          r.verification_status = "skipped";
          r.exit_code = gpu_suite_optional_int_value(1);
          r.status = "failure";
          r.message = "cuBLAS pipeline failed";
          any_failure = true;
          fatal_failure = true;
        }
        if (!w.write(r)) {
          fatal_failure = true;
          any_failure = true;
        }
        gpu_suite_result_destroy(&r);
        if (fatal_failure) {
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
  } catch (const std::length_error &) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host vector size exceeds max_size");
    any_failure = true;
    fatal_failure = true;
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
    fatal_failure = true;
  }
  if (!w.close())
    any_failure = true;
  return any_failure ? 1 : 0;
}
