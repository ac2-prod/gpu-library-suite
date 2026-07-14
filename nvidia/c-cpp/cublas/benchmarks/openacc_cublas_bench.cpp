#include "gpu_suite/benchmark.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool set_verification(gpu_suite_result &result,
                             const gpu_suite_options &options,
                             const std::vector<double> &values, int inner_size,
                             int updates) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return true;
  }
  double expected = 1.0;
  for (int update = 0; update < updates; ++update)
    expected = options.alpha * inner_size + options.beta * expected;
  double maximum = 0.0;
  for (double value : values)
    maximum = std::max(maximum, std::fabs(value - expected));
  if (!std::isfinite(maximum)) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "nonfinite";
    return false;
  }
  gpu_suite::json_add_double(result.verification_metrics, "max_abs_error",
                             maximum);
  gpu_suite_json_value *threshold = gpu_suite_json_object();
  gpu_suite::json_add_string(threshold, "method", "absolute-plus-relative");
  gpu_suite::json_add_double(threshold, "reference_scale", std::fabs(expected));
  gpu_suite::json_add_double(threshold, "abs_tolerance", options.abs_tolerance);
  gpu_suite::json_add_double(threshold, "rel_tolerance", options.rel_tolerance);
  gpu_suite_json_object_set(result.verification_thresholds, "max_abs_error",
                            threshold);
  const bool pass = maximum <= options.abs_tolerance +
                                   options.rel_tolerance * std::fabs(expected);
  result.verification_primary_metric = "max_abs_error";
  result.verification_status = pass ? "pass" : "failure";
  return pass;
}

// OpenACC owns all A/B/C device allocation and movement in this benchmark.
int main(int argc, char **argv) {
  gpu_suite_options o;
  char error[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
  auto parsed = gpu_suite_options_parse(&o, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUBLAS);
    return 0;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    return 1;
  }
  int m = (int)(o.size_set ? o.size : o.m),
      n = (int)(o.size_set ? o.size : o.n),
      k = (int)(o.size_set ? o.size : o.k);
  std::vector<double> a((size_t)m * k, 1), b((size_t)k * n, 1),
      c((size_t)m * n, 1);
  double *ap = a.data(), *bp = b.data(), *cp = c.data();
  size_t ac = a.size(), bc = b.size(), cc = c.size();
  gpu_suite::ResultWriter writer;
  if (!writer.open(o))
    return 1;
  bool failed = false;
  if (cudaSetDevice(o.device) != cudaSuccess) {
    gpu_suite::emit_unmeasured(writer, o, 0, true, "benchmark",
                               "cudaSetDevice failed");
    failed = true;
  } else if (o.scope == GPU_SUITE_SCOPE_COMPUTE) {
    cublasHandle_t h = nullptr;
    failed = cublasCreate(&h) != CUBLAS_STATUS_SUCCESS;
    int next_trial = 0;
#pragma acc data copyin(ap[0 : ac], bp[0 : bc]) copy(cp[0 : cc])
    {
      for (int warm = 0; warm < o.warmup && !failed; ++warm) {
#pragma acc host_data use_device(ap, bp, cp)
        {
          failed =
              cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &o.alpha, ap, m,
                          bp, k, &o.beta, cp, m) != CUBLAS_STATUS_SUCCESS;
        }
      }
      if (!failed)
        failed = cudaDeviceSynchronize() != cudaSuccess;
      for (int trial = 0; trial < o.trials && !failed; ++trial) {
        std::fill(c.begin(), c.end(), 1.0);
#pragma acc update device(cp[0 : cc])
        gpu_suite_result r;
        gpu_suite::initialize_result(r, o, trial);
        char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
             et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec ts, te;
        gpu_suite_utc_timestamp(st, error, sizeof(error));
        cudaDeviceSynchronize();
        gpu_suite_clock_now(&ts, error, sizeof(error));
        bool ok = true;
#pragma acc host_data use_device(ap, bp, cp)
        {
          for (int rep = 0; rep < o.repeat; ++rep)
            ok = ok &&
                 cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &o.alpha, ap,
                             m, bp, k, &o.beta, cp, m) == CUBLAS_STATUS_SUCCESS;
        }
        ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
             gpu_suite_clock_now(&te, error, sizeof(error)) == GPU_SUITE_OK;
        gpu_suite_utc_timestamp(et, error, sizeof(error));
#pragma acc update self(cp[0 : cc])
        double elapsed = gpu_suite_clock_elapsed(&ts, &te);
        r.measurement_start_timestamp = st;
        r.measurement_end_timestamp = et;
        r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
        bool pass = ok && set_verification(r, o, c, k, o.repeat);
        r.attempted = true;
        r.failure_origin = pass ? nullptr : (ok ? "verification" : "benchmark");
        r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        r.status = pass ? "success" : "failure";
        r.message = pass ? "" : "OpenACC DGEMM failed";
        writer.write(r);
        gpu_suite_result_destroy(&r);
        next_trial = trial + 1;
        failed |= !pass;
        if (failed) {
          gpu_suite::emit_unmeasured(writer, o, next_trial, false,
                                     "prior-failure",
                                     "prior OpenACC cuBLAS failure");
          break;
        }
      }
    }
    if (failed && next_trial == 0) {
      gpu_suite::emit_unmeasured(writer, o, 0, true, "benchmark",
                                 "OpenACC cuBLAS setup/warmup failed");
    }
    cublasDestroy(h);
  } else {
    for (int warmup = 0; warmup < o.warmup && !failed; ++warmup) {
      std::fill(c.begin(), c.end(), 1.0);
      cublasHandle_t handle = nullptr;
      failed = cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS;
#pragma acc data copyin(ap[0 : ac], bp[0 : bc]) copy(cp[0 : cc])
      {
#pragma acc host_data use_device(ap, bp, cp)
        {
          if (!failed) {
            failed = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &o.alpha, ap, m, bp, k, &o.beta, cp,
                                 m) != CUBLAS_STATUS_SUCCESS;
          }
        }
        if (!failed)
          failed = cudaDeviceSynchronize() != cudaSuccess;
      }
      if (handle)
        cublasDestroy(handle);
    }
    if (failed) {
      gpu_suite::emit_unmeasured(writer, o, 0, true, "benchmark",
                                 "OpenACC cuBLAS warmup failed");
    }
    for (int trial = 0; trial < o.trials && !failed; ++trial) {
      gpu_suite_result r;
      gpu_suite::initialize_result(r, o, trial);
      char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
           et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      gpu_suite_utc_timestamp(st, error, sizeof(error));
      double elapsed = 0;
      bool ok = true;
      for (int rep = 0; rep < o.repeat && ok; ++rep) {
        std::fill(c.begin(), c.end(), 1.0);
        struct timespec ts, te;
        gpu_suite_clock_now(&ts, error, sizeof(error));
        cublasHandle_t h = nullptr;
        ok = cublasCreate(&h) == CUBLAS_STATUS_SUCCESS;
#pragma acc data copyin(ap[0 : ac], bp[0 : bc]) copy(cp[0 : cc])
        {
#pragma acc host_data use_device(ap, bp, cp)
          {
            if (ok)
              ok = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &o.alpha,
                               ap, m, bp, k, &o.beta, cp,
                               m) == CUBLAS_STATUS_SUCCESS;
          }
          ok = ok && cudaDeviceSynchronize() == cudaSuccess;
        }
        gpu_suite_clock_now(&te, error, sizeof(error));
        elapsed += gpu_suite_clock_elapsed(&ts, &te);
        if (h)
          cublasDestroy(h);
      }
      gpu_suite_utc_timestamp(et, error, sizeof(error));
      r.measurement_start_timestamp = st;
      r.measurement_end_timestamp = et;
      r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
      bool pass = ok && set_verification(r, o, c, k, 1);
      r.attempted = true;
      r.failure_origin = pass ? nullptr : (ok ? "verification" : "benchmark");
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? "" : "OpenACC end-to-end DGEMM failed";
      writer.write(r);
      gpu_suite_result_destroy(&r);
      failed |= !pass;
      if (failed) {
        gpu_suite::emit_unmeasured(writer, o, trial + 1, false, "prior-failure",
                                   "prior OpenACC cuBLAS failure");
      }
    }
  }
  if (!writer.close())
    failed = true;
  return failed ? 1 : 0;
}
