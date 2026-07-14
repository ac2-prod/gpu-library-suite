#include <cstdint>
#include "gpu_suite/benchmark.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

static gpu_suite::VerificationOutcome
set_verification(gpu_suite_result &result, const gpu_suite_options &options,
                 const std::vector<double> &values, int inner_size,
                 int updates) {
  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }
  double expected = 1.0;
  for (int update = 0; update < updates; ++update)
    expected = options.alpha * inner_size + options.beta * expected;
  const double reference_scale =
      std::isfinite(expected) ? std::fabs(expected) : 0.0;
  if (gpu_suite_verification_add_absolute_relative_threshold(
          &result, "max_abs_error", reference_scale, options.abs_tolerance,
          options.rel_tolerance) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  double maximum = 0.0;
  bool finite = std::isfinite(expected);
  for (double value : values) {
    double error = 0.0;
    if (!gpu_suite_finite_absolute_error(value, expected, &error) ||
        !gpu_suite_finite_max_update(error, &maximum))
      finite = false;
  }
  result.verification_primary_metric = "max_abs_error";
  if (!finite) {
    if (gpu_suite_json_add_null(result.verification_metrics,
                                "max_abs_error") != GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    result.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }
  if (gpu_suite::json_add_double(result.verification_metrics, "max_abs_error",
                                 maximum) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  const bool pass = maximum <= options.abs_tolerance +
                                   options.rel_tolerance * reference_scale;
  result.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

static bool cuda_success(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}

static bool cublas_success(cublasStatus_t status, const char *api) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuBLAS status %d\n", api,
               static_cast<int>(status));
  return false;
}

// OpenACC owns all A/B/C device allocation and movement in this benchmark.
int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
  const auto parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUBLAS);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }

  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;
  int m = 0;
  int n = 0;
  int k = 0;
  std::size_t a_count = 0;
  std::size_t b_count = 0;
  std::size_t c_count = 0;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
  const std::uint64_t m_value = options.size_set ? options.size : options.m;
  const std::uint64_t n_value = options.size_set ? options.size : options.n;
  const std::uint64_t k_value = options.size_set ? options.size : options.k;
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
    (void)a_bytes;
    (void)b_bytes;
    (void)c_bytes;
    std::fprintf(stderr,
                 "DGEMM dimensions or byte counts exceed supported range\n");
    gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "DGEMM dimensions exceed supported integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
  try {
  std::vector<double> a(a_count, 1.0);
  std::vector<double> b(b_count, 1.0);
  std::vector<double> c(c_count, 1.0);
  double *a_ptr = a.data();
  double *b_ptr = b.data();
  double *c_ptr = c.data();
  bool emitted_result_rows = false;
  bool fatal_failure =
      !cuda_success(cudaSetDevice(options.device), "cudaSetDevice");

  if (!fatal_failure && options.scope == GPU_SUITE_SCOPE_COMPUTE) {
    cublasHandle_t handle = nullptr;
    fatal_failure =
        !cublas_success(cublasCreate(&handle), "cublasCreate");
#pragma acc data copyin(a_ptr[0 : a_count], b_ptr[0 : b_count])                \
    copy(c_ptr[0 : c_count])
    {
      for (int warmup = 0; warmup < options.warmup && !fatal_failure;
           ++warmup) {
#pragma acc host_data use_device(a_ptr, b_ptr, c_ptr)
        {
          fatal_failure = !cublas_success(
              cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &options.alpha, a_ptr, m, b_ptr, k, &options.beta,
                           c_ptr, m),
              "warmup cublasDgemm");
        }
      }
      if (!fatal_failure)
        fatal_failure = !cuda_success(cudaDeviceSynchronize(),
                                      "warmup synchronize");

      for (int trial = 0; trial < options.trials && !fatal_failure; ++trial) {
        gpu_suite_result result;
        if (!gpu_suite::initialize_result(result, options, trial)) {
          fatal_failure = true;
          any_failure = true;
          break;
        }
        std::fill(c.begin(), c.end(), 1.0);
#pragma acc update device(c_ptr[0 : c_count])
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec start;
        struct timespec end;
        bool ok =
            cuda_success(cudaDeviceSynchronize(),
                         "OpenACC update device C") &&
            gpu_suite_measurement_start(start_timestamp, &start, error,
                                         sizeof(error)) == GPU_SUITE_OK;
#pragma acc host_data use_device(a_ptr, b_ptr, c_ptr)
        {
          for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
            ok = cublas_success(
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                             &options.alpha, a_ptr, m, b_ptr, k,
                             &options.beta, c_ptr, m),
                "cublasDgemm");
          }
        }
        ok = ok &&
             cuda_success(cudaDeviceSynchronize(), "post-timing synchronize") &&
             gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error)) == GPU_SUITE_OK;
#pragma acc update self(c_ptr[0 : c_count])
        ok = ok && cuda_success(cudaDeviceSynchronize(),
                                "OpenACC update self C");

        bool pass = false;
        bool constructed = true;
        if (ok) {
          const double elapsed = gpu_suite_clock_elapsed(&start, &end);
          result.measurement_start_timestamp = start_timestamp;
          result.measurement_end_timestamp = end_timestamp;
          result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
          result.elapsed_sec = gpu_suite_optional_double_value(
              elapsed / static_cast<double>(options.repeat));
          const gpu_suite::VerificationOutcome outcome =
              set_verification(result, options, c, k, options.repeat);
          constructed =
              outcome != gpu_suite::VerificationOutcome::construction_error;
          pass = outcome == gpu_suite::VerificationOutcome::pass;
          result.failure_origin =
              pass ? nullptr : (constructed ? "verification" : "benchmark");
          if (!constructed)
            result.verification_status = "skipped";
          any_failure = any_failure || !pass;
          fatal_failure = fatal_failure || !constructed;
        } else {
          result.verification_status = "skipped";
          result.failure_origin = "benchmark";
          fatal_failure = true;
          any_failure = true;
        }
        result.attempted = true;
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message =
            pass ? ""
                 : (constructed ? "OpenACC cuBLAS compute trial failed"
                                : "verification result construction failed");
        if (!writer.write(result)) {
          fatal_failure = true;
          any_failure = true;
        }
        emitted_result_rows = true;
        gpu_suite_result_destroy(&result);
        if (fatal_failure) {
          gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                     "prior-failure",
                                     "prior OpenACC cuBLAS failure");
        }
      }
    }
    if (handle != nullptr &&
        !cublas_success(cublasDestroy(handle), "cublasDestroy")) {
      fatal_failure = true;
    }
  } else if (!fatal_failure) {
    for (int warmup = 0; warmup < options.warmup && !fatal_failure; ++warmup) {
      std::fill(c.begin(), c.end(), 1.0);
      cublasHandle_t handle = nullptr;
      fatal_failure =
          !cublas_success(cublasCreate(&handle), "warmup cublasCreate");
#pragma acc data copyin(a_ptr[0 : a_count], b_ptr[0 : b_count])                \
    copy(c_ptr[0 : c_count])
      {
#pragma acc host_data use_device(a_ptr, b_ptr, c_ptr)
        {
          if (!fatal_failure)
            fatal_failure = !cublas_success(
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                             &options.alpha, a_ptr, m, b_ptr, k,
                             &options.beta, c_ptr, m),
                "warmup cublasDgemm");
        }
        if (!fatal_failure)
          fatal_failure = !cuda_success(cudaDeviceSynchronize(),
                                        "warmup synchronize");
      }
      if (handle != nullptr)
        fatal_failure =
            !cublas_success(cublasDestroy(handle), "warmup cublasDestroy") ||
            fatal_failure;
    }

    for (int trial = 0; trial < options.trials && !fatal_failure; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial)) {
        fatal_failure = true;
        any_failure = true;
        break;
      }
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      double elapsed_total = 0.0;
      bool ok = true;
      for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
        std::fill(c.begin(), c.end(), 1.0);
        struct timespec start;
        struct timespec end;
        ok = (repeat == 0
                  ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                                sizeof(error))
                  : gpu_suite_clock_now(&start, error, sizeof(error))) ==
             GPU_SUITE_OK;
        cublasHandle_t handle = nullptr;
        if (ok)
          ok = cublas_success(cublasCreate(&handle), "cublasCreate");
#pragma acc data copyin(a_ptr[0 : a_count], b_ptr[0 : b_count])                \
    copy(c_ptr[0 : c_count])
        {
#pragma acc host_data use_device(a_ptr, b_ptr, c_ptr)
          {
            if (ok)
              ok = cublas_success(
                  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                               &options.alpha, a_ptr, m, b_ptr, k,
                               &options.beta, c_ptr, m),
                  "cublasDgemm");
          }
          if (ok)
            ok = cuda_success(cudaDeviceSynchronize(),
                              "end-to-end synchronize");
        }
        if (ok)
          ok = (repeat + 1 == options.repeat
                    ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                                sizeof(error))
                    : gpu_suite_clock_now(&end, error, sizeof(error))) ==
               GPU_SUITE_OK;
        if (ok)
          elapsed_total += gpu_suite_clock_elapsed(&start, &end);
        if (handle != nullptr)
          ok = cublas_success(cublasDestroy(handle), "cublasDestroy") && ok;
      }

      bool pass = false;
      bool constructed = true;
      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec = gpu_suite_optional_double_value(
            elapsed_total / static_cast<double>(options.repeat));
        const gpu_suite::VerificationOutcome outcome =
            set_verification(result, options, c, k, 1);
        constructed =
            outcome != gpu_suite::VerificationOutcome::construction_error;
        pass = outcome == gpu_suite::VerificationOutcome::pass;
        result.failure_origin =
            pass ? nullptr : (constructed ? "verification" : "benchmark");
        if (!constructed)
          result.verification_status = "skipped";
        any_failure = any_failure || !pass;
        fatal_failure = fatal_failure || !constructed;
      } else {
        result.verification_status = "skipped";
        result.failure_origin = "benchmark";
        fatal_failure = true;
        any_failure = true;
      }
      result.attempted = true;
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.status = pass ? "success" : "failure";
      result.message =
          pass ? ""
               : (constructed ? "OpenACC cuBLAS end-to-end trial failed"
                              : "verification result construction failed");
      if (!writer.write(result)) {
        fatal_failure = true;
        any_failure = true;
      }
      emitted_result_rows = true;
      gpu_suite_result_destroy(&result);
      if (fatal_failure)
        gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                   "prior-failure",
                                   "prior OpenACC cuBLAS failure");
    }
  }

  if (fatal_failure) {
    any_failure = true;
    if (!emitted_result_rows && options.trials > 0)
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "OpenACC cuBLAS setup/warmup failed");
  }
  } catch (const std::length_error &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host vector size exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  }
  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
