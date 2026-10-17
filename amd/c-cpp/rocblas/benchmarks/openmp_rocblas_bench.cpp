#include <cstdint>
#include "gpu_suite/benchmark.hpp"

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

static bool mapped_device(const void *pointer) {
  hipPointerAttribute_t attributes{};
  int device = -1;
  return hipGetDevice(&device) == hipSuccess &&
         hipPointerGetAttributes(&attributes, pointer) == hipSuccess &&
         attributes.type == hipMemoryTypeDevice && attributes.device == device;
}

static bool prove_offload() {
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices()) {
    std::fprintf(stderr, "OpenMP target device unavailable\n");
    return false;
  }
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) std::fprintf(stderr, "OpenMP host fallback rejected\n");
  return initial == 0;
}

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

static bool hip_success(hipError_t status, const char *api) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, hipGetErrorString(status));
  return false;
}

static bool rocblas_success(rocblas_status status, const char *api) {
  if (status == rocblas_status_success)
    return true;
  std::fprintf(stderr, "%s: rocBLAS status %d\n", api,
               static_cast<int>(status));
  return false;
}

// OpenMP owns all A/B/C device allocation and movement in this benchmark.
int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCBLAS,
                         GPU_SUITE_IMPLEMENTATION_OPENMP);
  const auto parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCBLAS);
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
      !hip_success(hipSetDevice(options.device), "hipSetDevice") || !prove_offload();

  if (!fatal_failure && options.scope == GPU_SUITE_SCOPE_COMPUTE) {
    rocblas_handle handle = nullptr;
    fatal_failure =
        !rocblas_success(rocblas_create_handle(&handle), "rocblas_create_handle");
#pragma omp target data map(to: a_ptr[0 : a_count], \
    b_ptr[0 : b_count]) map(tofrom: c_ptr[0 : c_count])
    {
      for (int warmup = 0; warmup < options.warmup && !fatal_failure;
           ++warmup) {
#pragma omp target data use_device_ptr(a_ptr, b_ptr, c_ptr)
        {
      fatal_failure = fatal_failure || !(mapped_device(a_ptr) && mapped_device(b_ptr) && mapped_device(c_ptr));
          fatal_failure = fatal_failure || !rocblas_success(
              rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                           &options.alpha, a_ptr, m, b_ptr, k, &options.beta,
                           c_ptr, m),
              "warmup rocblas_dgemm");
        }
      }
      if (!fatal_failure)
        fatal_failure = !hip_success(hipDeviceSynchronize(),
                                      "warmup synchronize");

      for (int trial = 0; trial < options.trials && !fatal_failure; ++trial) {
        gpu_suite_result result;
        if (!gpu_suite::initialize_result(result, options, trial)) {
          fatal_failure = true;
          any_failure = true;
          break;
        }
        std::fill(c.begin(), c.end(), 1.0);
#pragma omp target update to(c_ptr[0 : c_count])
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec start;
        struct timespec end;
        bool ok =
            hip_success(hipDeviceSynchronize(), "OpenMP update device C");
#pragma omp target data use_device_ptr(a_ptr, b_ptr, c_ptr)
        {
      ok = ok && (mapped_device(a_ptr) && mapped_device(b_ptr) && mapped_device(c_ptr));
          ok = ok && gpu_suite_measurement_start(start_timestamp, &start, error, sizeof(error)) == GPU_SUITE_OK;
          for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
            ok = rocblas_success(
                rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                             &options.alpha, a_ptr, m, b_ptr, k,
                             &options.beta, c_ptr, m),
                "rocblas_dgemm");
          }
        }
        ok = ok &&
             hip_success(hipDeviceSynchronize(), "post-timing synchronize") &&
             gpu_suite_measurement_end(&end, end_timestamp, error,
                                        sizeof(error)) == GPU_SUITE_OK;
#pragma omp target update from(c_ptr[0 : c_count])
        ok = ok && hip_success(hipDeviceSynchronize(),
                                "OpenMP update self C");

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
                 : (constructed ? "OpenMP rocBLAS compute trial failed"
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
                                     "prior OpenMP rocBLAS failure");
        }
      }
    }
    if (handle != nullptr &&
        !rocblas_success(rocblas_destroy_handle(handle), "rocblas_destroy_handle")) {
      fatal_failure = true;
    }
  } else if (!fatal_failure) {
    for (int warmup = 0; warmup < options.warmup && !fatal_failure; ++warmup) {
      std::fill(c.begin(), c.end(), 1.0);
      rocblas_handle handle = nullptr;
      fatal_failure =
          !rocblas_success(rocblas_create_handle(&handle), "warmup rocblas_create_handle");
#pragma omp target data map(to: a_ptr[0 : a_count], \
    b_ptr[0 : b_count]) map(tofrom: c_ptr[0 : c_count])
      {
#pragma omp target data use_device_ptr(a_ptr, b_ptr, c_ptr)
        {
      fatal_failure = fatal_failure || !(mapped_device(a_ptr) && mapped_device(b_ptr) && mapped_device(c_ptr));
          if (!fatal_failure)
            fatal_failure = !rocblas_success(
                rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                             &options.alpha, a_ptr, m, b_ptr, k,
                             &options.beta, c_ptr, m),
                "warmup rocblas_dgemm");
        }
        if (!fatal_failure)
          fatal_failure = !hip_success(hipDeviceSynchronize(),
                                        "warmup synchronize");
      }
      if (handle != nullptr)
        fatal_failure =
            !rocblas_success(rocblas_destroy_handle(handle), "warmup rocblas_destroy_handle") ||
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
        rocblas_handle handle = nullptr;
        if (ok)
          ok = rocblas_success(rocblas_create_handle(&handle), "rocblas_create_handle");
#pragma omp target data map(to: a_ptr[0 : a_count], \
    b_ptr[0 : b_count]) map(tofrom: c_ptr[0 : c_count])
        {
#pragma omp target data use_device_ptr(a_ptr, b_ptr, c_ptr)
          {
      ok = ok && (mapped_device(a_ptr) && mapped_device(b_ptr) && mapped_device(c_ptr));
            if (ok)
              ok = rocblas_success(
                  rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                               &options.alpha, a_ptr, m, b_ptr, k,
                               &options.beta, c_ptr, m),
                  "rocblas_dgemm");
          }
          if (ok)
            ok = hip_success(hipDeviceSynchronize(),
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
          ok = rocblas_success(rocblas_destroy_handle(handle), "rocblas_destroy_handle") && ok;
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
               : (constructed ? "OpenMP rocBLAS end-to-end trial failed"
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
                                   "prior OpenMP rocBLAS failure");
    }
  }

  if (fatal_failure) {
    any_failure = true;
    if (!emitted_result_rows && options.trials > 0)
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "OpenMP rocBLAS setup/warmup failed");
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
