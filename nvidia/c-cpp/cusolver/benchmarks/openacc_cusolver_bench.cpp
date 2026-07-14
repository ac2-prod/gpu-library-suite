#include "gpu_suite/benchmark.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

void make_dense_system(int n, int nrhs, double *matrix, double *rhs) {
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      matrix[row + static_cast<std::size_t>(column) * n] =
          row == column ? n + 1.0 : 1.0;
    }
  }
  std::fill(rhs, rhs + static_cast<std::size_t>(n) * nrhs, 2.0 * n);
}

bool add_threshold(gpu_suite_json_value *thresholds, const char *name,
                   const gpu_suite_options &options) {
  gpu_suite_json_value *threshold = gpu_suite_json_object();
  return threshold != nullptr &&
         gpu_suite::json_add_string(threshold, "method",
                                    "absolute-plus-relative") == GPU_SUITE_OK &&
         gpu_suite::json_add_double(threshold, "reference_scale", 1.0) ==
             GPU_SUITE_OK &&
         gpu_suite::json_add_double(threshold, "abs_tolerance",
                                    options.abs_tolerance) == GPU_SUITE_OK &&
         gpu_suite::json_add_double(threshold, "rel_tolerance",
                                    options.rel_tolerance) == GPU_SUITE_OK &&
         gpu_suite_json_object_set(thresholds, name, threshold) == GPU_SUITE_OK;
}

bool verify(gpu_suite_result &result, const gpu_suite_options &options,
            const std::vector<double> &solution,
            const std::vector<double> &matrix, const std::vector<double> &rhs,
            int n, int nrhs) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return true;
  }
  double solution_error = 0.0;
  double residual = 0.0;
  double matrix_norm = 0.0;
  double rhs_norm = 0.0;
  for (int row = 0; row < n; ++row) {
    double row_sum = 0.0;
    for (int column = 0; column < n; ++column) {
      row_sum += std::fabs(matrix[row + static_cast<std::size_t>(column) * n]);
    }
    matrix_norm = std::max(matrix_norm, row_sum);
  }
  for (int column = 0; column < nrhs; ++column) {
    for (int row = 0; row < n; ++row) {
      const std::size_t index = row + static_cast<std::size_t>(column) * n;
      solution_error =
          std::max(solution_error, std::fabs(solution[index] - 1.0));
      rhs_norm = std::max(rhs_norm, std::fabs(rhs[index]));
      double product = 0.0;
      for (int inner = 0; inner < n; ++inner) {
        product += matrix[row + static_cast<std::size_t>(inner) * n] *
                   solution[inner + static_cast<std::size_t>(column) * n];
      }
      residual = std::max(residual, std::fabs(product - rhs[index]));
    }
  }
  const double relative_residual = residual / (matrix_norm + rhs_norm);
  gpu_suite::json_add_double(result.verification_metrics,
                             "solution_relative_error", solution_error);
  gpu_suite::json_add_double(result.verification_metrics, "relative_residual",
                             relative_residual);
  add_threshold(result.verification_thresholds, "solution_relative_error",
                options);
  add_threshold(result.verification_thresholds, "relative_residual", options);
  result.verification_primary_metric = "relative_residual";
  const double bound = options.abs_tolerance + options.rel_tolerance;
  const bool pass = std::isfinite(solution_error) &&
                    std::isfinite(relative_residual) &&
                    solution_error <= bound && relative_residual <= bound;
  result.verification_status = pass ? "pass" : "failure";
  return pass;
}

bool run_library_calls(cusolverDnHandle_t handle, int n, int nrhs,
                       double *matrix, double *rhs, double *workspace,
                       int *pivots, int *getrf_info, int *getrs_info) {
  bool ok = true;
#pragma acc host_data use_device(matrix, rhs, pivots, getrf_info, getrs_info)
  {
    ok = cusolverDnDgetrf(handle, n, n, matrix, n, workspace, pivots,
                          getrf_info) == CUSOLVER_STATUS_SUCCESS &&
         cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs, matrix, n, pivots, rhs,
                          n, getrs_info) == CUSOLVER_STATUS_SUCCESS;
  }
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSOLVER,
                         GPU_SUITE_IMPLEMENTATION_OPENACC);
  const gpu_suite_parse_result parsed =
      gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_CUSOLVER);
    return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }

  const int n = static_cast<int>(options.size);
  const int nrhs = static_cast<int>(options.nrhs);
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;

  try {
    std::vector<double> canonical_matrix(static_cast<std::size_t>(n) * n);
    std::vector<double> canonical_rhs(static_cast<std::size_t>(n) * nrhs);
    std::vector<double> matrix(canonical_matrix.size());
    std::vector<double> rhs(canonical_rhs.size());
    std::vector<int> pivots(n);
    std::vector<int> getrf_info(1);
    std::vector<int> getrs_info(1);
    make_dense_system(n, nrhs, canonical_matrix.data(), canonical_rhs.data());
    if (cudaSetDevice(options.device) != cudaSuccess) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      cusolverDnHandle_t handle = nullptr;
      bool setup_ok = cusolverDnCreate(&handle) == CUSOLVER_STATUS_SUCCESS;
      double *matrix_ptr = matrix.data();
      double *rhs_ptr = rhs.data();
      int *pivots_ptr = pivots.data();
      int *getrf_info_ptr = getrf_info.data();
      int *getrs_info_ptr = getrs_info.data();
      const std::size_t matrix_count = matrix.size();
      const std::size_t rhs_count = rhs.size();
      int workspace_count = 0;
      double *workspace = nullptr;

#pragma acc data copy(matrix_ptr[0 : matrix_count], rhs_ptr[0 : rhs_count])    \
    create(pivots_ptr[0 : n], getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
      {
#pragma acc host_data use_device(matrix_ptr)
        {
          if (setup_ok) {
            setup_ok = cusolverDnDgetrf_bufferSize(handle, n, n, matrix_ptr, n,
                                                   &workspace_count) ==
                       CUSOLVER_STATUS_SUCCESS;
          }
        }
        if (setup_ok) {
          setup_ok = cudaMalloc(reinterpret_cast<void **>(&workspace),
                                static_cast<std::size_t>(workspace_count) *
                                    sizeof(double)) == cudaSuccess;
        }
        for (int warmup = 0; warmup < options.warmup && setup_ok; ++warmup) {
          matrix = canonical_matrix;
          rhs = canonical_rhs;
          std::fill(pivots.begin(), pivots.end(), 0);
          getrf_info[0] = 0;
          getrs_info[0] = 0;
#pragma acc update device(matrix_ptr[0 : matrix_count],                        \
                          rhs_ptr[0 : rhs_count], pivots_ptr[0 : n],           \
                          getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
          setup_ok =
              run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr, workspace,
                                pivots_ptr, getrf_info_ptr, getrs_info_ptr) &&
              cudaDeviceSynchronize() == cudaSuccess;
        }

        for (int trial = 0; trial < options.trials && setup_ok; ++trial) {
          matrix = canonical_matrix;
          rhs = canonical_rhs;
          std::fill(pivots.begin(), pivots.end(), 0);
          getrf_info[0] = 0;
          getrs_info[0] = 0;
#pragma acc update device(matrix_ptr[0 : matrix_count],                        \
                          rhs_ptr[0 : rhs_count], pivots_ptr[0 : n],           \
                          getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
          gpu_suite_result result;
          gpu_suite::initialize_result(result, options, trial);
          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          struct timespec start;
          struct timespec end;
          bool ok =
              gpu_suite_utc_timestamp(start_timestamp, error, sizeof(error)) ==
                  GPU_SUITE_OK &&
              cudaDeviceSynchronize() == cudaSuccess &&
              gpu_suite_clock_now(&start, error, sizeof(error)) ==
                  GPU_SUITE_OK &&
              run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr, workspace,
                                pivots_ptr, getrf_info_ptr, getrs_info_ptr) &&
              cudaDeviceSynchronize() == cudaSuccess &&
              gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK;
#pragma acc update self(rhs_ptr[0 : rhs_count], getrf_info_ptr[0 : 1],         \
                        getrs_info_ptr[0 : 1])
          ok = ok && gpu_suite_utc_timestamp(end_timestamp, error,
                                             sizeof(error)) == GPU_SUITE_OK;
          if (ok) {
            const double elapsed = gpu_suite_clock_elapsed(&start, &end);
            result.measurement_start_timestamp = start_timestamp;
            result.measurement_end_timestamp = end_timestamp;
            result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
            result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
            result.getrf_info = gpu_suite_optional_int_value(getrf_info[0]);
            result.getrs_info = gpu_suite_optional_int_value(getrs_info[0]);
            const bool verified = verify(result, options, rhs, canonical_matrix,
                                         canonical_rhs, n, nrhs);
            const bool pass =
                getrf_info[0] == 0 && getrs_info[0] == 0 && verified;
            result.attempted = true;
            result.failure_origin = pass ? nullptr : "verification";
            result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
            result.status = pass ? "success" : "failure";
            result.message = pass ? "" : "OpenACC solve verification failed";
            any_failure = any_failure || !pass;
          } else {
            result.attempted = true;
            result.failure_origin = "benchmark";
            result.verification_status = "skipped";
            result.exit_code = gpu_suite_optional_int_value(1);
            result.status = "failure";
            result.message = "OpenACC cuSOLVER pipeline failed";
            any_failure = true;
          }
          writer.write(result);
          gpu_suite_result_destroy(&result);
          if (!ok) {
            gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                       "prior-failure",
                                       "prior OpenACC cuSOLVER failure");
            break;
          }
        }
        if (workspace != nullptr)
          cudaFree(workspace);
      }
      if (handle != nullptr)
        cusolverDnDestroy(handle);
      if (!setup_ok) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "OpenACC cuSOLVER setup/warmup failed");
        any_failure = true;
      }
    } else {
      for (int trial = 0; trial < options.trials; ++trial) {
        matrix = canonical_matrix;
        rhs = canonical_rhs;
        std::fill(pivots.begin(), pivots.end(), 0);
        getrf_info[0] = 0;
        getrs_info[0] = 0;
        double *matrix_ptr = matrix.data();
        double *rhs_ptr = rhs.data();
        int *pivots_ptr = pivots.data();
        int *getrf_info_ptr = getrf_info.data();
        int *getrs_info_ptr = getrs_info.data();
        const std::size_t matrix_count = matrix.size();
        const std::size_t rhs_count = rhs.size();
        cusolverDnHandle_t handle = nullptr;
        double *workspace = nullptr;
        int workspace_count = 0;
        gpu_suite_result result;
        gpu_suite::initialize_result(result, options, trial);
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec start;
        struct timespec end;
        bool ok =
            gpu_suite_utc_timestamp(start_timestamp, error, sizeof(error)) ==
                GPU_SUITE_OK &&
            gpu_suite_clock_now(&start, error, sizeof(error)) == GPU_SUITE_OK &&
            cusolverDnCreate(&handle) == CUSOLVER_STATUS_SUCCESS;
#pragma acc data copy(matrix_ptr[0 : matrix_count], rhs_ptr[0 : rhs_count])    \
    create(pivots_ptr[0 : n], getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
        {
#pragma acc host_data use_device(matrix_ptr)
          {
            if (ok) {
              ok = cusolverDnDgetrf_bufferSize(handle, n, n, matrix_ptr, n,
                                               &workspace_count) ==
                   CUSOLVER_STATUS_SUCCESS;
            }
          }
          if (ok) {
            ok = cudaMalloc(reinterpret_cast<void **>(&workspace),
                            static_cast<std::size_t>(workspace_count) *
                                sizeof(double)) == cudaSuccess;
          }
          if (ok) {
            ok = run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr,
                                   workspace, pivots_ptr, getrf_info_ptr,
                                   getrs_info_ptr) &&
                 cudaDeviceSynchronize() == cudaSuccess;
          }
        }
        ok = ok &&
             gpu_suite_clock_now(&end, error, sizeof(error)) == GPU_SUITE_OK &&
             gpu_suite_utc_timestamp(end_timestamp, error, sizeof(error)) ==
                 GPU_SUITE_OK;
        if (ok) {
          const double elapsed = gpu_suite_clock_elapsed(&start, &end);
          result.measurement_start_timestamp = start_timestamp;
          result.measurement_end_timestamp = end_timestamp;
          result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
          result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
          result.getrf_info = gpu_suite_optional_int_value(getrf_info[0]);
          result.getrs_info = gpu_suite_optional_int_value(getrs_info[0]);
          const bool verified = verify(result, options, rhs, canonical_matrix,
                                       canonical_rhs, n, nrhs);
          const bool pass =
              getrf_info[0] == 0 && getrs_info[0] == 0 && verified;
          result.attempted = true;
          result.failure_origin = pass ? nullptr : "verification";
          result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          result.status = pass ? "success" : "failure";
          result.message = pass ? "" : "OpenACC solve verification failed";
          any_failure = any_failure || !pass;
        } else {
          result.attempted = true;
          result.failure_origin = "benchmark";
          result.verification_status = "skipped";
          result.exit_code = gpu_suite_optional_int_value(1);
          result.status = "failure";
          result.message = "OpenACC end-to-end solve failed";
          any_failure = true;
        }
        writer.write(result);
        gpu_suite_result_destroy(&result);
        if (workspace != nullptr)
          cudaFree(workspace);
        if (handle != nullptr)
          cusolverDnDestroy(handle);
        if (!ok) {
          gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                     "prior-failure",
                                     "prior OpenACC cuSOLVER failure");
          break;
        }
      }
    }
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  }

  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
