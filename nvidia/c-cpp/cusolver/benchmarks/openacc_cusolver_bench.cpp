#include "gpu_suite/benchmark.hpp"
#include "solver_bench_common.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_success(cudaError_t status, const char *api) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, cudaGetErrorString(status));
  return false;
}

bool solver_success(cusolverStatus_t status, const char *api) {
  if (status == CUSOLVER_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuSOLVER status %d\n", api,
               static_cast<int>(status));
  return false;
}

void make_dense_system(int n, int nrhs, double *matrix, double *rhs) {
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      matrix[row + static_cast<std::size_t>(column) * n] =
          row == column ? n + 1.0 : 1.0;
    }
  }
  std::fill(rhs, rhs + static_cast<std::size_t>(n) * nrhs, 2.0 * n);
}

bool run_library_calls(cusolverDnHandle_t handle, int n, int nrhs,
                       double *matrix, double *rhs, double *workspace,
                       int *pivots, int *getrf_info, int *getrs_info) {
  bool ok = true;
#pragma acc host_data use_device(matrix, rhs, pivots, getrf_info, getrs_info)
  {
    ok = solver_success(cusolverDnDgetrf(handle, n, n, matrix, n, workspace,
                                         pivots, getrf_info),
                        "cusolverDnDgetrf") &&
         solver_success(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs, matrix,
                                         n, pivots, rhs, n, getrs_info),
                        "cusolverDnDgetrs");
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

  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  int n = 0;
  int nrhs = 0;
  std::size_t matrix_count_checked = 0;
  std::size_t rhs_count_checked = 0;
  std::size_t matrix_bytes = 0;
  std::size_t rhs_bytes = 0;
  std::size_t pivot_bytes = 0;
  if (!gpu_suite_checked_u64_to_int(options.size, &n) ||
      !gpu_suite_checked_u64_to_int(options.nrhs, &nrhs) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(n),
                                  static_cast<std::size_t>(n),
                                  &matrix_count_checked) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(n),
                                  static_cast<std::size_t>(nrhs),
                                  &rhs_count_checked) ||
      !gpu_suite_checked_bytes(matrix_count_checked, sizeof(double),
                               &matrix_bytes) ||
      !gpu_suite_checked_bytes(rhs_count_checked, sizeof(double), &rhs_bytes) ||
      !gpu_suite_checked_bytes(static_cast<std::size_t>(n), sizeof(int),
                               &pivot_bytes)) {
    (void)matrix_bytes;
    (void)rhs_bytes;
    (void)pivot_bytes;
    std::fprintf(stderr,
                 "solver dimensions or byte counts exceed supported range\n");
    gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "solver dimensions exceed supported integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
  (void)matrix_bytes;
  (void)rhs_bytes;
  (void)pivot_bytes;
  bool any_failure = false;

  try {
    std::vector<double> canonical_matrix(matrix_count_checked);
    std::vector<double> canonical_rhs(rhs_count_checked);
    std::vector<double> matrix(canonical_matrix.size());
    std::vector<double> rhs(canonical_rhs.size());
    std::vector<int> pivots(n);
    std::vector<int> getrf_info(1);
    std::vector<int> getrs_info(1);
    make_dense_system(n, nrhs, canonical_matrix.data(), canonical_rhs.data());
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cudaSetDevice failed");
      writer.close();
      return EXIT_FAILURE;
    }

    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      cusolverDnHandle_t handle = nullptr;
      bool setup_ok =
          solver_success(cusolverDnCreate(&handle), "cusolverDnCreate");
      double *matrix_ptr = matrix.data();
      double *rhs_ptr = rhs.data();
      int *pivots_ptr = pivots.data();
      int *getrf_info_ptr = getrf_info.data();
      int *getrs_info_ptr = getrs_info.data();
      const std::size_t matrix_count = matrix.size();
      const std::size_t rhs_count = rhs.size();
      int workspace_count = 0;
      std::size_t workspace_bytes = 0;
      double *workspace = nullptr;

#pragma acc data copy(matrix_ptr[0 : matrix_count], rhs_ptr[0 : rhs_count])    \
    create(pivots_ptr[0 : n], getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
      {
#pragma acc host_data use_device(matrix_ptr)
        {
          if (setup_ok) {
            setup_ok = solver_success(
                cusolverDnDgetrf_bufferSize(handle, n, n, matrix_ptr, n,
                                            &workspace_count),
                "cusolverDnDgetrf_bufferSize");
          }
        }
        if (setup_ok) {
          setup_ok = workspace_count >= 0 &&
                     gpu_suite_checked_bytes(
                         static_cast<std::size_t>(workspace_count),
                         sizeof(double), &workspace_bytes);
          if (!setup_ok)
            std::fprintf(stderr, "invalid cuSOLVER workspace byte count\n");
          if (setup_ok && workspace_bytes != 0U)
            setup_ok = cuda_success(
                cudaMalloc(reinterpret_cast<void **>(&workspace),
                           workspace_bytes),
                "cudaMalloc workspace");
        }
        for (int warmup = 0; warmup < options.warmup && setup_ok; ++warmup) {
          matrix = canonical_matrix;
          rhs = canonical_rhs;
          std::fill(pivots.begin(), pivots.end(), 0);
          getrf_info[0] = -1;
          getrs_info[0] = -1;
#pragma acc update device(matrix_ptr[0 : matrix_count],                        \
                          rhs_ptr[0 : rhs_count], pivots_ptr[0 : n],           \
                          getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
          setup_ok =
              run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr, workspace,
                                pivots_ptr, getrf_info_ptr, getrs_info_ptr) &&
              cuda_success(cudaDeviceSynchronize(), "warmup synchronize");
        }

        for (int trial = 0; trial < options.trials && setup_ok; ++trial) {
          matrix = canonical_matrix;
          rhs = canonical_rhs;
          std::fill(pivots.begin(), pivots.end(), 0);
          getrf_info[0] = -1;
          getrs_info[0] = -1;
#pragma acc update device(matrix_ptr[0 : matrix_count],                        \
                          rhs_ptr[0 : rhs_count], pivots_ptr[0 : n],           \
                          getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
          gpu_suite_result result;
          if (!gpu_suite::initialize_result(result, options, trial)) {
            setup_ok = false;
            any_failure = true;
            break;
          }
          char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
          struct timespec start;
          struct timespec end;
          bool ok =
              cuda_success(cudaDeviceSynchronize(),
                           "OpenACC update device solver state") &&
              gpu_suite_measurement_start(start_timestamp, &start, error,
                                           sizeof(error)) == GPU_SUITE_OK &&
              run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr, workspace,
                                pivots_ptr, getrf_info_ptr, getrs_info_ptr) &&
              cuda_success(cudaDeviceSynchronize(),
                           "post-solve synchronize") &&
              gpu_suite_measurement_end(&end, end_timestamp, error,
                                         sizeof(error)) == GPU_SUITE_OK;
#pragma acc update self(rhs_ptr[0 : rhs_count], getrf_info_ptr[0 : 1],         \
                        getrs_info_ptr[0 : 1])
          ok = ok && cuda_success(cudaDeviceSynchronize(),
                                  "OpenACC update self solver results");
          if (ok) {
            const double elapsed = gpu_suite_clock_elapsed(&start, &end);
            result.measurement_start_timestamp = start_timestamp;
            result.measurement_end_timestamp = end_timestamp;
            result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
            result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
            result.getrf_info = gpu_suite_optional_int_value(getrf_info[0]);
            result.getrs_info = gpu_suite_optional_int_value(getrs_info[0]);
            const gpu_suite::VerificationOutcome outcome =
                gpu_suite_cusolver::set_solver_verification(
                    result, options, rhs, canonical_matrix, canonical_rhs, n,
                    nrhs);
            const bool constructed =
                outcome != gpu_suite::VerificationOutcome::construction_error;
            if (constructed &&
                (getrf_info[0] != 0 || getrs_info[0] != 0))
              result.verification_status = "failure";
            const bool pass =
                getrf_info[0] == 0 && getrs_info[0] == 0 &&
                outcome == gpu_suite::VerificationOutcome::pass;
            result.attempted = true;
            result.failure_origin =
                pass ? nullptr
                     : (constructed ? "verification" : "benchmark");
            if (!constructed)
              result.verification_status = "skipped";
            result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
            result.status = pass ? "success" : "failure";
            result.message =
                pass ? ""
                     : (constructed
                            ? "OpenACC solve verification failed"
                            : "verification result construction failed");
            any_failure = any_failure || !pass;
            ok = ok && constructed;
          } else {
            result.attempted = true;
            result.failure_origin = "benchmark";
            result.verification_status = "skipped";
            result.exit_code = gpu_suite_optional_int_value(1);
            result.status = "failure";
            result.message = "OpenACC cuSOLVER pipeline failed";
            any_failure = true;
          }
          if (!writer.write(result)) {
            ok = false;
            any_failure = true;
          }
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
      bool end_to_end_ready = true;
      for (int warmup = 0;
           warmup < options.warmup && end_to_end_ready; ++warmup) {
        matrix = canonical_matrix;
        rhs = canonical_rhs;
        std::fill(pivots.begin(), pivots.end(), 0);
        getrf_info[0] = -1;
        getrs_info[0] = -1;
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
        std::size_t workspace_bytes = 0;
        end_to_end_ready = solver_success(cusolverDnCreate(&handle),
                                          "warmup cusolverDnCreate");
#pragma acc data copyin(matrix_ptr[0 : matrix_count], rhs_ptr[0 : rhs_count])  \
    create(pivots_ptr[0 : n], getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
        {
#pragma acc host_data use_device(matrix_ptr)
          {
            if (end_to_end_ready)
              end_to_end_ready = solver_success(
                  cusolverDnDgetrf_bufferSize(handle, n, n, matrix_ptr, n,
                                              &workspace_count),
                  "warmup cusolverDnDgetrf_bufferSize");
          }
          if (end_to_end_ready) {
            end_to_end_ready =
                workspace_count >= 0 &&
                gpu_suite_checked_bytes(
                    static_cast<std::size_t>(workspace_count), sizeof(double),
                    &workspace_bytes);
            if (!end_to_end_ready)
              std::fprintf(stderr,
                           "invalid warmup cuSOLVER workspace byte count\n");
            if (end_to_end_ready && workspace_bytes != 0U)
              end_to_end_ready = cuda_success(
                  cudaMalloc(reinterpret_cast<void **>(&workspace),
                             workspace_bytes),
                  "warmup cudaMalloc workspace");
          }
          if (end_to_end_ready)
            end_to_end_ready =
                run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr,
                                  workspace, pivots_ptr, getrf_info_ptr,
                                  getrs_info_ptr) &&
                cuda_success(cudaDeviceSynchronize(),
                             "warmup solve synchronize");
#pragma acc update self(rhs_ptr[0 : rhs_count], getrf_info_ptr[0 : 1],         \
                        getrs_info_ptr[0 : 1])
          if (end_to_end_ready)
            end_to_end_ready = cuda_success(
                cudaDeviceSynchronize(), "warmup OpenACC update self");
        }
        if (workspace != nullptr)
          cudaFree(workspace);
        if (handle != nullptr)
          cusolverDnDestroy(handle);
      }
      if (!end_to_end_ready) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "OpenACC cuSOLVER warmup failed");
        any_failure = true;
      }
      for (int trial = 0;
           trial < options.trials && end_to_end_ready; ++trial) {
        matrix = canonical_matrix;
        rhs = canonical_rhs;
        std::fill(pivots.begin(), pivots.end(), 0);
        getrf_info[0] = -1;
        getrs_info[0] = -1;
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
        std::size_t workspace_bytes = 0;
        gpu_suite_result result;
        if (!gpu_suite::initialize_result(result, options, trial)) {
          any_failure = true;
          break;
        }
        char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
        struct timespec start;
        struct timespec end;
        bool ok = gpu_suite_measurement_start(start_timestamp, &start, error,
                                               sizeof(error)) == GPU_SUITE_OK &&
                  solver_success(cusolverDnCreate(&handle),
                                 "cusolverDnCreate");
#pragma acc data copyin(matrix_ptr[0 : matrix_count], rhs_ptr[0 : rhs_count])  \
    create(pivots_ptr[0 : n], getrf_info_ptr[0 : 1], getrs_info_ptr[0 : 1])
        {
#pragma acc host_data use_device(matrix_ptr)
          {
            if (ok) {
              ok = solver_success(
                  cusolverDnDgetrf_bufferSize(handle, n, n, matrix_ptr, n,
                                              &workspace_count),
                  "cusolverDnDgetrf_bufferSize");
            }
          }
          if (ok) {
            ok = workspace_count >= 0 &&
                 gpu_suite_checked_bytes(
                     static_cast<std::size_t>(workspace_count), sizeof(double),
                     &workspace_bytes);
            if (!ok)
              std::fprintf(stderr,
                           "invalid cuSOLVER workspace byte count\n");
            if (ok && workspace_bytes != 0U)
              ok = cuda_success(
                  cudaMalloc(reinterpret_cast<void **>(&workspace),
                             workspace_bytes),
                  "cudaMalloc workspace");
          }
          if (ok) {
            ok = run_library_calls(handle, n, nrhs, matrix_ptr, rhs_ptr,
                                   workspace, pivots_ptr, getrf_info_ptr,
                                   getrs_info_ptr) &&
                 cuda_success(cudaDeviceSynchronize(),
                              "solve synchronize");
          }
#pragma acc update self(rhs_ptr[0 : rhs_count], getrf_info_ptr[0 : 1],         \
                        getrs_info_ptr[0 : 1])
          ok = ok && cuda_success(cudaDeviceSynchronize(),
                                  "OpenACC update self solver results");
        }
        ok = ok && gpu_suite_measurement_end(&end, end_timestamp, error,
                                              sizeof(error)) == GPU_SUITE_OK;
        if (ok) {
          const double elapsed = gpu_suite_clock_elapsed(&start, &end);
          result.measurement_start_timestamp = start_timestamp;
          result.measurement_end_timestamp = end_timestamp;
          result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
          result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
          result.getrf_info = gpu_suite_optional_int_value(getrf_info[0]);
          result.getrs_info = gpu_suite_optional_int_value(getrs_info[0]);
          const gpu_suite::VerificationOutcome outcome =
              gpu_suite_cusolver::set_solver_verification(
                  result, options, rhs, canonical_matrix, canonical_rhs, n,
                  nrhs);
          const bool constructed =
              outcome != gpu_suite::VerificationOutcome::construction_error;
          if (constructed && (getrf_info[0] != 0 || getrs_info[0] != 0))
            result.verification_status = "failure";
          const bool pass =
              getrf_info[0] == 0 && getrs_info[0] == 0 &&
              outcome == gpu_suite::VerificationOutcome::pass;
          result.attempted = true;
          result.failure_origin =
              pass ? nullptr : (constructed ? "verification" : "benchmark");
          if (!constructed)
            result.verification_status = "skipped";
          result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
          result.status = pass ? "success" : "failure";
          result.message =
              pass ? ""
                   : (constructed ? "OpenACC solve verification failed"
                                  : "verification result construction failed");
          any_failure = any_failure || !pass;
          ok = ok && constructed;
        } else {
          result.attempted = true;
          result.failure_origin = "benchmark";
          result.verification_status = "skipped";
          result.exit_code = gpu_suite_optional_int_value(1);
          result.status = "failure";
          result.message = "OpenACC end-to-end solve failed";
          any_failure = true;
        }
        if (!writer.write(result)) {
          ok = false;
          any_failure = true;
        }
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
