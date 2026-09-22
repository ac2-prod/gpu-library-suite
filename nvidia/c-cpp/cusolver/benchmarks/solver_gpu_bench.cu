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

struct SolverContext {
  cusolverDnHandle_t handle = nullptr;
  double *matrix = nullptr;
  double *rhs = nullptr;
  double *workspace = nullptr;
  int *pivots = nullptr;
  int *getrf_info = nullptr;
  int *getrs_info = nullptr;
  int n = 0;
  int workspace_count = 0;
  std::size_t matrix_bytes = 0;
  std::size_t rhs_bytes = 0;
  std::size_t pivot_bytes = 0;
  std::size_t workspace_bytes = 0;
};

void make_dense_system(int n, int nrhs, double *matrix, double *rhs) {
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      matrix[row + static_cast<std::size_t>(column) * n] =
          row == column ? n + 1.0 : 1.0;
    }
  }
  std::fill(rhs, rhs + static_cast<std::size_t>(n) * nrhs, 2.0 * n);
}

bool cuda_success(cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool solver_success(cusolverStatus_t status, const char *operation) {
  if (status == CUSOLVER_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: cuSOLVER status %d\n", operation,
               static_cast<int>(status));
  return false;
}

void destroy_context(SolverContext &context) {
  if (context.workspace != nullptr)
    cudaFree(context.workspace);
  if (context.getrs_info != nullptr)
    cudaFree(context.getrs_info);
  if (context.getrf_info != nullptr)
    cudaFree(context.getrf_info);
  if (context.pivots != nullptr)
    cudaFree(context.pivots);
  if (context.rhs != nullptr)
    cudaFree(context.rhs);
  if (context.matrix != nullptr)
    cudaFree(context.matrix);
  if (context.handle != nullptr)
    cusolverDnDestroy(context.handle);
  context = SolverContext{};
}

bool create_context(SolverContext &context, int n, int nrhs,
                    const std::vector<double> &matrix,
                    const std::vector<double> &rhs,
                    std::size_t matrix_bytes, std::size_t rhs_bytes,
                    std::size_t pivot_bytes) {
  context.n = n;
  context.matrix_bytes = matrix_bytes;
  context.rhs_bytes = rhs_bytes;
  context.pivot_bytes = pivot_bytes;
  bool ok =
      solver_success(cusolverDnCreate(&context.handle), "cusolverDnCreate") &&
      cuda_success(
          cudaMalloc(reinterpret_cast<void **>(&context.matrix), matrix_bytes),
          "cudaMalloc matrix") &&
      cuda_success(
          cudaMalloc(reinterpret_cast<void **>(&context.rhs), rhs_bytes),
          "cudaMalloc rhs") &&
      cuda_success(cudaMalloc(reinterpret_cast<void **>(&context.pivots),
                              pivot_bytes),
                   "cudaMalloc pivots") &&
      cuda_success(cudaMalloc(reinterpret_cast<void **>(&context.getrf_info),
                              sizeof(int)),
                   "cudaMalloc getrf_info") &&
      cuda_success(cudaMalloc(reinterpret_cast<void **>(&context.getrs_info),
                              sizeof(int)),
                   "cudaMalloc getrs_info") &&
      cuda_success(cudaMemcpy(context.matrix, matrix.data(), matrix_bytes,
                              cudaMemcpyHostToDevice),
                   "copy matrix") &&
      cuda_success(cudaMemcpy(context.rhs, rhs.data(), rhs_bytes,
                              cudaMemcpyHostToDevice),
                   "copy rhs") &&
      solver_success(cusolverDnDgetrf_bufferSize(context.handle, n, n,
                                                 context.matrix, n,
                                                 &context.workspace_count),
                     "cusolverDnDgetrf_bufferSize");
  if (ok) {
    if (context.workspace_count < 0 ||
        !gpu_suite_checked_bytes(
            static_cast<std::size_t>(context.workspace_count), sizeof(double),
            &context.workspace_bytes)) {
      std::fprintf(stderr,
                   "cuSOLVER workspace count or byte count is invalid\n");
      ok = false;
    } else if (context.workspace_bytes != 0U) {
      ok = cuda_success(
          cudaMalloc(reinterpret_cast<void **>(&context.workspace),
                     context.workspace_bytes),
          "cudaMalloc workspace");
    }
  }
  (void)nrhs;
  return ok;
}

bool restore_context(SolverContext &context, const std::vector<double> &matrix,
                     const std::vector<double> &rhs) {
  return cuda_success(cudaMemcpy(context.matrix, matrix.data(),
                                 context.matrix_bytes,
                                 cudaMemcpyHostToDevice),
                      "restore matrix") &&
         cuda_success(cudaMemcpy(context.rhs, rhs.data(),
                                 context.rhs_bytes,
                                 cudaMemcpyHostToDevice),
                      "restore rhs") &&
         cuda_success(cudaMemset(context.pivots, 0, context.pivot_bytes),
                      "restore pivots") &&
         cuda_success(cudaMemset(context.getrf_info, 0, sizeof(int)),
                      "restore getrf_info") &&
         cuda_success(cudaMemset(context.getrs_info, 0, sizeof(int)),
                      "restore getrs_info");
}

bool solve(SolverContext &context, int nrhs);

bool run_end_to_end_pipeline(
    int n, int nrhs, const std::vector<double> &matrix,
    const std::vector<double> &rhs,
    std::vector<double> &solution, std::size_t matrix_bytes,
    std::size_t rhs_bytes, std::size_t pivot_bytes, bool measure,
    char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY], char *error,
    std::size_t error_size, double *elapsed, int *getrf_info,
    int *getrs_info) {
  SolverContext context;
  struct timespec start;
  struct timespec end;
  bool ok = !measure ||
            gpu_suite_measurement_start(start_timestamp, &start, error,
                                        error_size) == GPU_SUITE_OK;
  ok = ok && create_context(context, n, nrhs, matrix, rhs, matrix_bytes,
                            rhs_bytes, pivot_bytes) &&
       solve(context, nrhs) &&
       cuda_success(cudaDeviceSynchronize(), "solve synchronize") &&
       cuda_success(cudaMemcpy(solution.data(), context.rhs, rhs_bytes,
                               cudaMemcpyDeviceToHost),
                    "copy solution") &&
       cuda_success(cudaMemcpy(getrf_info, context.getrf_info, sizeof(int),
                               cudaMemcpyDeviceToHost),
                    "copy getrf_info") &&
       cuda_success(cudaMemcpy(getrs_info, context.getrs_info, sizeof(int),
                               cudaMemcpyDeviceToHost),
                    "copy getrs_info");
  if (ok && measure) {
    ok = gpu_suite_measurement_end(&end, end_timestamp, error, error_size) ==
         GPU_SUITE_OK;
    if (ok)
      *elapsed = gpu_suite_clock_elapsed(&start, &end);
  }
  destroy_context(context);
  return ok;
}

bool solve(SolverContext &context, int nrhs) {
  const int n = context.n;
  return solver_success(cusolverDnDgetrf(context.handle, n, n, context.matrix,
                                         n, context.workspace, context.pivots,
                                         context.getrf_info),
                        "cusolverDnDgetrf") &&
         solver_success(cusolverDnDgetrs(context.handle, CUBLAS_OP_N, n, nrhs,
                                         context.matrix, n, context.pivots,
                                         context.rhs, n, context.getrs_info),
                        "cusolverDnDgetrs");
}

} // namespace

int main(int argc, char **argv) {
  gpu_suite_options options;
  char error[256] = {0};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSOLVER,
                         GPU_SUITE_IMPLEMENTATION_CUDA);
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
  std::size_t matrix_count = 0;
  std::size_t rhs_count = 0;
  std::size_t matrix_bytes = 0;
  std::size_t rhs_bytes = 0;
  std::size_t pivot_bytes = 0;
  if (!gpu_suite_checked_u64_to_int(options.size, &n) ||
      !gpu_suite_checked_u64_to_int(options.nrhs, &nrhs) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(n),
                                  static_cast<std::size_t>(n),
                                  &matrix_count) ||
      !gpu_suite_checked_mul_size(static_cast<std::size_t>(n),
                                  static_cast<std::size_t>(nrhs),
                                  &rhs_count) ||
      !gpu_suite_checked_bytes(matrix_count, sizeof(double), &matrix_bytes) ||
      !gpu_suite_checked_bytes(rhs_count, sizeof(double), &rhs_bytes) ||
      !gpu_suite_checked_bytes(static_cast<std::size_t>(n), sizeof(int),
                               &pivot_bytes)) {
    std::fprintf(stderr,
                 "solver dimensions or byte counts exceed supported range\n");
    gpu_suite::emit_unmeasured(
        writer, options, 0, false, "prerequisite",
        "solver dimensions exceed supported integer or byte range");
    (void)writer.close();
    return EXIT_FAILURE;
  }
  bool any_failure = false;

  try {
    std::vector<double> matrix(matrix_count);
    std::vector<double> rhs(rhs_count);
    std::vector<double> solution(rhs_count);
    make_dense_system(n, nrhs, matrix.data(), rhs.data());

    SolverContext persistent;
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice")) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cuSOLVER setup failed");
      writer.close();
      return EXIT_FAILURE;
    }
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      if (!create_context(persistent, n, nrhs, matrix, rhs, matrix_bytes,
                          rhs_bytes, pivot_bytes)) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "cuSOLVER compute setup failed");
        destroy_context(persistent);
        writer.close();
        return EXIT_FAILURE;
      }
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        if (!restore_context(persistent, matrix, rhs) ||
            !solve(persistent, nrhs) ||
            !cuda_success(cudaDeviceSynchronize(), "warmup synchronize")) {
          gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                     "cuSOLVER warmup failed");
          destroy_context(persistent);
          writer.close();
          return EXIT_FAILURE;
        }
      }
      if (!restore_context(persistent, matrix, rhs)) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "cuSOLVER state restoration failed");
        destroy_context(persistent);
        writer.close();
        return EXIT_FAILURE;
      }
    } else {
      std::vector<double> warmup_solution(rhs_count);
      int warmup_getrf_info = 0;
      int warmup_getrs_info = 0;
      for (int warmup = 0; warmup < options.warmup; ++warmup) {
        if (!run_end_to_end_pipeline(
                n, nrhs, matrix, rhs, warmup_solution, matrix_bytes,
                rhs_bytes, pivot_bytes, false, nullptr, nullptr, error,
                sizeof(error), nullptr, &warmup_getrf_info,
                &warmup_getrs_info)) {
          gpu_suite::emit_unmeasured(
              writer, options, 0, true, "benchmark",
              "cuSOLVER end-to-end warmup failed");
          writer.close();
          return EXIT_FAILURE;
        }
      }
    }

    for (int trial = 0; trial < options.trials; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial)) {
        destroy_context(persistent);
        return EXIT_FAILURE;
      }
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec start;
      struct timespec end;
      double elapsed = 0.0;
      int getrf_info = 0;
      int getrs_info = 0;
      bool ok = true;

      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok =
            restore_context(persistent, matrix, rhs) &&
            cuda_success(cudaDeviceSynchronize(), "pre-timing synchronize") &&
            gpu_suite_measurement_start(start_timestamp, &start, error,
                                         sizeof(error)) == GPU_SUITE_OK &&
            solve(persistent, nrhs) &&
            cuda_success(cudaDeviceSynchronize(), "post-solve synchronize") &&
            gpu_suite_measurement_end(&end, end_timestamp, error,
                                       sizeof(error)) == GPU_SUITE_OK;
        if (ok) {
          elapsed = gpu_suite_clock_elapsed(&start, &end);
          ok = cuda_success(cudaMemcpy(solution.data(), persistent.rhs,
                                       rhs_bytes,
                                       cudaMemcpyDeviceToHost),
                            "copy solution") &&
               cuda_success(cudaMemcpy(&getrf_info, persistent.getrf_info,
                                       sizeof(int), cudaMemcpyDeviceToHost),
                            "copy getrf_info") &&
               cuda_success(cudaMemcpy(&getrs_info, persistent.getrs_info,
                                       sizeof(int), cudaMemcpyDeviceToHost),
                            "copy getrs_info");
        }
      } else {
        ok = run_end_to_end_pipeline(
            n, nrhs, matrix, rhs, solution, matrix_bytes, rhs_bytes,
            pivot_bytes, true, start_timestamp, end_timestamp, error,
            sizeof(error), &elapsed, &getrf_info, &getrs_info);
      }

      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
        result.getrf_info = gpu_suite_optional_int_value(getrf_info);
        result.getrs_info = gpu_suite_optional_int_value(getrs_info);
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_cusolver::set_solver_verification(
                result, options, solution, matrix, rhs, n, nrhs);
        const bool constructed =
            outcome != gpu_suite::VerificationOutcome::construction_error;
        if (constructed && (getrf_info != 0 || getrs_info != 0))
          result.verification_status = "failure";
        const bool pass = getrf_info == 0 && getrs_info == 0 &&
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
                 : (constructed ? "cuSOLVER verification/info failure"
                                : "verification result construction failed");
        any_failure = any_failure || !pass;
        ok = ok && constructed;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "cuSOLVER pipeline failed";
        any_failure = true;
      }
      if (!writer.write(result)) {
        gpu_suite_result_destroy(&result);
        destroy_context(persistent);
        return EXIT_FAILURE;
      }
      gpu_suite_result_destroy(&result);
      if (!ok) {
        gpu_suite::emit_unmeasured(writer, options, trial + 1, false,
                                   "prior-failure", "prior cuSOLVER failure");
        break;
      }
    }
    destroy_context(persistent);
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
