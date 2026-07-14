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

struct SolverContext {
  cusolverDnHandle_t handle = nullptr;
  double *matrix = nullptr;
  double *rhs = nullptr;
  double *workspace = nullptr;
  int *pivots = nullptr;
  int *getrf_info = nullptr;
  int *getrs_info = nullptr;
  int workspace_count = 0;
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
                    const std::vector<double> &rhs) {
  const std::size_t matrix_bytes = matrix.size() * sizeof(double);
  const std::size_t rhs_bytes = rhs.size() * sizeof(double);
  bool ok =
      solver_success(cusolverDnCreate(&context.handle), "cusolverDnCreate") &&
      cuda_success(
          cudaMalloc(reinterpret_cast<void **>(&context.matrix), matrix_bytes),
          "cudaMalloc matrix") &&
      cuda_success(
          cudaMalloc(reinterpret_cast<void **>(&context.rhs), rhs_bytes),
          "cudaMalloc rhs") &&
      cuda_success(cudaMalloc(reinterpret_cast<void **>(&context.pivots),
                              static_cast<std::size_t>(n) * sizeof(int)),
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
    ok = cuda_success(
        cudaMalloc(reinterpret_cast<void **>(&context.workspace),
                   static_cast<std::size_t>(context.workspace_count) *
                       sizeof(double)),
        "cudaMalloc workspace");
  }
  (void)nrhs;
  return ok;
}

bool restore_context(SolverContext &context, const std::vector<double> &matrix,
                     const std::vector<double> &rhs) {
  return cuda_success(cudaMemcpy(context.matrix, matrix.data(),
                                 matrix.size() * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "restore matrix") &&
         cuda_success(cudaMemcpy(context.rhs, rhs.data(),
                                 rhs.size() * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "restore rhs") &&
         cuda_success(cudaMemset(context.pivots, 0,
                                 matrix.size() > 0
                                     ? static_cast<std::size_t>(
                                           std::sqrt(matrix.size())) *
                                           sizeof(int)
                                     : 0),
                      "restore pivots") &&
         cuda_success(cudaMemset(context.getrf_info, 0, sizeof(int)),
                      "restore getrf_info") &&
         cuda_success(cudaMemset(context.getrs_info, 0, sizeof(int)),
                      "restore getrs_info");
}

bool solve(SolverContext &context, int n, int nrhs) {
  return solver_success(cusolverDnDgetrf(context.handle, n, n, context.matrix,
                                         n, context.workspace, context.pivots,
                                         context.getrf_info),
                        "cusolverDnDgetrf") &&
         solver_success(cusolverDnDgetrs(context.handle, CUBLAS_OP_N, n, nrhs,
                                         context.matrix, n, context.pivots,
                                         context.rhs, n, context.getrs_info),
                        "cusolverDnDgetrs");
}

bool set_verification(gpu_suite_result &result,
                      const gpu_suite_options &options,
                      const std::vector<double> &solution,
                      const std::vector<double> &matrix,
                      const std::vector<double> &rhs, int n, int nrhs) {
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
  for (const char *name : {"solution_relative_error", "relative_residual"}) {
    gpu_suite_json_value *threshold = gpu_suite_json_object();
    gpu_suite::json_add_string(threshold, "method", "absolute-plus-relative");
    gpu_suite::json_add_double(threshold, "reference_scale", 1.0);
    gpu_suite::json_add_double(threshold, "abs_tolerance",
                               options.abs_tolerance);
    gpu_suite::json_add_double(threshold, "rel_tolerance",
                               options.rel_tolerance);
    gpu_suite_json_object_set(result.verification_thresholds, name, threshold);
  }
  result.verification_primary_metric = "relative_residual";
  const double bound = options.abs_tolerance + options.rel_tolerance;
  const bool pass = std::isfinite(solution_error) &&
                    std::isfinite(relative_residual) &&
                    solution_error <= bound && relative_residual <= bound;
  result.verification_status = pass ? "pass" : "failure";
  return pass;
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

  const int n = static_cast<int>(options.size);
  const int nrhs = static_cast<int>(options.nrhs);
  gpu_suite::ResultWriter writer;
  if (!writer.open(options))
    return EXIT_FAILURE;
  bool any_failure = false;

  try {
    std::vector<double> matrix(static_cast<std::size_t>(n) * n);
    std::vector<double> rhs(static_cast<std::size_t>(n) * nrhs);
    std::vector<double> solution(rhs.size());
    make_dense_system(n, nrhs, matrix.data(), rhs.data());

    SolverContext persistent;
    if (!cuda_success(cudaSetDevice(options.device), "cudaSetDevice") ||
        !create_context(persistent, n, nrhs, matrix, rhs)) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                 "cuSOLVER setup failed");
      destroy_context(persistent);
      writer.close();
      return EXIT_FAILURE;
    }
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
      if (!restore_context(persistent, matrix, rhs) ||
          !solve(persistent, n, nrhs) ||
          !cuda_success(cudaDeviceSynchronize(), "warmup synchronize")) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                   "cuSOLVER warmup failed");
        destroy_context(persistent);
        writer.close();
        return EXIT_FAILURE;
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
            solve(persistent, n, nrhs) &&
            cuda_success(cudaDeviceSynchronize(), "post-solve synchronize") &&
            gpu_suite_measurement_end(&end, end_timestamp, error,
                                       sizeof(error)) == GPU_SUITE_OK;
        if (ok) {
          elapsed = gpu_suite_clock_elapsed(&start, &end);
          ok = cuda_success(cudaMemcpy(solution.data(), persistent.rhs,
                                       solution.size() * sizeof(double),
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
        SolverContext current;
        ok =
            gpu_suite_measurement_start(start_timestamp, &start, error,
                                         sizeof(error)) == GPU_SUITE_OK &&
            create_context(current, n, nrhs, matrix, rhs) &&
            solve(current, n, nrhs) &&
            cuda_success(cudaDeviceSynchronize(), "solve synchronize") &&
            cuda_success(cudaMemcpy(solution.data(), current.rhs,
                                    solution.size() * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "copy solution") &&
            cuda_success(cudaMemcpy(&getrf_info, current.getrf_info,
                                    sizeof(int), cudaMemcpyDeviceToHost),
                         "copy getrf_info") &&
            cuda_success(cudaMemcpy(&getrs_info, current.getrs_info,
                                    sizeof(int), cudaMemcpyDeviceToHost),
                         "copy getrs_info") &&
            gpu_suite_measurement_end(&end, end_timestamp, error,
                                       sizeof(error)) == GPU_SUITE_OK;
        if (ok)
          elapsed = gpu_suite_clock_elapsed(&start, &end);
        destroy_context(current);
      }

      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
        result.getrf_info = gpu_suite_optional_int_value(getrf_info);
        result.getrs_info = gpu_suite_optional_int_value(getrs_info);
        const bool verified =
            set_verification(result, options, solution, matrix, rhs, n, nrhs);
        const bool pass = getrf_info == 0 && getrs_info == 0 && verified;
        result.attempted = true;
        result.failure_origin = pass ? nullptr : "verification";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message = pass ? "" : "cuSOLVER verification/info failure";
        any_failure = any_failure || !pass;
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
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  }

  if (!writer.close())
    any_failure = true;
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
