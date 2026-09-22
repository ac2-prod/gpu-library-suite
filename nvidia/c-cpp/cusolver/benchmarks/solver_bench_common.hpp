#ifndef GPU_SUITE_CUSOLVER_SOLVER_BENCH_COMMON_HPP
#define GPU_SUITE_CUSOLVER_SOLVER_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <cmath>
#include <vector>

namespace gpu_suite_cusolver {

inline gpu_suite::VerificationOutcome
set_solver_verification(gpu_suite_result &result,
                        const gpu_suite_options &options,
                        const std::vector<double> &solution,
                        const std::vector<double> &matrix,
                        const std::vector<double> &rhs, int n, int nrhs) {
  if (gpu_suite_verification_reset(&result) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return gpu_suite::VerificationOutcome::pass;
  }
  if (gpu_suite_verification_add_absolute_relative_threshold(
          &result, "solution_relative_error", 1.0, options.abs_tolerance,
          options.rel_tolerance) != GPU_SUITE_OK ||
      gpu_suite_verification_add_absolute_relative_threshold(
          &result, "relative_residual", 1.0, options.abs_tolerance,
          options.rel_tolerance) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;

  bool finite = true;
  double solution_error = 0.0;
  double residual = 0.0;
  double matrix_norm = 0.0;
  double rhs_norm = 0.0;
  for (int row = 0; row < n; ++row) {
    double row_sum = 0.0;
    for (int column = 0; column < n; ++column) {
      const double entry =
          matrix[static_cast<std::size_t>(row) +
                 static_cast<std::size_t>(column) *
                     static_cast<std::size_t>(n)];
      if (!std::isfinite(entry) ||
          !std::isfinite(row_sum + std::fabs(entry))) {
        finite = false;
        continue;
      }
      row_sum += std::fabs(entry);
    }
    if (!gpu_suite_finite_max_update(row_sum, &matrix_norm))
      finite = false;
  }
  for (int column = 0; column < nrhs; ++column) {
    for (int row = 0; row < n; ++row) {
      const std::size_t index =
          static_cast<std::size_t>(row) +
          static_cast<std::size_t>(column) * static_cast<std::size_t>(n);
      double error = 0.0;
      if (!gpu_suite_finite_absolute_error(solution[index], 1.0, &error) ||
          !gpu_suite_finite_max_update(error, &solution_error))
        finite = false;
      if (!std::isfinite(rhs[index]) ||
          !gpu_suite_finite_max_update(std::fabs(rhs[index]), &rhs_norm))
        finite = false;

      double product = 0.0;
      bool product_finite = true;
      for (int inner = 0; inner < n; ++inner) {
        const double matrix_value =
            matrix[static_cast<std::size_t>(row) +
                   static_cast<std::size_t>(inner) *
                       static_cast<std::size_t>(n)];
        const double solution_value =
            solution[static_cast<std::size_t>(inner) +
                     static_cast<std::size_t>(column) *
                         static_cast<std::size_t>(n)];
        const double term = matrix_value * solution_value;
        if (!std::isfinite(matrix_value) || !std::isfinite(solution_value) ||
            !std::isfinite(term) || !std::isfinite(product + term)) {
          finite = false;
          product_finite = false;
          continue;
        }
        product += term;
      }
      if (product_finite &&
          (!gpu_suite_finite_absolute_error(product, rhs[index], &error) ||
           !gpu_suite_finite_max_update(error, &residual)))
        finite = false;
    }
  }
  const double denominator = matrix_norm + rhs_norm;
  const double relative_residual = residual / denominator;
  finite = finite && std::isfinite(denominator) && denominator > 0.0 &&
           std::isfinite(relative_residual);
  result.verification_primary_metric = "relative_residual";
  if (!finite) {
    if (gpu_suite_json_add_null(result.verification_metrics,
                                "solution_relative_error") != GPU_SUITE_OK ||
        gpu_suite_json_add_null(result.verification_metrics,
                                "relative_residual") != GPU_SUITE_OK)
      return gpu_suite::VerificationOutcome::construction_error;
    result.verification_status = "nonfinite";
    return gpu_suite::VerificationOutcome::failure;
  }
  if (gpu_suite::json_add_double(result.verification_metrics,
                                 "solution_relative_error",
                                 solution_error) != GPU_SUITE_OK ||
      gpu_suite::json_add_double(result.verification_metrics,
                                 "relative_residual",
                                 relative_residual) != GPU_SUITE_OK)
    return gpu_suite::VerificationOutcome::construction_error;
  const double bound = options.abs_tolerance + options.rel_tolerance;
  const bool pass =
      solution_error <= bound && relative_residual <= bound;
  result.verification_status = pass ? "pass" : "failure";
  return pass ? gpu_suite::VerificationOutcome::pass
              : gpu_suite::VerificationOutcome::failure;
}

} // namespace gpu_suite_cusolver

#endif
