#ifndef GPU_SUITE_CUSPARSE_SPARSE_BENCH_COMMON_HPP
#define GPU_SUITE_CUSPARSE_SPARSE_BENCH_COMMON_HPP

#include "gpu_suite/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpu_suite_cusparse {

inline bool set_spmv_verification(gpu_suite_result &result,
                                  const gpu_suite_options &options,
                                  const std::vector<double> &values, int nx,
                                  int ny, int updates) {
  gpu_suite_json_free(result.verification_metrics);
  gpu_suite_json_free(result.verification_thresholds);
  result.verification_metrics = gpu_suite_json_object();
  result.verification_thresholds = gpu_suite_json_object();
  if (!options.verify) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "skipped";
    return true;
  }

  double maximum = 0.0;
  double reference_scale = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int neighbor_count =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      const double product = 4.0 - neighbor_count;
      double expected = 1.0;
      for (int update = 0; update < updates; ++update) {
        expected = options.alpha * product + options.beta * expected;
      }
      maximum = std::max(maximum, std::fabs(values[iy * nx + ix] - expected));
      reference_scale = std::max(reference_scale, std::fabs(expected));
    }
  }
  if (!std::isfinite(maximum) || !std::isfinite(reference_scale)) {
    result.verification_primary_metric = nullptr;
    result.verification_status = "nonfinite";
    return false;
  }
  gpu_suite::json_add_double(result.verification_metrics, "max_abs_error",
                             maximum);
  gpu_suite_json_value *threshold = gpu_suite_json_object();
  gpu_suite::json_add_string(threshold, "method", "absolute-plus-relative");
  gpu_suite::json_add_double(threshold, "reference_scale", reference_scale);
  gpu_suite::json_add_double(threshold, "abs_tolerance", options.abs_tolerance);
  gpu_suite::json_add_double(threshold, "rel_tolerance", options.rel_tolerance);
  gpu_suite_json_object_set(result.verification_thresholds, "max_abs_error",
                            threshold);
  const bool pass = maximum <= options.abs_tolerance +
                                   options.rel_tolerance * reference_scale;
  result.verification_primary_metric = "max_abs_error";
  result.verification_status = pass ? "pass" : "failure";
  return pass;
}

} // namespace gpu_suite_cusparse

#endif
