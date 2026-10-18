#ifndef GPU_SUITE_ROCFFT_CONTEXT_HPP
#define GPU_SUITE_ROCFFT_CONTEXT_HPP

#include "gpu_suite/fft_bench_result.hpp"
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
#include <cstdio>
#include <vector>

namespace gpu_suite_rocfft {
inline bool hip_ok(hipError_t status, const char *operation) {
  if (status == hipSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}
inline bool fft_ok(rocfft_status status, const char *operation) {
  if (status == rocfft_status_success) return true;
  std::fprintf(stderr, "%s failed: rocFFT status %d\n", operation, static_cast<int>(status));
  return false;
}

// rocFFT global initialization is process-wide, like HIP runtime setup.
// Per-transform plans, execution information and workspace remain scope-owned.
struct RuntimeContext {
  bool initialized = false;
  bool initialize() {
    initialized = fft_ok(rocfft_setup(), "rocfft_setup");
    return initialized;
  }
  bool destroy() {
    if (!initialized) return true;
    initialized = false;
    return fft_ok(rocfft_cleanup(), "rocfft_cleanup");
  }
  ~RuntimeContext() { (void)destroy(); }
};

// Only rocFFT workspace is HIP-owned. OpenMP array storage is never owned here.
struct PlanContext {
  rocfft_plan plan = nullptr;
  rocfft_execution_info info = nullptr;
  void *workspace = nullptr;

  bool destroy() {
    bool ok = true;
    if (info) ok = fft_ok(rocfft_execution_info_destroy(info), "rocfft_execution_info_destroy") && ok;
    if (plan) ok = fft_ok(rocfft_plan_destroy(plan), "rocfft_plan_destroy") && ok;
    if (workspace) ok = hip_ok(hipFree(workspace), "hipFree(workspace)") && ok;
    info = nullptr; plan = nullptr; workspace = nullptr;
    return ok;
  }
  bool create(std::size_t nfft, std::size_t batch) {
    if (!fft_ok(rocfft_plan_create(&plan, rocfft_placement_notinplace,
                                  rocfft_transform_type_complex_forward,
                                  rocfft_precision_single, 1, &nfft, batch,
                                  nullptr), "rocfft_plan_create") ||
        !fft_ok(rocfft_execution_info_create(&info), "rocfft_execution_info_create")) {
      (void)destroy(); return false;
    }
    std::size_t bytes = 0;
    if (!fft_ok(rocfft_plan_get_work_buffer_size(plan, &bytes), "rocfft_plan_get_work_buffer_size") ||
        (bytes != 0 && (!hip_ok(hipMalloc(&workspace, bytes), "hipMalloc(workspace)") ||
                       !fft_ok(rocfft_execution_info_set_work_buffer(info, workspace, bytes), "rocfft_execution_info_set_work_buffer")))) {
      (void)destroy(); return false;
    }
    return true;
  }
  bool execute(float2 *input, float2 *output, int repeat) const {
    void *in = input;
    void *out = output;
    bool ok = true;
    for (int i = 0; i < repeat; ++i) {
      if (!fft_ok(rocfft_execute(plan, &in, &out, info), "rocfft_execute")) {
        ok = false; break;
      }
    }
    // One completion synchronization, after the repeated calls (also on error).
    return hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") && ok;
  }
};

inline void canonical_host(std::vector<float2> &input, std::vector<float2> &output) {
  for (auto &v : input) v = {1.0F, 0.0F};
  for (auto &v : output) v = {0.0F, 0.0F};
}
inline bool dimensions(const gpu_suite_options &options, std::size_t &count,
                       std::size_t &bytes, int &nfft, int &batch) {
  // Keep the existing FFTW-backed comparison domain: both dimensions fit int.
  return gpu_suite_checked_u64_to_int(options.size, &nfft) &&
         gpu_suite_checked_u64_to_int(options.batch, &batch) &&
         gpu_suite_checked_u64_to_size(options.size, &count) &&
         gpu_suite_checked_mul_size(count, static_cast<std::size_t>(batch), &count) &&
         gpu_suite_checked_bytes(count, sizeof(float2), &bytes);
}
} // namespace gpu_suite_rocfft
#endif
