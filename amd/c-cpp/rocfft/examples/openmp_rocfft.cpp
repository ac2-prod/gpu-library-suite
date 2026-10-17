#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>

#ifndef _OPENMP
#error "OpenMP target support is required"
#endif

#define HIP_CHECK(call) do { if ((call) != hipSuccess) { std::fprintf(stderr, "%s failed\n", #call); goto cleanup; } } while (0)
#define FFT_CHECK(call) do { if ((call) != rocfft_status_success) { std::fprintf(stderr, "%s failed\n", #call); goto cleanup; } } while (0)

int main() {
  const size_t nfft = 1024;
  const size_t batch = 4096;
  const size_t count = nfft * batch;
  float2 *input = new (std::nothrow) float2[count];
  float2 *output = new (std::nothrow) float2[count];
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  rocfft_plan plan = nullptr;
  rocfft_execution_info info = nullptr;
  bool initialized = false;
  bool executed = false;
  int initial_device = 1;
  int hip_device = -1;
  float max_error = 0.0F;
  int status = EXIT_FAILURE;
  if (!input || !output) { std::fprintf(stderr, "host allocation failed\n"); goto cleanup; }
  if (omp_get_num_devices() <= 0) {
    std::fprintf(stderr, "OpenMP target device unavailable\n"); goto cleanup;
  }
#pragma omp target map(from: initial_device)
  { initial_device = omp_is_initial_device(); }
  if (initial_device) { std::fprintf(stderr, "OpenMP host fallback rejected\n"); goto cleanup; }
  HIP_CHECK(hipGetDevice(&hip_device));
  for (size_t i = 0; i < count; ++i) input[i] = {1.0F, 0.0F};

  FFT_CHECK(rocfft_setup());
  initialized = true;
  FFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
      rocfft_transform_type_complex_forward, rocfft_precision_single,
      1, &nfft, batch, nullptr));
  FFT_CHECK(rocfft_execution_info_create(&info));
  FFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &workspace_bytes));
  if (workspace_bytes != 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
    FFT_CHECK(rocfft_execution_info_set_work_buffer(info, workspace, workspace_bytes));
  }
#pragma omp target data map(to: input[0:count]) map(from: output[0:count]) use_device_ptr(input, output)
  {
    hipPointerAttribute_t in_attributes{}, out_attributes{};
    // HIP identifies the allocations' physical device; ordinal equality alone
    // between two runtimes is insufficient evidence of interoperability.
    if (hipPointerGetAttributes(&in_attributes, input) == hipSuccess &&
        hipPointerGetAttributes(&out_attributes, output) == hipSuccess &&
        in_attributes.type == hipMemoryTypeDevice &&
        out_attributes.type == hipMemoryTypeDevice &&
        in_attributes.device == hip_device && out_attributes.device == hip_device) {
      void *in_buffer = input;
      void *out_buffer = output;
      executed = rocfft_execute(plan, &in_buffer, &out_buffer, info) == rocfft_status_success;
      const bool synchronized = hipDeviceSynchronize() == hipSuccess;
      executed = executed && synchronized;
    }
  } // map(from:) completes only after rocFFT synchronization above.
  if (!executed) { std::fprintf(stderr, "OpenMP/HIP device mapping or rocFFT execution failed\n"); goto cleanup; }
  for (size_t i = 0; i < count; ++i) {
    const float expected = i % nfft == 0 ? static_cast<float>(nfft) : 0.0F;
    const float real_error = std::fabs(output[i].x - expected);
    const float imag_error = std::fabs(output[i].y);
    if (!std::isfinite(real_error) || !std::isfinite(imag_error)) {
      std::fprintf(stderr, "nonfinite rocFFT output\n"); goto cleanup;
    }
    if (real_error > max_error) max_error = real_error;
    if (imag_error > max_error) max_error = imag_error;
  }
  std::printf("OpenMP rocFFT batched C2C forward complete; max error = %.9g\n", max_error);
  status = max_error <= 1.0e-4F ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
  if (info) (void)rocfft_execution_info_destroy(info);
  if (plan) (void)rocfft_plan_destroy(plan);
  if (workspace) (void)hipFree(workspace);
  if (initialized) (void)rocfft_cleanup();
  delete[] output;
  delete[] input;
  return status;
}
