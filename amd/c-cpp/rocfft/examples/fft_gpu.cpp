#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>

#define HIP_CHECK(call) do { if ((call) != hipSuccess) { std::fprintf(stderr, "%s failed\n", #call); goto cleanup; } } while (0)
#define FFT_CHECK(call) do { if ((call) != rocfft_status_success) { std::fprintf(stderr, "%s failed\n", #call); goto cleanup; } } while (0)

int main() {
  const size_t nfft = 1024;
  const size_t batch = 4096;
  const size_t count = nfft * batch;
  const size_t bytes = count * sizeof(float2);
  float2 *input = new (std::nothrow) float2[count];
  float2 *output = new (std::nothrow) float2[count];
  void *device_input = nullptr;
  void *device_output = nullptr;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  rocfft_plan plan = nullptr;
  rocfft_execution_info info = nullptr;
  bool initialized = false;
  float max_error = 0.0F;
  int status = EXIT_FAILURE;
  if (!input || !output) { std::fprintf(stderr, "host allocation failed\n"); goto cleanup; }
  for (size_t i = 0; i < count; ++i) input[i] = {1.0F, 0.0F};

  FFT_CHECK(rocfft_setup());
  initialized = true;
  HIP_CHECK(hipMalloc(&device_input, bytes));
  HIP_CHECK(hipMalloc(&device_output, bytes));
  HIP_CHECK(hipMemcpy(device_input, input, bytes, hipMemcpyHostToDevice));
  // Default layout is contiguous interleaved complex data, distance nfft.
  FFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
      rocfft_transform_type_complex_forward, rocfft_precision_single,
      1, &nfft, batch, nullptr));
  FFT_CHECK(rocfft_execution_info_create(&info));
  FFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &workspace_bytes));
  if (workspace_bytes != 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
    FFT_CHECK(rocfft_execution_info_set_work_buffer(info, workspace, workspace_bytes));
  }
  FFT_CHECK(rocfft_execute(plan, &device_input, &device_output, info));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output, device_output, bytes, hipMemcpyDeviceToHost));

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
  std::printf("rocFFT batched C2C forward complete; max error = %.9g\n", max_error);
  status = max_error <= 1.0e-4F ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
  if (info) (void)rocfft_execution_info_destroy(info);
  if (plan) (void)rocfft_plan_destroy(plan);
  if (workspace) (void)hipFree(workspace);
  if (device_output) (void)hipFree(device_output);
  if (device_input) (void)hipFree(device_input);
  if (initialized) (void)rocfft_cleanup();
  delete[] output;
  delete[] input;
  return status;
}
