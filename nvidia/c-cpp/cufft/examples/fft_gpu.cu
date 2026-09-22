#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error_ = (call);                                         \
    if (error_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(error_));                                \
      status = EXIT_FAILURE;                                                   \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    const cufftResult error_ = (call);                                         \
    if (error_ != CUFFT_SUCCESS) {                                             \
      std::fprintf(stderr, "%s failed with cuFFT status %d\n", #call,          \
                   static_cast<int>(error_));                                  \
      status = EXIT_FAILURE;                                                   \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main() {
  const int nfft = 1024;
  const int batch = 4096;
  const std::size_t count =
      static_cast<std::size_t>(nfft) * static_cast<std::size_t>(batch);
  std::vector<cufftComplex> input(count);
  std::vector<cufftComplex> output(count);
  cufftComplex *device_input = nullptr;
  cufftComplex *device_output = nullptr;
  cufftHandle plan = 0;
  bool plan_created = false;
  int length[1] = {nfft};
  int status = EXIT_FAILURE;
  float max_error = 0.0F;

  for (cufftComplex &value : input) {
    value.x = 1.0F;
    value.y = 0.0F;
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_input),
                        count * sizeof(*device_input)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output),
                        count * sizeof(*device_output)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                        count * sizeof(*device_input), cudaMemcpyHostToDevice));
  CUFFT_CHECK(cufftPlanMany(&plan, 1, length, nullptr, 1, nfft, nullptr, 1,
                            nfft, CUFFT_C2C, batch));
  plan_created = true;
  CUFFT_CHECK(cufftExecC2C(plan, device_input, device_output, CUFFT_FORWARD));
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        count * sizeof(*device_output),
                        cudaMemcpyDeviceToHost));

  for (std::size_t index = 0; index < count; ++index) {
    const int frequency =
        static_cast<int>(index % static_cast<std::size_t>(nfft));
    const float expected = frequency == 0 ? static_cast<float>(nfft) : 0.0F;
    max_error = std::fmax(max_error, std::fabs(output[index].x - expected));
    max_error = std::fmax(max_error, std::fabs(output[index].y));
  }
  std::printf(
      "cuFFT batched C2C forward transform complete; max error = %.9g\n",
      static_cast<double>(max_error));
  status = max_error <= 1.0e-4F ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
  if (plan_created) {
    const cufftResult destroy_status = cufftDestroy(plan);
    if (destroy_status != CUFFT_SUCCESS) {
      std::fprintf(stderr, "cufftDestroy failed with status %d\n",
                   static_cast<int>(destroy_status));
      status = EXIT_FAILURE;
    }
  }
  if (device_output != nullptr) {
    const cudaError_t free_status = cudaFree(device_output);
    if (free_status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree(output) failed: %s\n",
                   cudaGetErrorString(free_status));
      status = EXIT_FAILURE;
    }
  }
  if (device_input != nullptr) {
    const cudaError_t free_status = cudaFree(device_input);
    if (free_status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree(input) failed: %s\n",
                   cudaGetErrorString(free_status));
      status = EXIT_FAILURE;
    }
  }
  return status;
}
