#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const int nfft = 1024;
  const int batch = 4096;
  const std::size_t count =
      static_cast<std::size_t>(nfft) * static_cast<std::size_t>(batch);
  std::vector<cufftComplex> input(count);
  std::vector<cufftComplex> output(count);
  cufftComplex *input_data = input.data();
  cufftComplex *output_data = output.data();
  cufftHandle plan = 0;
  int length[1] = {nfft};

  for (cufftComplex &value : input) {
    value.x = 1.0F;
    value.y = 0.0F;
  }
  const cufftResult plan_status = cufftPlanMany(
      &plan, 1, length, nullptr, 1, nfft, nullptr, 1, nfft, CUFFT_C2C, batch);
  if (plan_status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "cufftPlanMany failed with status %d\n",
                 static_cast<int>(plan_status));
    return EXIT_FAILURE;
  }

  int execution_status = EXIT_SUCCESS;
#pragma acc data copyin(input_data[0 : count]) copyout(output_data[0 : count])
  {
#pragma acc host_data use_device(input_data, output_data)
    {
      const cufftResult result =
          cufftExecC2C(plan, input_data, output_data, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecC2C failed with status %d\n",
                     static_cast<int>(result));
        execution_status = EXIT_FAILURE;
      }
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                   cudaGetErrorString(sync_status));
      execution_status = EXIT_FAILURE;
    }
  }

  const cufftResult destroy_status = cufftDestroy(plan);
  if (destroy_status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "cufftDestroy failed with status %d\n",
                 static_cast<int>(destroy_status));
    execution_status = EXIT_FAILURE;
  }
  if (execution_status != EXIT_SUCCESS) {
    return execution_status;
  }

  float max_error = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    const int frequency =
        static_cast<int>(index % static_cast<std::size_t>(nfft));
    const float expected = frequency == 0 ? static_cast<float>(nfft) : 0.0F;
    max_error = std::max(max_error, std::fabs(output[index].x - expected));
    max_error = std::max(max_error, std::fabs(output[index].y));
  }
  std::printf(
      "OpenACC-managed cuFFT batched transform complete; max error = %.9g\n",
      static_cast<double>(max_error));
  return max_error <= 1.0e-4F ? EXIT_SUCCESS : EXIT_FAILURE;
}
