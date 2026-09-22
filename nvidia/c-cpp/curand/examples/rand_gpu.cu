#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const size_t count = size_t{1} << 24;
  std::vector<double> values(count);
  double *device = nullptr;
  curandGenerator_t generator = nullptr;
  int status = EXIT_FAILURE;
  if (cudaMalloc((void **)&device, count * sizeof(double)) != cudaSuccess ||
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) !=
          CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(generator, 1234ULL) !=
          CURAND_STATUS_SUCCESS ||
      curandSetGeneratorOffset(generator, 0ULL) != CURAND_STATUS_SUCCESS ||
      curandGenerateUniformDouble(generator, device, count) !=
          CURAND_STATUS_SUCCESS ||
      cudaMemcpy(values.data(), device, count * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "cuRAND pipeline failed\n");
    goto cleanup;
  }
  {
    auto bounds = std::minmax_element(values.begin(), values.end());
    std::printf("cuRAND generated %zu uniform doubles in (0,1]; observed %.17g "
                "to %.17g\n",
                count, *bounds.first, *bounds.second);
    status =
        *bounds.first >= 0 && *bounds.second <= 1 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (generator)
    curandDestroyGenerator(generator);
  if (device)
    cudaFree(device);
  return status;
}
