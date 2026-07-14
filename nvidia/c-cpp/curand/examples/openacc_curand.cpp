#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const size_t count = size_t{1} << 24;
  std::vector<double> values(count);
  double *data = values.data();
  curandGenerator_t generator = nullptr;
  if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) !=
      CURAND_STATUS_SUCCESS)
    return EXIT_FAILURE;
  bool ok = curandSetPseudoRandomGeneratorSeed(generator, 1234ULL) ==
                CURAND_STATUS_SUCCESS &&
            curandSetGeneratorOffset(generator, 0ULL) == CURAND_STATUS_SUCCESS;
#pragma acc data copyout(data[0 : count])
  {
#pragma acc host_data use_device(data)
    {
      if (ok)
        ok = curandGenerateUniformDouble(generator, data, count) ==
             CURAND_STATUS_SUCCESS;
    }
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
  }
  ok = ok && curandDestroyGenerator(generator) == CURAND_STATUS_SUCCESS;
  auto bounds = std::minmax_element(values.begin(), values.end());
  std::printf(
      "OpenACC-managed cuRAND generated %zu values; observed %.17g to %.17g\n",
      count, *bounds.first, *bounds.second);
  return ok && *bounds.first >= 0 && *bounds.second <= 1 ? EXIT_SUCCESS
                                                         : EXIT_FAILURE;
}
