#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdio>
#include <cstdlib>

struct square_value {
  __host__ __device__ double operator()(double x) const { return x * x; }
};
int main() {
  const size_t count = size_t{1} << 24;
  thrust::device_vector<double> values(count, 1.0);
  double result =
      thrust::transform_reduce(values.begin(), values.end(), square_value{},
                               0.0, thrust::plus<double>());
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "cudaDeviceSynchronize failed\n");
    return EXIT_FAILURE;
  }
  std::printf("Thrust sum(values[i]^2) = %.17g\n", result);
  return result == (double)count ? EXIT_SUCCESS : EXIT_FAILURE;
}
