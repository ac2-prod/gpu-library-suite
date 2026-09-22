#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

struct square_value {
  __host__ __device__ double operator()(double x) const { return x * x; }
};
int main() {
  const size_t count = size_t{1} << 24;
  std::vector<double> values(count, 1.0);
  double *data = values.data();
  double result = 0.0;
  bool synchronized = false;
#pragma acc data copyin(data[0 : count])
  {
#pragma acc host_data use_device(data)
    {
      auto begin = thrust::device_pointer_cast(data);
      result = thrust::transform_reduce(begin, begin + count, square_value{},
                                        0.0, thrust::plus<double>());
    }
    // transform_reduce is synchronous; this explicit call keeps the teaching
    // boundary visible.
    synchronized = cudaDeviceSynchronize() == cudaSuccess;
  }
  std::printf("OpenACC-managed Thrust sum(values[i]^2) = %.17g\n", result);
  return synchronized && result == (double)count ? EXIT_SUCCESS : EXIT_FAILURE;
}
