#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>

struct SquareValue {
  __host__ __device__ double operator()(double value) const { return value * value; }
};
int main() {
  const std::size_t num_elem = std::size_t{1} << 24;
  try {
    thrust::device_vector<double> values(num_elem, 1.0);
    // The par policy completes the reduction and returns the host scalar.
    const double result = thrust::transform_reduce(thrust::hip_rocprim::par,
        values.begin(), values.end(), SquareValue{}, 0.0, thrust::plus<double>());
    std::printf("sum(values[i]^2) = %.17g (expected %zu)\n", result, num_elem);
    return std::isfinite(result) && result == static_cast<double>(num_elem)
        ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rocThrust: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
