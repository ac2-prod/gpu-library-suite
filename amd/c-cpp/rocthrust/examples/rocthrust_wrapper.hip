#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>

struct SquareValue {
  __host__ __device__ double operator()(double value) const { return value * value; }
};

// Caller owns the device mapping. This wrapper never frees or retains it.
// The synchronous par policy returns a host scalar. No exception crosses C ABI.
extern "C" int rocthrust_square_sum(const double *device_values, std::size_t count,
                                    double *result) noexcept {
  if (device_values == nullptr || result == nullptr || count == 0 ||
      count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    return 1;
  try {
    const auto first = thrust::device_pointer_cast(device_values);
    *result = thrust::transform_reduce(thrust::hip_rocprim::par, first, first + count,
                                      SquareValue{}, 0.0, thrust::plus<double>());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rocThrust: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "rocThrust: unknown exception\n");
    return 1;
  }
}
