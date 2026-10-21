// Benchmark-only lifetime adapter. Teaching thrust_wrapper.cu is unchanged.
// Match the C++ CUDA benchmark's device_vector allocation and scalar reduction.
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cstdio>
#include <exception>
#include <limits>

namespace {
struct SquareValue {
  __host__ __device__ double operator()(double value) const {
    return value * value;
  }
};
using Values = thrust::device_vector<double>;
}

extern "C" void *gpu_suite_thrust_create(const double *values, int count) {
  try {
    return new Values(values, values + count);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Thrust device_vector: %s\n", error.what());
    return nullptr;
  }
}

extern "C" double gpu_suite_thrust_reduce(void *context) {
  const auto &values = *static_cast<const Values *>(context);
  try {
    return thrust::transform_reduce(values.begin(), values.end(), SquareValue{},
                                    0.0, thrust::plus<double>());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Thrust transform_reduce: %s\n", error.what());
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" void gpu_suite_thrust_destroy(void *context) {
  delete static_cast<Values *>(context);
}
