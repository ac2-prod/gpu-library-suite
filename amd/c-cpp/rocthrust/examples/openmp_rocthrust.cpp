#include <hip/hip_runtime.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif
// Compile rocthrust_wrapper.hip separately with HIP and explicitly link it.
extern "C" int rocthrust_square_sum(const double *, std::size_t, double *) noexcept;

int main() {
  const std::size_t num_elem = std::size_t{1} << 24;
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices() || hipSetDevice(0) != hipSuccess)
    return EXIT_FAILURE;
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) {
    std::fprintf(stderr, "OpenMP host fallback rejected\n");
    return EXIT_FAILURE;
  }
  std::vector<double> values(num_elem, 1.0);
  double *data = values.data();
  double result = 0.0;
  bool ok = false;
#pragma omp target data device(device) map(to: data[0:num_elem]) use_device_ptr(data)
  {
    hipPointerAttribute_t attributes{};
    ok = hipPointerGetAttributes(&attributes, data) == hipSuccess &&
         attributes.type == hipMemoryTypeDevice && attributes.device == 0 &&
         rocthrust_square_sum(data, num_elem, &result) == 0;
  }
  std::printf("sum(values[i]^2) = %.17g (expected %zu)\n", result, num_elem);
  return ok && std::isfinite(result) && result == static_cast<double>(num_elem)
      ? EXIT_SUCCESS : EXIT_FAILURE;
}
