#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif
static bool hip_ok(hipError_t code) {
  if (code == hipSuccess) return true;
  std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(code));
  return false;
}
static bool rand_ok(rocrand_status code) {
  if (code == ROCRAND_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "rocRAND status %d\n", static_cast<int>(code));
  return false;
}
int main() {
  const std::size_t num_rand = std::size_t{1} << 24;
  const int device = omp_get_default_device();
  int initial = 1;
  if (device < 0 || device >= omp_get_num_devices() || !hip_ok(hipSetDevice(0)))
    return EXIT_FAILURE;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) {
    std::fprintf(stderr, "OpenMP host fallback rejected\n");
    return EXIT_FAILURE;
  }
  std::vector<double> values(num_rand);
  double *data = values.data();
  rocrand_generator generator = nullptr;
  bool ok = rand_ok(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT)) &&
            rand_ok(rocrand_set_seed(generator, 1234)) &&
            rand_ok(rocrand_set_offset(generator, 0)) &&
            rand_ok(rocrand_set_ordering(generator, ROCRAND_ORDERING_PSEUDO_DEFAULT));
  if (ok) {
#pragma omp target data device(device) map(from: data[0:num_rand]) use_device_ptr(data)
    {
      hipPointerAttribute_t attributes{};
      ok = hip_ok(hipPointerGetAttributes(&attributes, data)) &&
           attributes.type == hipMemoryTypeDevice && attributes.device == 0 &&
           rand_ok(rocrand_generate_uniform_double(generator, data, num_rand)) &&
           hip_ok(hipDeviceSynchronize());
    } // OpenMP owns the mapping; completion precedes the copy back.
  }
  if (ok) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    ok = std::all_of(values.begin(), values.end(), [](double x) {
      return std::isfinite(x) && x > 0.0 && x <= 1.0;
    });
    std::printf("rocRAND/OpenMP generated %zu doubles; range %.17g to %.17g\n",
                num_rand, *bounds.first, *bounds.second);
  }
  if (generator != nullptr && !rand_ok(rocrand_destroy_generator(generator))) ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
