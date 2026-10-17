#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
  std::vector<double> values(num_rand);
  double *device_values = nullptr;
  rocrand_generator generator = nullptr;
  bool ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&device_values),
                             num_rand * sizeof(double))) &&
            rand_ok(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT)) &&
            rand_ok(rocrand_set_seed(generator, 1234)) &&
            rand_ok(rocrand_set_offset(generator, 0)) &&
            rand_ok(rocrand_set_ordering(generator, ROCRAND_ORDERING_PSEUDO_DEFAULT)) &&
            rand_ok(rocrand_generate_uniform_double(generator, device_values, num_rand)) &&
            hip_ok(hipDeviceSynchronize()) &&
            hip_ok(hipMemcpy(values.data(), device_values, num_rand * sizeof(double),
                             hipMemcpyDeviceToHost));
  if (ok) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    ok = std::all_of(values.begin(), values.end(), [](double x) {
      return std::isfinite(x) && x > 0.0 && x <= 1.0;
    });
    std::printf("rocRAND generated %zu uniform doubles in (0,1]; range %.17g to %.17g\n",
                num_rand, *bounds.first, *bounds.second);
  }
  if (generator != nullptr && !rand_ok(rocrand_destroy_generator(generator))) ok = false;
  if (device_values != nullptr && !hip_ok(hipFree(device_values))) ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
