#ifndef GPU_SUITE_TEST_SOLVER_OMP_H
#define GPU_SUITE_TEST_SOLVER_OMP_H
#include <hip/hip_runtime.h>
inline int omp_get_default_device() {
  const char *value = std::getenv("OMP_DEFAULT_DEVICE");
  return value ? std::atoi(value) : 0;
}
inline int omp_get_num_devices() { return 2; }
inline int omp_is_initial_device() { return fixture_mode("host-fallback") ? 1 : 0; }
#endif
