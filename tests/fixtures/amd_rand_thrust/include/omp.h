#ifndef GPU_SUITE_FAKE_RAND_THRUST_OMP_H
#define GPU_SUITE_FAKE_RAND_THRUST_OMP_H
#include <cstdlib>
// Used only with ignored target pragmas to test host-fallback rejection.
inline int omp_get_num_devices() { return 1; }
inline int omp_get_default_device() { return 0; }
inline int omp_is_initial_device() { return 1; }
#endif
