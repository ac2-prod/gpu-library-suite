#ifndef GPU_SUITE_SYNTAX_OMP_H
#define GPU_SUITE_SYNTAX_OMP_H
// Declarations only; no OpenMP runtime or offload execution is provided.
extern "C" {
int omp_get_num_devices();
int omp_get_default_device();
int omp_is_initial_device();
}
#endif
