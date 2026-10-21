// Interoperability validation only; all benchmark computations remain Fortran.
#include <hip/hip_runtime_api.h>
#include <cstdio>

extern "C" int gpu_suite_f_check_mapped_device(const void *pointer, int device) {
  hipPointerAttribute_t attributes{};
  const hipError_t status = hipPointerGetAttributes(&attributes, pointer);
  if (status != hipSuccess) {
    std::fprintf(stderr, "OpenMP mapped pointer query: %s\n", hipGetErrorString(status));
    return 1;
  }
  if (attributes.type != hipMemoryTypeDevice || attributes.device != device) {
    std::fprintf(stderr, "OpenMP mapping is not HIP device memory on requested device %d\n", device);
    return 1;
  }
  return 0;
}
