include(FindPackageHandleStandardArgs)
include(GpuSuiteAmdDiscovery)
gpu_suite_amd_find_library(RocRAND rocrand "rocrand/rocrand.h")
find_package_handle_standard_args(RocRAND REQUIRED_VARS RocRAND_INCLUDE_DIR RocRAND_LIBRARY RocRAND_HIP_INCLUDE_DIR RocRAND_HIP_LIBRARY)
if(RocRAND_FOUND)
  gpu_suite_amd_import_library(RocRAND rocrand)
endif()
set(GPU_SUITE_ROCRAND_API_PROBE [=[
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
int main() {
  auto volatile symbol_0 = &rocrand_create_generator;
  if (!symbol_0) return 1;
  auto volatile symbol_1 = &rocrand_destroy_generator;
  if (!symbol_1) return 1;
  auto volatile symbol_2 = &rocrand_set_seed;
  if (!symbol_2) return 1;
  auto volatile symbol_3 = &rocrand_set_offset;
  if (!symbol_3) return 1;
  auto volatile symbol_4 = &rocrand_set_ordering;
  if (!symbol_4) return 1;
  auto volatile symbol_5 = &rocrand_initialize_generator;
  if (!symbol_5) return 1;
  auto volatile symbol_6 = &rocrand_generate_uniform_double;
  if (!symbol_6) return 1;
  auto volatile symbol_7 = &rocrand_get_version;
  if (!symbol_7) return 1;
  void *p = nullptr;
  hipDeviceProp_t properties{};
  hipPointerAttribute_t attributes{};
  int version = 0;
  (void)hipGetDeviceProperties(&properties, 0);
  (void)hipRuntimeGetVersion(&version);
  (void)hipDriverGetVersion(&version);
  (void)hipPointerGetAttributes(&attributes, p);
  (void)attributes.type; (void)attributes.device;
  (void)hipMalloc(&p, 64);
  (void)hipDeviceSynchronize();
  (void)hipFree(p);
  return 0;
}
]=])
