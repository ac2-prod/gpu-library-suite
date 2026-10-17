include(FindPackageHandleStandardArgs)
include(GpuSuiteAmdDiscovery)
gpu_suite_amd_find_library(RocBLAS rocblas "rocblas/rocblas.h")
find_package_handle_standard_args(RocBLAS REQUIRED_VARS RocBLAS_INCLUDE_DIR RocBLAS_LIBRARY RocBLAS_HIP_INCLUDE_DIR RocBLAS_HIP_LIBRARY)
if(RocBLAS_FOUND)
  gpu_suite_amd_import_library(RocBLAS rocblas)
endif()
set(GPU_SUITE_ROCBLAS_API_PROBE [=[
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
int main() {
  auto volatile symbol_0 = &rocblas_create_handle;
  if (!symbol_0) return 1;
  auto volatile symbol_1 = &rocblas_destroy_handle;
  if (!symbol_1) return 1;
  auto volatile symbol_2 = &rocblas_dgemm;
  if (!symbol_2) return 1;
  auto volatile symbol_3 = &rocblas_get_version_string;
  if (!symbol_3) return 1;
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
