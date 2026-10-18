include(FindPackageHandleStandardArgs)
include(GpuSuiteAmdDiscovery)
gpu_suite_amd_find_library(RocSOLVER rocsolver "rocsolver/rocsolver.h")
find_package(RocBLAS QUIET)
set(RocSOLVER_ROCBLAS_FOUND "${RocBLAS_FOUND}")
find_package_handle_standard_args(RocSOLVER REQUIRED_VARS RocSOLVER_INCLUDE_DIR RocSOLVER_LIBRARY RocSOLVER_HIP_INCLUDE_DIR RocSOLVER_HIP_LIBRARY RocSOLVER_ROCBLAS_FOUND)
if(RocSOLVER_FOUND)
  gpu_suite_amd_import_library(RocSOLVER rocsolver)
  if(NOT TARGET RocSOLVER::rocblas)
    add_library(RocSOLVER::rocblas INTERFACE IMPORTED)
    set_property(TARGET RocSOLVER::rocblas PROPERTY INTERFACE_LINK_LIBRARIES RocBLAS::rocblas)
  endif()
endif()
set(GPU_SUITE_ROCSOLVER_API_PROBE [=[
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
int main() {
  auto volatile symbol_0 = &rocblas_create_handle;
  if (!symbol_0) return 1;
  auto volatile symbol_1 = &rocblas_destroy_handle;
  if (!symbol_1) return 1;
  auto volatile symbol_2 = &rocblas_start_device_memory_size_query;
  if (!symbol_2) return 1;
  auto volatile symbol_3 = &rocblas_stop_device_memory_size_query;
  if (!symbol_3) return 1;
  auto volatile symbol_4 = &rocblas_set_workspace;
  if (!symbol_4) return 1;
  auto volatile symbol_5 = &rocsolver_dgetrf;
  if (!symbol_5) return 1;
  auto volatile symbol_6 = &rocsolver_dgetrs;
  if (!symbol_6) return 1;
  auto volatile symbol_7 = &rocsolver_get_version_string;
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
