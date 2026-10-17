include(FindPackageHandleStandardArgs)
include(GpuSuiteAmdDiscovery)
gpu_suite_amd_find_library(RocSPARSE rocsparse "rocsparse/rocsparse.h")
find_package_handle_standard_args(RocSPARSE REQUIRED_VARS RocSPARSE_INCLUDE_DIR RocSPARSE_LIBRARY RocSPARSE_HIP_INCLUDE_DIR RocSPARSE_HIP_LIBRARY)
if(RocSPARSE_FOUND)
  gpu_suite_amd_import_library(RocSPARSE rocsparse)
endif()
set(GPU_SUITE_ROCSPARSE_API_PROBE [=[
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
int main() {
  auto volatile symbol_0 = &rocsparse_create_handle;
  if (!symbol_0) return 1;
  auto volatile symbol_1 = &rocsparse_destroy_handle;
  if (!symbol_1) return 1;
  auto volatile symbol_2 = &rocsparse_create_csr_descr;
  if (!symbol_2) return 1;
  auto volatile symbol_3 = &rocsparse_create_dnvec_descr;
  if (!symbol_3) return 1;
  auto volatile symbol_4 = &rocsparse_destroy_spmat_descr;
  if (!symbol_4) return 1;
  auto volatile symbol_5 = &rocsparse_destroy_dnvec_descr;
  if (!symbol_5) return 1;
  auto volatile symbol_6 = &rocsparse_spmv;
  if (!symbol_6) return 1;
  auto volatile symbol_7 = &rocsparse_get_version;
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
