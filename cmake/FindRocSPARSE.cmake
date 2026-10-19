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
  auto volatile symbol_6 = &rocsparse_v2_spmv;
  if (!symbol_6) return 1;
  auto volatile symbol_7 = &rocsparse_get_version;
  if (!symbol_7) return 1;
  auto volatile symbol_8 = &rocsparse_create_spmv_descr;
  if (!symbol_8) return 1;
  auto volatile symbol_9 = &rocsparse_destroy_spmv_descr;
  if (!symbol_9) return 1;
  auto volatile symbol_10 = &rocsparse_spmv_set_input;
  if (!symbol_10) return 1;
  auto volatile symbol_11 = &rocsparse_v2_spmv_buffer_size;
  if (!symbol_11) return 1;
  rocsparse_spmv_descr spmv = nullptr;
  const auto alg = rocsparse_spmv_alg_csr_rowsplit;
  const auto op = rocsparse_operation_none;
  const auto datatype = rocsparse_datatype_f64_r;
  (void)rocsparse_spmv_set_input(nullptr, spmv, rocsparse_spmv_input_alg,
      &alg, sizeof(alg), nullptr);
  (void)rocsparse_spmv_set_input(nullptr, spmv, rocsparse_spmv_input_operation,
      &op, sizeof(op), nullptr);
  (void)rocsparse_spmv_set_input(nullptr, spmv, rocsparse_spmv_input_scalar_datatype,
      &datatype, sizeof(datatype), nullptr);
  (void)rocsparse_spmv_set_input(nullptr, spmv, rocsparse_spmv_input_compute_datatype,
      &datatype, sizeof(datatype), nullptr);
  size_t bytes = 0;
  (void)rocsparse_v2_spmv_buffer_size(nullptr, spmv, nullptr, nullptr, nullptr,
      rocsparse_v2_spmv_stage_analysis, &bytes, nullptr);
  (void)rocsparse_v2_spmv(nullptr, spmv, nullptr, nullptr, nullptr, nullptr, nullptr,
      rocsparse_v2_spmv_stage_compute, bytes, nullptr, nullptr);
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
