include(FindPackageHandleStandardArgs)
include(GpuSuiteAmdDiscovery)
gpu_suite_amd_find_library(RocThrust rocthrust "thrust/version.h")
# rocThrust and rocPRIM are header-only; their version headers are provenance
# inputs, never passed to the linker as shared objects.
find_path(RocThrust_ROCPRIM_INCLUDE_DIR NAMES rocprim/rocprim.hpp
  HINTS ${_amd_hints} PATH_SUFFIXES include)
find_package_handle_standard_args(RocThrust REQUIRED_VARS
  RocThrust_INCLUDE_DIR RocThrust_ROCPRIM_INCLUDE_DIR
  RocThrust_HIP_INCLUDE_DIR RocThrust_HIP_LIBRARY)
if(RocThrust_FOUND)
  if(NOT TARGET RocThrust::rocthrust)
    add_library(RocThrust::rocthrust INTERFACE IMPORTED)
    set_target_properties(RocThrust::rocthrust PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${RocThrust_INCLUDE_DIR};${RocThrust_ROCPRIM_INCLUDE_DIR}"
      INTERFACE_COMPILE_DEFINITIONS "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP")
  endif()
  if(NOT TARGET RocThrust::hip)
    add_library(RocThrust::hip UNKNOWN IMPORTED)
    set_target_properties(RocThrust::hip PROPERTIES
      IMPORTED_LOCATION "${RocThrust_HIP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RocThrust_HIP_INCLUDE_DIR}"
      INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1")
  endif()
endif()
set(GPU_SUITE_ROCTHRUST_API_PROBE [=[
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/version.h>
struct Square { __host__ __device__ double operator()(double x) const { return x*x; } };
int main() {
  double *p = nullptr;
  (void)hipMalloc(reinterpret_cast<void**>(&p), sizeof(double));
  auto first = thrust::device_pointer_cast(p);
  double result = thrust::transform_reduce(thrust::hip_rocprim::par,
      first, first + 1, Square{}, 0.0, thrust::plus<double>{});
  (void)hipDeviceSynchronize();
  (void)hipFree(p);
  return result == 0.0 && THRUST_VERSION > 0 ? 0 : 1;
}
]=])
