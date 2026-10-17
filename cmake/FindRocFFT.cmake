include(FindPackageHandleStandardArgs)
# Resolve concrete library files for both linking and provenance. Do not invent
# a site prefix: explicit hints and CMake's usual search paths remain usable.
set(_rocm_hints "${GPU_SUITE_ROCM_ROOT}")
foreach(_variable ROCM_PATH HIP_PATH)
  if(DEFINED ENV{${_variable}} AND NOT "$ENV{${_variable}}" STREQUAL "")
    list(APPEND _rocm_hints "$ENV{${_variable}}")
  endif()
endforeach()
find_path(RocFFT_INCLUDE_DIR NAMES rocfft/rocfft.h
  HINTS ${_rocm_hints} PATH_SUFFIXES include)
find_path(RocFFT_HIP_INCLUDE_DIR NAMES hip/hip_runtime_api.h
  HINTS ${_rocm_hints} PATH_SUFFIXES include)
find_library(RocFFT_LIBRARY NAMES rocfft HINTS ${_rocm_hints} PATH_SUFFIXES lib lib64)
find_library(RocFFT_HIP_LIBRARY NAMES amdhip64 HINTS ${_rocm_hints} PATH_SUFFIXES lib lib64)
set(RocFFT_DISABLE_REASON "rocFFT and HIP Runtime headers/libraries were not found")
find_package_handle_standard_args(RocFFT REQUIRED_VARS RocFFT_INCLUDE_DIR
  RocFFT_HIP_INCLUDE_DIR RocFFT_LIBRARY RocFFT_HIP_LIBRARY)
if(RocFFT_FOUND)
  if(NOT TARGET RocFFT::rocfft)
    add_library(RocFFT::rocfft UNKNOWN IMPORTED)
    set_target_properties(RocFFT::rocfft PROPERTIES
      IMPORTED_LOCATION "${RocFFT_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RocFFT_INCLUDE_DIR}")
  endif()
  if(NOT TARGET RocFFT::hip)
    add_library(RocFFT::hip UNKNOWN IMPORTED)
    set_target_properties(RocFFT::hip PROPERTIES
      IMPORTED_LOCATION "${RocFFT_HIP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RocFFT_HIP_INCLUDE_DIR}"
      INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1")
  endif()
endif()
# This is compiled and linked, never run at configure time. In particular it
# verifies the workspace API and C++ allocation runtime used by the measured
# implementations. The host buffer escapes to HIP, so an optimizer cannot
# discard the new[]/delete[] pair and hide a C++ driver/link configuration error.
set(GPU_SUITE_ROCFFT_API_PROBE [=[
#include <new>
#include <hip/hip_runtime_api.h>
#include <rocfft/rocfft.h>
int main() {
  float *host_buffer = new (std::nothrow) float[16];
  if (!host_buffer) return 1;
  rocfft_plan plan = nullptr;
  rocfft_execution_info info = nullptr;
  size_t length = 4, work_size = 0;
  void *input = nullptr, *output = nullptr, *work = nullptr;
  hipPointerAttribute_t attributes;
  hipDeviceProp_t properties;
  int runtime_version = 0, driver_version = 0;
  char version[128];
  (void)hipGetDeviceProperties(&properties, 0);
  (void)hipRuntimeGetVersion(&runtime_version);
  (void)hipDriverGetVersion(&driver_version);
  (void)hipPointerGetAttributes(&attributes, input);
  (void)attributes.type; (void)attributes.device;
  (void)rocfft_get_version_string(version, sizeof(version));
  (void)rocfft_setup();
  (void)rocfft_plan_create(&plan, rocfft_placement_notinplace,
    rocfft_transform_type_complex_forward, rocfft_precision_single,
    1, &length, 2, nullptr);
  (void)rocfft_execution_info_create(&info);
  (void)rocfft_plan_get_work_buffer_size(plan, &work_size);
  (void)hipMalloc(&work, work_size);
  (void)rocfft_execution_info_set_work_buffer(info, work, work_size);
  (void)hipMalloc(&input, 64); (void)hipMalloc(&output, 64);
  (void)hipMemcpy(input, host_buffer, 64, hipMemcpyHostToDevice);
  (void)rocfft_execute(plan, &input, &output, info);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(output, input, 64, hipMemcpyDeviceToHost);
  (void)hipFree(input); (void)hipFree(output); (void)hipFree(work);
  (void)rocfft_execution_info_destroy(info);
  (void)rocfft_plan_destroy(plan); (void)rocfft_cleanup();
  delete[] host_buffer;
  return 0;
}
]=])
mark_as_advanced(RocFFT_INCLUDE_DIR RocFFT_HIP_INCLUDE_DIR RocFFT_LIBRARY RocFFT_HIP_LIBRARY)
