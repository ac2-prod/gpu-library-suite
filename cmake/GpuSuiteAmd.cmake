include(GpuSuiteProbe)
include(GpuSuiteAmdDiscovery)
# These are API link checks even when a parent toolchain defaults try_compile
# to static libraries. They never execute a produced binary.
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
set(GPU_SUITE_ROCM_ROOT "" CACHE PATH "Optional externally selected ROCm prefix")
set(GPU_SUITE_AMD_GPU_TARGET "" CACHE STRING "Explicit AMD OpenMP GPU target")
set(GPU_SUITE_HIP_TARGETS_ENABLED FALSE)
set(GPU_SUITE_HIP_DISABLE_REASON "not requested")
set(GPU_SUITE_OPENMP_TARGETS_ENABLED FALSE)
set(GPU_SUITE_OPENMP_DISABLE_REASON "not requested")
set(GPU_SUITE_OPENMP_COMPILE_OPTIONS "")
set(GPU_SUITE_OPENMP_LINK_OPTIONS "")

# CPU dependency resolution never requires ROCm or a GPU compiler.
if(GPU_SUITE_BUILD_CPU AND "rocfft" IN_LIST GPU_SUITE_AMD_LIBRARIES)
  find_package(FFTW3f QUIET)
endif()
if((GPU_SUITE_BUILD_HIP OR GPU_SUITE_BUILD_OPENMP) AND "rocfft" IN_LIST GPU_SUITE_AMD_LIBRARIES)
  find_package(RocFFT QUIET)
endif()

# Keep the mapping-only variables in their own scope: the shared API probe has
# void* input/output buffers for rocfft_execute. Build this exact source even
# when a compiler is unavailable so host-only regression tests can compile it.
if(GPU_SUITE_BUILD_OPENMP AND RocFFT_FOUND AND "rocfft" IN_LIST GPU_SUITE_AMD_LIBRARIES)
  set(GPU_SUITE_ROCFFT_OPENMP_API_PROBE "${GPU_SUITE_ROCFFT_API_PROBE}")
  string(REPLACE "int main() {" [=[
#include <omp.h>
int main() {
  {
    int on_host = 1;
    float *input = new float[8], *output = new float[8];
    #pragma omp target map(from:on_host)
    { on_host = omp_is_initial_device(); }
    #pragma omp target data map(to:input[0:8]) map(from:output[0:8])
    {
      #pragma omp target data use_device_ptr(input, output)
      { void *in = input; void *out = output; (void)in; (void)out; }
    }
    delete[] input; delete[] output;
  }
]=] GPU_SUITE_ROCFFT_OPENMP_API_PROBE "${GPU_SUITE_ROCFFT_OPENMP_API_PROBE}")
endif()

if(GPU_SUITE_BUILD_HIP AND "rocfft" IN_LIST GPU_SUITE_AMD_LIBRARIES)
  if(CMAKE_HIP_COMPILER MATCHES "(^|/)hipcc([^/]*)$")
    message(FATAL_ERROR "Native CMake HIP requires Clang, not the hipcc wrapper")
  endif()
  if(CMAKE_VERSION VERSION_LESS 3.21)
    set(GPU_SUITE_HIP_DISABLE_REASON "native HIP requires CMake 3.21 or later")
  elseif(NOT CMAKE_HIP_ARCHITECTURES OR
         "native" IN_LIST CMAKE_HIP_ARCHITECTURES)
    set(GPU_SUITE_HIP_DISABLE_REASON "supply explicit CMAKE_HIP_ARCHITECTURES, native GPU discovery is forbidden")
  elseif(NOT RocFFT_FOUND)
    set(GPU_SUITE_HIP_DISABLE_REASON "${RocFFT_DISABLE_REASON}")
  else()
    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
      set(GPU_SUITE_HIP_DISABLE_REASON "no native HIP Clang compiler was found")
    else()
      enable_language(HIP)
      set(CMAKE_HIP_STANDARD 17)
      set(CMAKE_HIP_STANDARD_REQUIRED ON)
      set(CMAKE_HIP_EXTENSIONS OFF)
      if(NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
        set(GPU_SUITE_HIP_DISABLE_REASON "native HIP requires a supported Clang compiler")
      else()
        gpu_suite_check_hip_link(GPU_SUITE_ROCFFT_HIP_LINKS
          SOURCE "${GPU_SUITE_ROCFFT_API_PROBE}"
          LIBRARIES RocFFT::rocfft RocFFT::hip)
        if(GPU_SUITE_ROCFFT_HIP_LINKS)
          set(GPU_SUITE_HIP_TARGETS_ENABLED TRUE)
          set(GPU_SUITE_HIP_DISABLE_REASON "")
        else()
          set(GPU_SUITE_HIP_DISABLE_REASON "HIP Runtime/rocFFT API or C++ runtime compile-and-link probe failed")
        endif()
      endif()
    endif()
  endif()
endif()

if(GPU_SUITE_BUILD_OPENMP AND "rocfft" IN_LIST GPU_SUITE_AMD_LIBRARIES)
  if(NOT GPU_SUITE_AMD_GPU_TARGET OR
     NOT GPU_SUITE_AMD_GPU_TARGET MATCHES "^gfx[0-9a-z]+(:[a-z0-9]+[+-])*$")
    set(GPU_SUITE_OPENMP_DISABLE_REASON "supply an explicit GPU_SUITE_AMD_GPU_TARGET, native discovery is forbidden")
  elseif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(GPU_SUITE_OPENMP_DISABLE_REASON "AMD OpenMP target requires AMD Clang C++")
  elseif(NOT RocFFT_FOUND)
    set(GPU_SUITE_OPENMP_DISABLE_REASON "${RocFFT_DISABLE_REASON}")
  else()
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --version
      OUTPUT_VARIABLE _amd_compiler_version ERROR_VARIABLE _amd_compiler_error
      RESULT_VARIABLE _amd_compiler_status)
    if(NOT _amd_compiler_status EQUAL 0 OR
       NOT _amd_compiler_version MATCHES "AMD clang")
      set(GPU_SUITE_OPENMP_DISABLE_REASON "CMAKE_CXX_COMPILER must identify an AMD Clang toolchain")
    else()
      set(GPU_SUITE_OPENMP_COMPILE_OPTIONS "-fopenmp;--offload-arch=${GPU_SUITE_AMD_GPU_TARGET}")
      set(GPU_SUITE_OPENMP_LINK_OPTIONS "${GPU_SUITE_OPENMP_COMPILE_OPTIONS}")
      gpu_suite_check_cxx_link(GPU_SUITE_ROCFFT_OPENMP_LINKS
        SOURCE "${GPU_SUITE_ROCFFT_OPENMP_API_PROBE}" LIBRARIES RocFFT::rocfft RocFFT::hip
        COMPILE_OPTIONS ${GPU_SUITE_OPENMP_COMPILE_OPTIONS}
        LINK_OPTIONS ${GPU_SUITE_OPENMP_LINK_OPTIONS})
      if(GPU_SUITE_ROCFFT_OPENMP_LINKS)
        set(GPU_SUITE_OPENMP_TARGETS_ENABLED TRUE)
        set(GPU_SUITE_OPENMP_DISABLE_REASON "")
      else()
        set(GPU_SUITE_OPENMP_DISABLE_REASON "OpenMP target/HIP Runtime/rocFFT API compile-and-link probe failed")
      endif()
    endif()
  endif()
endif()

include(GpuSuiteAmdFamilies)
