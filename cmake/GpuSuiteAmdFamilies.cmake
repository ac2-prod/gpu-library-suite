# Additional AMD libraries are independently optional. rocFFT's existing
# discovery and acceptance checks remain in GpuSuiteAmd.cmake.
include(GpuSuiteCpuDependencies)
if(GPU_SUITE_BUILD_CPU)
  if("rocblas" IN_LIST GPU_SUITE_AMD_LIBRARIES)
    gpu_suite_find_cblas()
  endif()
  if("rocsolver" IN_LIST GPU_SUITE_AMD_LIBRARIES)
    gpu_suite_find_lapacke()
  endif()
  if("rocsparse" IN_LIST GPU_SUITE_AMD_LIBRARIES)
    gpu_suite_find_onemkl_sparse()
    gpu_suite_find_aocl_sparse()
  endif()
  if("rocthrust" IN_LIST GPU_SUITE_AMD_LIBRARIES AND GPU_SUITE_CPU_THRUST_BACKEND STREQUAL "OPENMP")
    find_package(OpenMP QUIET COMPONENTS CXX)
  endif()
endif()

set(_amd_packages "")
foreach(_package RocBLAS RocSPARSE RocSOLVER RocRAND RocThrust)
  string(TOLOWER "${_package}" _library)
  if(_library IN_LIST GPU_SUITE_AMD_LIBRARIES)
    list(APPEND _amd_packages "${_package}")
  endif()
endforeach()
if(GPU_SUITE_BUILD_HIP OR GPU_SUITE_BUILD_OPENMP)
  foreach(_package IN LISTS _amd_packages)
    find_package(${_package} QUIET)
  endforeach()
endif()

# OpenMP rocThrust needs a native HIP translation unit as an explicit link
# input. It uses the same externally selected architecture, never discovery.
set(_amd_need_hip FALSE)
if(GPU_SUITE_BUILD_HIP)
  foreach(_package IN LISTS _amd_packages)
    if(${_package}_FOUND)
      set(_amd_need_hip TRUE)
    endif()
  endforeach()
elseif(GPU_SUITE_BUILD_OPENMP AND RocThrust_FOUND)
  set(_amd_need_hip TRUE)
  if(NOT CMAKE_HIP_ARCHITECTURES AND GPU_SUITE_AMD_GPU_TARGET)
    set(CMAKE_HIP_ARCHITECTURES "${GPU_SUITE_AMD_GPU_TARGET}")
  elseif(CMAKE_HIP_ARCHITECTURES AND GPU_SUITE_AMD_GPU_TARGET AND
         NOT CMAKE_HIP_ARCHITECTURES STREQUAL GPU_SUITE_AMD_GPU_TARGET)
    message(FATAL_ERROR "OpenMP rocThrust requires CMAKE_HIP_ARCHITECTURES to equal GPU_SUITE_AMD_GPU_TARGET for its HIP wrapper")
  endif()
endif()
if(_amd_need_hip AND NOT CMAKE_HIP_COMPILER_LOADED AND
   NOT CMAKE_VERSION VERSION_LESS 3.21 AND CMAKE_HIP_ARCHITECTURES AND
   NOT "native" IN_LIST CMAKE_HIP_ARCHITECTURES)
  if(CMAKE_HIP_COMPILER MATCHES "(^|/)hipcc([^/]*)$")
    message(FATAL_ERROR "Native CMake HIP requires Clang, not the hipcc wrapper")
  endif()
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
  endif()
endif()

set(_amd_openmp_toolchain FALSE)
if(GPU_SUITE_BUILD_OPENMP AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   GPU_SUITE_AMD_GPU_TARGET MATCHES "^gfx[0-9a-z]+(:[a-z0-9]+[+-])*$")
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --version
    RESULT_VARIABLE _amd_version_status OUTPUT_VARIABLE _amd_version
    ERROR_VARIABLE _amd_version_error)
  if(_amd_version_status EQUAL 0 AND _amd_version MATCHES "AMD clang")
    set(_amd_openmp_toolchain TRUE)
    set(GPU_SUITE_OPENMP_COMPILE_OPTIONS "-fopenmp;--offload-arch=${GPU_SUITE_AMD_GPU_TARGET}")
    set(GPU_SUITE_OPENMP_LINK_OPTIONS "${GPU_SUITE_OPENMP_COMPILE_OPTIONS}")
  endif()
endif()

foreach(_package IN LISTS _amd_packages)
  string(TOLOWER "${_package}" _library)
  string(TOUPPER "${_package}" _macro)
  foreach(_profile HIP OPENMP)
    set(GPU_SUITE_${_macro}_${_profile}_ENABLED FALSE)
    set(GPU_SUITE_${_macro}_${_profile}_DISABLE_REASON "not requested")
    if(GPU_SUITE_BUILD_${_profile})
      set(GPU_SUITE_${_macro}_${_profile}_DISABLE_REASON "${_package} or HIP Runtime dependency missing")
      if(${_package}_FOUND)
        set(_libraries ${_package}::${_library} ${_package}::hip)
        if(_package STREQUAL RocSOLVER)
          list(APPEND _libraries RocSOLVER::rocblas)
        endif()
        if(_profile STREQUAL HIP)
          set(GPU_SUITE_${_macro}_${_profile}_DISABLE_REASON "explicit architecture and native HIP Clang required")
          if(CMAKE_HIP_COMPILER_LOADED AND CMAKE_HIP_COMPILER_ID STREQUAL "Clang" AND
             CMAKE_HIP_ARCHITECTURES AND NOT "native" IN_LIST CMAKE_HIP_ARCHITECTURES)
            gpu_suite_check_hip_link(GPU_SUITE_${_macro}_HIP_LINKS
              SOURCE "${GPU_SUITE_${_macro}_API_PROBE}" LIBRARIES ${_libraries})
            set(GPU_SUITE_${_macro}_HIP_ENABLED "${GPU_SUITE_${_macro}_HIP_LINKS}")
            set(GPU_SUITE_${_macro}_HIP_DISABLE_REASON "${_package}/HIP compile-and-link API probe failed")
          endif()
        else()
          set(GPU_SUITE_${_macro}_OPENMP_DISABLE_REASON "explicit architecture and AMD Clang OpenMP required")
          if(_amd_openmp_toolchain)
            set(_probe "${GPU_SUITE_${_macro}_API_PROBE}")
            if(_package STREQUAL RocThrust)
              # Reference the actual wrapper ABI so a missing HIP object or an
              # incompatible mixed link fails before enabling these targets.
              set(_probe [=[
#include <hip/hip_runtime.h>
#include <cstddef>
extern "C" int rocthrust_square_sum(const double *, std::size_t, double *) noexcept;
int main() {
  double *values = nullptr, result = 0.0;
  (void)hipMalloc(reinterpret_cast<void **>(&values), sizeof(double));
  int status = rocthrust_square_sum(values, 1, &result);
  (void)hipFree(values);
  return status;
}
]=])
            endif()
            string(REPLACE "int main() {" "#include <omp.h>\nint main() {\n { double *p = new double[1]; int host = 1;\n #pragma omp target map(from:host)\n { host = omp_is_initial_device(); }\n #pragma omp target data map(tofrom:p[0:1])\n {\n #pragma omp target data use_device_ptr(p)\n { (void)p; }\n }\n delete[] p; }\n" _probe "${_probe}")
            if(_package STREQUAL RocThrust)
              gpu_suite_check_rocthrust_openmp_link(GPU_SUITE_ROCTHRUST_OPENMP_LINKS
                SOURCE "${_probe}"
                WRAPPER "${CMAKE_SOURCE_DIR}/amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"
                COMPILE_OPTIONS ${GPU_SUITE_OPENMP_COMPILE_OPTIONS}
                LINK_OPTIONS ${GPU_SUITE_OPENMP_LINK_OPTIONS})
            else()
              gpu_suite_check_cxx_link(GPU_SUITE_${_macro}_OPENMP_LINKS
                SOURCE "${_probe}" LIBRARIES ${_libraries}
                COMPILE_OPTIONS ${GPU_SUITE_OPENMP_COMPILE_OPTIONS}
                LINK_OPTIONS ${GPU_SUITE_OPENMP_LINK_OPTIONS})
            endif()
            if(GPU_SUITE_${_macro}_OPENMP_LINKS)
              set(GPU_SUITE_${_macro}_OPENMP_ENABLED TRUE)
            endif()
            set(GPU_SUITE_${_macro}_OPENMP_DISABLE_REASON "${_package}/OpenMP/HIP interoperation API probe failed")
          endif()
        endif()
        if(GPU_SUITE_${_macro}_${_profile}_ENABLED)
          set(GPU_SUITE_${_macro}_${_profile}_DISABLE_REASON "")
        endif()
      endif()
    endif()
  endforeach()
endforeach()
