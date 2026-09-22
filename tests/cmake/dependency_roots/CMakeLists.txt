cmake_minimum_required(VERSION 3.20)

project(gpu_suite_dependency_root_fixture LANGUAGES C)

if(NOT DEFINED GPU_SUITE_ROOT OR "${GPU_SUITE_ROOT}" STREQUAL "")
  message(FATAL_ERROR "GPU_SUITE_ROOT is required")
endif()
if(NOT FIXTURE_PROVIDER MATCHES "^(ONEMKL|FFTW)$")
  message(FATAL_ERROR "FIXTURE_PROVIDER must be ONEMKL or FFTW")
endif()

list(APPEND CMAKE_MODULE_PATH "${GPU_SUITE_ROOT}/cmake")

# Keep the fixture independent of libraries installed on the test host while
# retaining explicitly supplied HINTS and CMAKE_PREFIX_PATH.
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)
set(CMAKE_FIND_USE_PACKAGE_ROOT_PATH FALSE)

if(FIXTURE_PROVIDER STREQUAL "ONEMKL")
  set(GPU_SUITE_CPU_BLAS_BACKEND ONEMKL)
  set(GPU_SUITE_CPU_LAPACK_BACKEND ONEMKL)
  set(GPU_SUITE_CPU_SPARSE_BACKEND ONEMKL)
  include(GpuSuiteCpuDependencies)
  gpu_suite_find_cblas()
  gpu_suite_find_lapacke()
  gpu_suite_find_onemkl_sparse()
  set(_found
      GPU_SUITE_CBLAS_FOUND
      GPU_SUITE_LAPACKE_FOUND
      GPU_SUITE_MKL_SPARSE_FOUND)
else()
  find_package(FFTW3f QUIET)
  set(_found FFTW3f_FOUND FFTW3f_THREADS_FOUND)
endif()

foreach(_variable IN LISTS _found)
  if(EXPECT_FOUND AND NOT ${_variable})
    if(_variable STREQUAL "GPU_SUITE_CBLAS_FOUND")
      set(_diagnostic "${GPU_SUITE_CBLAS_REASON}")
    elseif(_variable STREQUAL "GPU_SUITE_LAPACKE_FOUND")
      set(_diagnostic "${GPU_SUITE_LAPACKE_REASON}")
    elseif(_variable STREQUAL "GPU_SUITE_MKL_SPARSE_FOUND")
      set(_diagnostic "${GPU_SUITE_MKL_SPARSE_REASON}")
    else()
      set(_diagnostic
          "${FFTW3f_NOT_FOUND_MESSAGE}; ${FFTW3f_THREADS_DISABLE_REASON}")
    endif()
    message(FATAL_ERROR
      "${_variable} was not found through the fixture root: ${_diagnostic}")
  elseif(NOT EXPECT_FOUND AND ${_variable})
    message(FATAL_ERROR
      "${_variable} unexpectedly resolved without a fixture root")
  endif()
endforeach()
