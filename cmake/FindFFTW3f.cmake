include(FindPackageHandleStandardArgs)
include(GpuSuiteProbe)

set(_FFTW3f_ROOT_HINTS "")
if(DEFINED FFTW_ROOT AND NOT "${FFTW_ROOT}" STREQUAL "")
  list(APPEND _FFTW3f_ROOT_HINTS "${FFTW_ROOT}")
endif()
if(DEFINED ENV{FFTW_ROOT} AND NOT "$ENV{FFTW_ROOT}" STREQUAL "")
  list(APPEND _FFTW3f_ROOT_HINTS "$ENV{FFTW_ROOT}")
endif()

find_path(FFTW3f_INCLUDE_DIR NAMES fftw3.h
  HINTS ${_FFTW3f_ROOT_HINTS}
  PATH_SUFFIXES include)
find_library(FFTW3f_LIBRARY NAMES fftw3f
  HINTS ${_FFTW3f_ROOT_HINTS}
  PATH_SUFFIXES lib lib64)

set(FFTW3f_BASE_LINKS FALSE)
if(FFTW3f_INCLUDE_DIR AND FFTW3f_LIBRARY)
  set(_fftw3f_probe [=[
#include <fftw3.h>
int main(void) {
  int n[1] = {4};
  fftwf_complex input[4];
  fftwf_complex output[4];
  fftwf_plan plan = fftwf_plan_many_dft(1, n, 1, input, 0, 1, 4,
                                        output, 0, 1, 4,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
  if (plan == 0) return 1;
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  return 0;
}
]=])
  gpu_suite_check_c_link(FFTW3f_BASE_LINKS
    SOURCE "${_fftw3f_probe}"
    INCLUDES "${FFTW3f_INCLUDE_DIR}"
    LIBRARIES "${FFTW3f_LIBRARY}")
endif()

find_package_handle_standard_args(FFTW3f
  REQUIRED_VARS FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_BASE_LINKS
  REASON_FAILURE_MESSAGE
    "fftwf_plan_many_dft and fftwf_execute must compile and link")

if(FFTW3f_FOUND AND NOT TARGET FFTW3f::fftw3f)
  add_library(FFTW3f::fftw3f UNKNOWN IMPORTED)
  set_target_properties(FFTW3f::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW3f_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3f_INCLUDE_DIR}")
endif()

set(FFTW3f_THREADS_FOUND FALSE)
set(FFTW3f_THREADS_DISABLE_REASON "base FFTW3f probe failed")
if(FFTW3f_FOUND)
  find_package(Threads QUIET)
  find_library(FFTW3f_THREADS_LIBRARY NAMES fftw3f_threads
    HINTS ${_FFTW3f_ROOT_HINTS}
    PATH_SUFFIXES lib lib64)
  if(NOT FFTW3f_THREADS_LIBRARY)
    set(FFTW3f_THREADS_DISABLE_REASON "fftw3f_threads library was not found")
  elseif(NOT Threads_FOUND)
    set(FFTW3f_THREADS_DISABLE_REASON "system Threads package was not found")
  else()
    set(_fftw3f_threads_probe [=[
#include <fftw3.h>
int main(void) {
  if (fftwf_init_threads() == 0) return 1;
  fftwf_plan_with_nthreads(2);
  fftwf_cleanup_threads();
  return 0;
}
]=])
    gpu_suite_check_c_link(FFTW3f_THREADS_LINKS
      SOURCE "${_fftw3f_threads_probe}"
      INCLUDES "${FFTW3f_INCLUDE_DIR}"
      LIBRARIES
        "${FFTW3f_THREADS_LIBRARY}"
        "${FFTW3f_LIBRARY}"
        Threads::Threads)
    if(FFTW3f_THREADS_LINKS)
      set(FFTW3f_THREADS_FOUND TRUE)
      set(FFTW3f_THREADS_DISABLE_REASON "")
      if(NOT TARGET FFTW3f::threads)
        add_library(FFTW3f::threads UNKNOWN IMPORTED)
        set_target_properties(FFTW3f::threads PROPERTIES
          IMPORTED_LOCATION "${FFTW3f_THREADS_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${FFTW3f_INCLUDE_DIR}"
          INTERFACE_LINK_LIBRARIES "FFTW3f::fftw3f;Threads::Threads")
      endif()
    else()
      set(FFTW3f_THREADS_DISABLE_REASON
          "fftwf_init_threads, fftwf_plan_with_nthreads, and fftwf_cleanup_threads did not compile and link")
    endif()
  endif()
endif()

mark_as_advanced(FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_THREADS_LIBRARY)
