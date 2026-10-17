include(FindPackageHandleStandardArgs)
include(GpuSuiteProbe)

set(FFTW3f_THREADING "AUTO" CACHE STRING "FFTW threading: AUTO, PTHREAD, OPENMP, COMBINED, NONE")
set_property(CACHE FFTW3f_THREADING PROPERTY STRINGS AUTO PTHREAD OPENMP COMBINED NONE)
if(NOT FFTW3f_THREADING MATCHES "^(AUTO|PTHREAD|OPENMP|COMBINED|NONE)$")
  message(FATAL_ERROR "FFTW3f_THREADING must be AUTO, PTHREAD, OPENMP, COMBINED, or NONE")
endif()
# These are executable link checks even with a static-library parent toolchain.
set(_fftw3f_had_try_compile_type FALSE)
if(DEFINED CMAKE_TRY_COMPILE_TARGET_TYPE)
  set(_fftw3f_had_try_compile_type TRUE)
endif()
set(_fftw3f_saved_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)

set(_FFTW3f_ROOT_HINTS "")
# Accept package-specific roots as well as the historical suite-wide hint.
# Collect CMake variables before environment hints.
foreach(_root FFTW3f_ROOT FFTW3F_ROOT FFTW3_ROOT FFTW_ROOT)
  if(DEFINED ${_root} AND NOT "${${_root}}" STREQUAL "")
    list(APPEND _FFTW3f_ROOT_HINTS "${${_root}}")
  endif()
endforeach()
foreach(_root FFTW3f_ROOT FFTW3F_ROOT FFTW3_ROOT FFTW_ROOT)
  if(DEFINED ENV{${_root}} AND NOT "$ENV{${_root}}" STREQUAL "")
    list(APPEND _FFTW3f_ROOT_HINTS "$ENV{${_root}}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _FFTW3f_ROOT_HINTS)

# Site modules may expose a .pc file without adding a CMake prefix. Metadata is
# only a location hint: the concrete library must still pass the API link test.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_FFTW3f QUIET fftw3f)
endif()

find_path(FFTW3f_INCLUDE_DIR NAMES fftw3.h
  HINTS ${_FFTW3f_ROOT_HINTS} ${PC_FFTW3f_INCLUDE_DIRS}
  PATH_SUFFIXES include)
find_library(FFTW3f_LIBRARY NAMES fftw3f
  HINTS ${_FFTW3f_ROOT_HINTS} ${PC_FFTW3f_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

  set(_fftw3f_threads_probe [=[
#include <fftw3.h>
int main(void) {
  int n[1] = {4};
  fftwf_complex input[4], output[4];
  fftwf_plan plan;
  if (fftwf_init_threads() == 0) return 1;
  fftwf_plan_with_nthreads(2);
  plan = fftwf_plan_many_dft(1, n, 1, input, 0, 1, 4, output, 0, 1, 4,
                            FFTW_FORWARD, FFTW_ESTIMATE);
  if (plan == 0) return 1;
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();
  return 0;
}
]=])

set(FFTW3f_BASE_LINKS FALSE)
set(_fftw3f_base_requires_threads FALSE)
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
  # A static combined build can require pthread symbols even for serial plans.
  # Only a complete integrated API link test may rescue this base discovery.
  if(NOT FFTW3f_BASE_LINKS AND FFTW3f_THREADING MATCHES "^(AUTO|COMBINED)$")
    find_package(Threads QUIET)
    if(Threads_FOUND)
      gpu_suite_check_c_link(FFTW3f_BASE_COMBINED_LINKS
        SOURCE "${_fftw3f_threads_probe}" INCLUDES "${FFTW3f_INCLUDE_DIR}"
        LIBRARIES "${FFTW3f_LIBRARY}" Threads::Threads)
      if(FFTW3f_BASE_COMBINED_LINKS)
        set(FFTW3f_BASE_LINKS TRUE)
        set(_fftw3f_base_requires_threads TRUE)
      endif()
    endif()
  endif()
endif()

set(FFTW3f_DISABLE_REASON "")
if(NOT FFTW3f_INCLUDE_DIR OR NOT FFTW3f_LIBRARY)
  set(_FFTW3f_MISSING "")
  if(NOT FFTW3f_INCLUDE_DIR)
    list(APPEND _FFTW3f_MISSING "fftw3.h")
  endif()
  if(NOT FFTW3f_LIBRARY)
    list(APPEND _FFTW3f_MISSING "fftw3f library")
  endif()
  list(JOIN _FFTW3f_MISSING " and " _FFTW3f_MISSING_TEXT)
  set(FFTW3f_DISABLE_REASON
      "FFTW3f discovery failed: ${_FFTW3f_MISSING_TEXT} not found. Supply FFTW3f_ROOT, FFTW_ROOT, CMAKE_PREFIX_PATH, or fftw3f pkg-config metadata")
elseif(NOT FFTW3f_BASE_LINKS)
  set(FFTW3f_DISABLE_REASON
      "FFTW3f plan/execute compile-and-link probe failed (header=${FFTW3f_INCLUDE_DIR}, library=${FFTW3f_LIBRARY}). See CMake configure diagnostics")
endif()

find_package_handle_standard_args(FFTW3f
  REQUIRED_VARS FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_BASE_LINKS
  REASON_FAILURE_MESSAGE "${FFTW3f_DISABLE_REASON}")

if(FFTW3f_FOUND AND NOT TARGET FFTW3f::fftw3f)
  add_library(FFTW3f::fftw3f UNKNOWN IMPORTED)
  set_target_properties(FFTW3f::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW3f_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3f_INCLUDE_DIR}")
  if(_fftw3f_base_requires_threads)
    set_property(TARGET FFTW3f::fftw3f PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads)
  endif()
endif()

# A separate threading library must belong to the selected base library's
# logical/resolved directory pair. Do not search unrelated system prefixes or
# pkg-config locations after selecting the base provider.
function(_fftw3f_same_provider library result)
  set(${result} FALSE PARENT_SCOPE)
  if(NOT EXISTS "${library}" OR IS_DIRECTORY "${library}")
    return()
  endif()
  get_filename_component(_logical "${library}" DIRECTORY)
  file(REAL_PATH "${library}" _real)
  get_filename_component(_resolved "${_real}" DIRECTORY)
  if(_logical IN_LIST _fftw3f_provider_dirs AND _resolved IN_LIST _fftw3f_provider_dirs)
    set(${result} TRUE PARENT_SCOPE)
  endif()
endfunction()

set(FFTW3f_THREADS_FOUND FALSE)
set(FFTW3f_THREAD_BACKEND "none")
set(FFTW3f_THREAD_COMPILE_OPTIONS "")
set(FFTW3f_THREAD_LINK_OPTIONS "")
set(FFTW3f_THREAD_RUNTIME_LIBRARIES "")
set(FFTW3f_THREAD_PTHREAD_RUNTIME_LIBRARIES "")
set(FFTW3f_THREAD_DEPENDENCY_NAMES "")
set(FFTW3f_THREADS_DISABLE_REASON "base FFTW3f unavailable: ${FFTW3f_DISABLE_REASON}")
if(FFTW3f_FOUND)
  set(FFTW3f_THREAD_DEPENDENCY_NAMES fftw3f)
  get_filename_component(_fftw3f_logical_dir "${FFTW3f_LIBRARY}" DIRECTORY)
  file(REAL_PATH "${FFTW3f_LIBRARY}" _fftw3f_real_library)
  get_filename_component(_fftw3f_resolved_dir "${_fftw3f_real_library}" DIRECTORY)
  set(_fftw3f_provider_dirs "${_fftw3f_logical_dir}" "${_fftw3f_resolved_dir}")
  list(REMOVE_DUPLICATES _fftw3f_provider_dirs)
  if(_fftw3f_base_requires_threads)
    # The base already proved integrated threading; do not combine it with a
    # second, possibly different threading runtime selected by a separate DSO.
    set(_fftw3f_candidates COMBINED)
  elseif(FFTW3f_THREADING STREQUAL AUTO)
    set(_fftw3f_candidates PTHREAD OPENMP COMBINED)
  elseif(FFTW3f_THREADING STREQUAL NONE)
    set(_fftw3f_candidates "")
    set(FFTW3f_THREADS_DISABLE_REASON "disabled by FFTW3f_THREADING=NONE")
  else()
    set(_fftw3f_candidates "${FFTW3f_THREADING}")
  endif()
  set(_fftw3f_failures "")
  foreach(_candidate IN LISTS _fftw3f_candidates)
    set(_candidate_libraries "")
    set(_candidate_compile_options "")
    set(_candidate_link_options "")
    set(_candidate_runtime_libraries "")
    set(_candidate_pthread_runtime_libraries "")
    set(_candidate_dependency_names fftw3f)
    set(_candidate_reason "")
    if(_candidate STREQUAL PTHREAD OR _candidate STREQUAL COMBINED)
      find_package(Threads QUIET)
      if(NOT Threads_FOUND)
        set(_candidate_reason "system Threads package was not found")
      else()
        if(CMAKE_THREAD_LIBS_INIT)
          separate_arguments(_candidate_link_options NATIVE_COMMAND "${CMAKE_THREAD_LIBS_INIT}")
          if("-pthread" IN_LIST _candidate_link_options)
            set(_candidate_compile_options -pthread)
          endif()
        endif()
        get_target_property(_thread_link_inputs Threads::Threads INTERFACE_LINK_LIBRARIES)
        if(NOT _thread_link_inputs)
          set(_thread_link_inputs "")
        endif()
        list(APPEND _thread_link_inputs ${_candidate_link_options})
        foreach(_runtime IN LISTS _thread_link_inputs)
          if(IS_ABSOLUTE "${_runtime}")
            if(NOT EXISTS "${_runtime}" OR IS_DIRECTORY "${_runtime}")
              set(_candidate_reason "explicit Threads runtime library does not exist")
            else()
              list(APPEND _candidate_pthread_runtime_libraries "${_runtime}")
            endif()
          endif()
        endforeach()
        list(REMOVE_DUPLICATES _candidate_pthread_runtime_libraries)
        set(_runtime_index 0)
        foreach(_runtime IN LISTS _candidate_pthread_runtime_libraries)
          list(APPEND _candidate_dependency_names "fftw_pthread_runtime_${_runtime_index}")
          math(EXPR _runtime_index "${_runtime_index} + 1")
        endforeach()
        set(_candidate_libraries "${FFTW3f_LIBRARY}" Threads::Threads)
        if(_candidate STREQUAL PTHREAD)
          find_library(FFTW3f_THREADS_LIBRARY NAMES fftw3f_threads
            PATHS ${_fftw3f_provider_dirs} NO_DEFAULT_PATH)
          _fftw3f_same_provider("${FFTW3f_THREADS_LIBRARY}" _same_provider)
          if(NOT _same_provider)
            set(_candidate_reason "fftw3f_threads missing or outside the selected base provider directories")
          else()
            list(PREPEND _candidate_libraries "${FFTW3f_THREADS_LIBRARY}")
            list(APPEND _candidate_dependency_names fftw3f_threads)
          endif()
        endif()
      endif()
    else()
      find_library(FFTW3f_OPENMP_LIBRARY NAMES fftw3f_omp
        PATHS ${_fftw3f_provider_dirs} NO_DEFAULT_PATH)
      _fftw3f_same_provider("${FFTW3f_OPENMP_LIBRARY}" _same_provider)
      if(NOT _same_provider)
        set(_candidate_reason "fftw3f_omp missing or outside the selected base provider directories")
      else()
        find_package(OpenMP QUIET COMPONENTS C)
        if(NOT OpenMP_C_FOUND OR NOT TARGET OpenMP::OpenMP_C)
          set(_candidate_reason "compiler OpenMP C support was not found")
        else()
          separate_arguments(_candidate_compile_options NATIVE_COMMAND "${OpenMP_C_FLAGS}")
          get_target_property(_candidate_link_options OpenMP::OpenMP_C INTERFACE_LINK_OPTIONS)
          if(NOT _candidate_link_options)
            set(_candidate_link_options "")
          endif()
          foreach(_runtime IN LISTS OpenMP_C_LIBRARIES)
            if(NOT IS_ABSOLUTE "${_runtime}" OR NOT EXISTS "${_runtime}" OR IS_DIRECTORY "${_runtime}")
              set(_candidate_reason "OpenMP runtime must resolve to concrete existing library files")
            else()
              list(APPEND _candidate_runtime_libraries "${_runtime}")
            endif()
          endforeach()
          list(REMOVE_DUPLICATES _candidate_runtime_libraries)
          if(NOT _candidate_runtime_libraries AND NOT _candidate_compile_options AND NOT _candidate_link_options)
            set(_candidate_reason "compiler OpenMP runtime/link evidence is missing")
          endif()
          set(_runtime_index 0)
          foreach(_runtime IN LISTS _candidate_runtime_libraries)
            list(APPEND _candidate_dependency_names "fftw_openmp_runtime_${_runtime_index}")
            math(EXPR _runtime_index "${_runtime_index} + 1")
          endforeach()
          list(APPEND _candidate_dependency_names fftw3f_omp)
          set(_candidate_libraries "${FFTW3f_OPENMP_LIBRARY}" "${FFTW3f_LIBRARY}" OpenMP::OpenMP_C)
        endif()
      endif()
    endif()
    if(NOT _candidate_reason)
      gpu_suite_check_c_link(FFTW3f_${_candidate}_LINKS
        SOURCE "${_fftw3f_threads_probe}" INCLUDES "${FFTW3f_INCLUDE_DIR}"
        LIBRARIES ${_candidate_libraries})
      if(NOT FFTW3f_${_candidate}_LINKS)
        set(_candidate_reason "FFTW plan/execute and all three thread APIs did not compile and link")
      endif()
    endif()
    if(_candidate_reason)
      list(APPEND _fftw3f_failures "${_candidate}: ${_candidate_reason}")
    else()
      set(FFTW3f_THREADS_FOUND TRUE)
      string(TOLOWER "${_candidate}" FFTW3f_THREAD_BACKEND)
      set(FFTW3f_THREAD_COMPILE_OPTIONS "${_candidate_compile_options}")
      set(FFTW3f_THREAD_LINK_OPTIONS "${_candidate_link_options}")
      set(FFTW3f_THREAD_RUNTIME_LIBRARIES "${_candidate_runtime_libraries}")
      set(FFTW3f_THREAD_PTHREAD_RUNTIME_LIBRARIES "${_candidate_pthread_runtime_libraries}")
      set(FFTW3f_THREAD_DEPENDENCY_NAMES "${_candidate_dependency_names}")
      set(FFTW3f_THREADS_DISABLE_REASON "")
      if(NOT TARGET FFTW3f::threads)
        add_library(FFTW3f::threads INTERFACE IMPORTED)
      endif()
      set_target_properties(FFTW3f::threads PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3f_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_candidate_libraries}")
      break()
    endif()
  endforeach()
  if(NOT FFTW3f_THREADS_FOUND AND _fftw3f_failures)
    list(JOIN _fftw3f_failures "; " FFTW3f_THREADS_DISABLE_REASON)
  endif()
endif()
if(_fftw3f_had_try_compile_type)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_fftw3f_saved_try_compile_type}")
else()
  unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
endif()
mark_as_advanced(FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_THREADS_LIBRARY FFTW3f_OPENMP_LIBRARY)
