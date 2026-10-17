include_guard(GLOBAL)

# Resolve only explicit/environment/CMake hints. No GPU or site discovery.
macro(gpu_suite_amd_find_library package library header)
  set(_amd_hints "${GPU_SUITE_ROCM_ROOT}")
  foreach(_variable ROCM_PATH HIP_PATH)
    if(DEFINED ENV{${_variable}})
      list(APPEND _amd_hints "$ENV{${_variable}}")
    endif()
  endforeach()
  find_path(${package}_INCLUDE_DIR NAMES "${header}"
    HINTS ${_amd_hints} PATH_SUFFIXES include)
  find_library(${package}_LIBRARY NAMES "${library}"
    HINTS ${_amd_hints} PATH_SUFFIXES lib lib64)
  find_path(${package}_HIP_INCLUDE_DIR NAMES hip/hip_runtime.h
    HINTS ${_amd_hints} PATH_SUFFIXES include)
  find_library(${package}_HIP_LIBRARY NAMES amdhip64
    HINTS ${_amd_hints} PATH_SUFFIXES lib lib64)
endmacro()

function(gpu_suite_amd_import_library package library)
  if(NOT TARGET ${package}::${library})
    add_library(${package}::${library} UNKNOWN IMPORTED)
    set_target_properties(${package}::${library} PROPERTIES
      IMPORTED_LOCATION "${${package}_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${${package}_INCLUDE_DIR}")
  endif()
  if(NOT TARGET ${package}::hip)
    add_library(${package}::hip UNKNOWN IMPORTED)
    set_target_properties(${package}::hip PROPERTIES
      IMPORTED_LOCATION "${${package}_HIP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${${package}_HIP_INCLUDE_DIR}"
      INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1")
  endif()
endfunction()
