function(_gpu_suite_escape_c_string input output)
  string(REPLACE "\\" "\\\\" _value "${input}")
  string(REPLACE "\"" "\\\"" _value "${_value}")
  string(REPLACE "\n" "\\n" _value "${_value}")
  set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(_gpu_suite_escape_json_string input output)
  string(REPLACE "\\" "\\\\" _value "${input}")
  string(REPLACE "\"" "\\\"" _value "${_value}")
  string(REPLACE "\r" "\\r" _value "${_value}")
  string(REPLACE "\n" "\\n" _value "${_value}")
  string(REPLACE "\t" "\\t" _value "${_value}")
  set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(gpu_suite_generate_build_metadata)
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE _git_result
    OUTPUT_VARIABLE _git_commit
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _git_result EQUAL 0 OR "${_git_commit}" STREQUAL "")
    set(_git_commit "unknown")
  endif()

  execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE _status_result
    OUTPUT_VARIABLE _git_status
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _status_result EQUAL 0)
    set(_git_dirty "unknown")
  elseif("${_git_status}" STREQUAL "")
    set(_git_dirty "false")
  else()
    set(_git_dirty "true")
  endif()

  if(CMAKE_BUILD_TYPE)
    set(_build_type "${CMAKE_BUILD_TYPE}")
  else()
    set(_build_type "unknown")
  endif()

  if(GPU_SUITE_BUILD_OPENACC)
    set(_build_profile "openacc")
  else()
    set(_build_profile "cpu-cuda")
  endif()

  string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
  set(_c_config_flags_variable "CMAKE_C_FLAGS_${_build_type_upper}")
  set(_cxx_config_flags_variable "CMAKE_CXX_FLAGS_${_build_type_upper}")
  set(_cuda_config_flags_variable "CMAKE_CUDA_FLAGS_${_build_type_upper}")
  set(_link_config_flags_variable
      "CMAKE_EXE_LINKER_FLAGS_${_build_type_upper}")
  set(_c_flags "${CMAKE_C_FLAGS} ${${_c_config_flags_variable}}")
  set(_cxx_flags "${CMAKE_CXX_FLAGS} ${${_cxx_config_flags_variable}}")
  set(_cuda_flags "${CMAKE_CUDA_FLAGS} ${${_cuda_config_flags_variable}}")
  set(_link_flags
      "${CMAKE_EXE_LINKER_FLAGS} ${${_link_config_flags_variable}}")
  if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    string(APPEND _cuda_flags
           " CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  if(GPU_SUITE_BUILD_OPENACC)
    set(_openacc_compile_flags
        "${_cxx_flags} ${OpenACC_CXX_FLAGS} ${GPU_SUITE_OPENACC_COMPILE_OPTIONS}")
    set(_openacc_link_flags
        "${_link_flags} ${OpenACC_CXX_FLAGS} ${GPU_SUITE_OPENACC_LINK_OPTIONS}")
    set(_openacc_thrust_compile_flags "${_openacc_compile_flags} -cuda")
    set(_openacc_thrust_link_flags "${_openacc_link_flags} -cuda")
  else()
    set(_openacc_compile_flags "")
    set(_openacc_link_flags "")
    set(_openacc_thrust_compile_flags "")
    set(_openacc_thrust_link_flags "")
  endif()
  foreach(_flags c_flags cxx_flags cuda_flags openacc_compile_flags
                 openacc_link_flags openacc_thrust_compile_flags
                 openacc_thrust_link_flags)
    string(REPLACE ";" " " _${_flags} "${_${_flags}}")
    string(STRIP "${_${_flags}}" _${_flags})
  endforeach()
  if(DEFINED ENV{NVHPC_CUDA_HOME})
    set(_nvhpc_cuda_home "$ENV{NVHPC_CUDA_HOME}")
  else()
    set(_nvhpc_cuda_home "")
  endif()

  _gpu_suite_escape_c_string("${_git_commit}" GPU_SUITE_META_GIT_COMMIT)
  _gpu_suite_escape_c_string("${_git_dirty}" GPU_SUITE_META_GIT_DIRTY)
  _gpu_suite_escape_c_string("${_build_type}" GPU_SUITE_META_BUILD_TYPE)
  _gpu_suite_escape_c_string("${_build_profile}"
                             GPU_SUITE_META_BUILD_PROFILE)
  _gpu_suite_escape_c_string("${CMAKE_C_COMPILER_ID}"
                             GPU_SUITE_META_C_COMPILER_ID)
  _gpu_suite_escape_c_string("${CMAKE_C_COMPILER_VERSION}"
                             GPU_SUITE_META_C_COMPILER_VERSION)
  _gpu_suite_escape_c_string("${CMAKE_CXX_COMPILER_ID}"
                             GPU_SUITE_META_CXX_COMPILER_ID)
  _gpu_suite_escape_c_string("${CMAKE_CXX_COMPILER_VERSION}"
                             GPU_SUITE_META_CXX_COMPILER_VERSION)
  _gpu_suite_escape_c_string("${_c_flags}" GPU_SUITE_META_C_FLAGS)
  _gpu_suite_escape_c_string("${_cxx_flags}" GPU_SUITE_META_CXX_FLAGS)
  _gpu_suite_escape_c_string("${CMAKE_CUDA_COMPILER_ID}"
                             GPU_SUITE_META_CUDA_COMPILER_ID)
  _gpu_suite_escape_c_string("${CMAKE_CUDA_COMPILER_VERSION}"
                             GPU_SUITE_META_CUDA_COMPILER_VERSION)
  _gpu_suite_escape_c_string("${_cuda_flags}" GPU_SUITE_META_CUDA_FLAGS)
  _gpu_suite_escape_c_string("${_openacc_compile_flags}"
                             GPU_SUITE_META_OPENACC_COMPILE_FLAGS)
  _gpu_suite_escape_c_string("${_openacc_thrust_compile_flags}"
                             GPU_SUITE_META_OPENACC_THRUST_COMPILE_FLAGS)
  _gpu_suite_escape_c_string("${CUDAToolkit_VERSION}"
                             GPU_SUITE_META_CUDA_TOOLKIT_VERSION)
  _gpu_suite_escape_c_string("${GPU_SUITE_CUDA_TOOLKIT_ROOT}"
                             GPU_SUITE_META_CUDA_TOOLKIT_ROOT)
  _gpu_suite_escape_c_string("${GPU_SUITE_NVHPC_GPU_TARGET}"
                             GPU_SUITE_META_NVHPC_GPU_TARGET)

  _gpu_suite_escape_json_string("${_git_commit}"
                                GPU_SUITE_META_GIT_COMMIT_JSON)
  _gpu_suite_escape_json_string("${_build_type}"
                                GPU_SUITE_META_BUILD_TYPE_JSON)
  _gpu_suite_escape_json_string("${_build_profile}"
                                GPU_SUITE_META_BUILD_PROFILE_JSON)
  _gpu_suite_escape_json_string("${CMAKE_C_COMPILER_ID}"
                                GPU_SUITE_META_C_COMPILER_ID_JSON)
  _gpu_suite_escape_json_string("${CMAKE_C_COMPILER_VERSION}"
                                GPU_SUITE_META_C_COMPILER_VERSION_JSON)
  _gpu_suite_escape_json_string("${CMAKE_CXX_COMPILER_ID}"
                                GPU_SUITE_META_CXX_COMPILER_ID_JSON)
  _gpu_suite_escape_json_string("${CMAKE_CXX_COMPILER_VERSION}"
                                GPU_SUITE_META_CXX_COMPILER_VERSION_JSON)
  _gpu_suite_escape_json_string("${_c_flags}" GPU_SUITE_META_C_FLAGS_JSON)
  _gpu_suite_escape_json_string("${_cxx_flags}" GPU_SUITE_META_CXX_FLAGS_JSON)
  _gpu_suite_escape_json_string("${CMAKE_CUDA_COMPILER_ID}"
                                GPU_SUITE_META_CUDA_COMPILER_ID_JSON)
  _gpu_suite_escape_json_string("${CMAKE_CUDA_COMPILER_VERSION}"
                                GPU_SUITE_META_CUDA_COMPILER_VERSION_JSON)
  _gpu_suite_escape_json_string("${_cuda_flags}"
                                GPU_SUITE_META_CUDA_FLAGS_JSON)
  _gpu_suite_escape_json_string("${CMAKE_CUDA_ARCHITECTURES}"
                                GPU_SUITE_META_CUDA_ARCHITECTURES_JSON)
  _gpu_suite_escape_json_string("${CUDAToolkit_VERSION}"
                                GPU_SUITE_META_CUDA_TOOLKIT_VERSION_JSON)
  _gpu_suite_escape_json_string("${GPU_SUITE_CUDA_TOOLKIT_ROOT}"
                                GPU_SUITE_META_CUDA_TOOLKIT_ROOT_JSON)
  _gpu_suite_escape_json_string("${_openacc_compile_flags}"
                                GPU_SUITE_META_OPENACC_COMPILE_FLAGS_JSON)
  _gpu_suite_escape_json_string("${_openacc_link_flags}"
                                GPU_SUITE_META_OPENACC_LINK_FLAGS_JSON)
  _gpu_suite_escape_json_string("${_openacc_thrust_compile_flags}"
                                GPU_SUITE_META_OPENACC_THRUST_COMPILE_FLAGS_JSON)
  _gpu_suite_escape_json_string("${_openacc_thrust_link_flags}"
                                GPU_SUITE_META_OPENACC_THRUST_LINK_FLAGS_JSON)
  _gpu_suite_escape_json_string("${GPU_SUITE_NVHPC_GPU_TARGET}"
                                GPU_SUITE_META_NVHPC_GPU_TARGET_JSON)
  _gpu_suite_escape_json_string("${_nvhpc_cuda_home}"
                                GPU_SUITE_META_NVHPC_CUDA_HOME_JSON)

  set(_generated_dir "${CMAKE_BINARY_DIR}/generated/include/gpu_suite")
  file(MAKE_DIRECTORY "${_generated_dir}")
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gpu_suite_build_metadata.json.in"
    "${CMAKE_BINARY_DIR}/build-metadata.json"
    @ONLY)
  file(SHA256 "${CMAKE_BINARY_DIR}/build-metadata.json"
       GPU_SUITE_BUILD_METADATA_SHA256)
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gpu_suite_build_metadata.h.in"
    "${_generated_dir}/build_metadata.h"
    @ONLY)
  set(GPU_SUITE_BUILD_METADATA_FILE
      "${CMAKE_BINARY_DIR}/build-metadata.json" CACHE INTERNAL
      "Generated build metadata document")
  set(GPU_SUITE_BUILD_METADATA_SHA256
      "${GPU_SUITE_BUILD_METADATA_SHA256}" CACHE INTERNAL
      "Generated build metadata SHA-256" FORCE)
  set(GPU_SUITE_GENERATED_INCLUDE_DIR
      "${CMAKE_BINARY_DIR}/generated/include" CACHE INTERNAL
      "Generated GPU Suite include directory")
endfunction()
