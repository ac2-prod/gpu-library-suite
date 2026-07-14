include(CMakeParseArguments)

function(gpu_suite_configure_executable target source)
  cmake_parse_arguments(TARGET "" "LIBRARY;IMPLEMENTATION;ROLE;BUILD_PROFILE;BACKEND_VARIANT"
                        "CPU_BACKENDS" ${ARGN})
  get_filename_component(_source_stem "${source}" NAME_WE)
  if(NOT "${target}" STREQUAL "${_source_stem}")
    message(FATAL_ERROR
      "target ${target} must equal source stem ${_source_stem} for ${source}")
  endif()
  foreach(_required LIBRARY IMPLEMENTATION ROLE BUILD_PROFILE BACKEND_VARIANT)
    if(NOT DEFINED TARGET_${_required} OR "${TARGET_${_required}}" STREQUAL "")
      message(FATAL_ERROR "gpu_suite_configure_executable requires ${_required}")
    endif()
  endforeach()
  if(TARGET_IMPLEMENTATION STREQUAL "openacc")
    target_compile_options(${target} PRIVATE
      ${GPU_SUITE_OPENACC_COMPILE_OPTIONS})
    target_link_options(${target} PRIVATE
      ${GPU_SUITE_OPENACC_LINK_OPTIONS})
  endif()
  get_filename_component(_extension "${source}" EXT)
  if(_extension STREQUAL ".c")
    set(_compiler_language "c")
  elseif(_extension STREQUAL ".cu")
    set(_compiler_language "cuda")
  else()
    set(_compiler_language "cxx")
  endif()
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${_source_stem}"
    GPU_SUITE_LIBRARY "${TARGET_LIBRARY}"
    GPU_SUITE_IMPLEMENTATION "${TARGET_IMPLEMENTATION}"
    GPU_SUITE_EXECUTABLE_ROLE "${TARGET_ROLE}"
    GPU_SUITE_BUILD_PROFILE "${TARGET_BUILD_PROFILE}"
    GPU_SUITE_BACKEND_VARIANT "${TARGET_BACKEND_VARIANT}"
    GPU_SUITE_COMPILER_LANGUAGE "${_compiler_language}"
    GPU_SUITE_CPU_BACKENDS "${TARGET_CPU_BACKENDS}")
  set_property(GLOBAL APPEND PROPERTY GPU_SUITE_EXECUTABLE_TARGETS "${target}")
endfunction()

function(_gpu_suite_json_escape input output)
  string(REPLACE "\\" "\\\\" _value "${input}")
  string(REPLACE "\"" "\\\"" _value "${_value}")
  set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(gpu_suite_add_partial_manifest_target)
  get_property(_targets GLOBAL PROPERTY GPU_SUITE_EXECUTABLE_TARGETS)
  if(NOT _targets)
    gpu_suite_record_status("Partial executable manifest" FALSE
                            "no executable targets are enabled")
    return()
  endif()
  find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
  if(NOT Python3_Interpreter_FOUND)
    gpu_suite_record_status("Partial executable manifest" FALSE
                            "Python 3.9+ interpreter was not found")
    return()
  endif()

  set(_entries "")
  set(_separator "")
  foreach(_target IN LISTS _targets)
    get_target_property(_library ${_target} GPU_SUITE_LIBRARY)
    get_target_property(_implementation ${_target} GPU_SUITE_IMPLEMENTATION)
    get_target_property(_role ${_target} GPU_SUITE_EXECUTABLE_ROLE)
    get_target_property(_profile ${_target} GPU_SUITE_BUILD_PROFILE)
    get_target_property(_variant ${_target} GPU_SUITE_BACKEND_VARIANT)
    get_target_property(_language ${_target} GPU_SUITE_COMPILER_LANGUAGE)
    get_target_property(_cpu_backends ${_target} GPU_SUITE_CPU_BACKENDS)
    if(_cpu_backends STREQUAL "_cpu_backends-NOTFOUND")
      set(_cpu_backends "")
    endif()
    set(_cpu_backends_json "[")
    set(_cpu_backend_separator "")
    foreach(_cpu_backend IN LISTS _cpu_backends)
      _gpu_suite_json_escape("${_cpu_backend}" _cpu_backend_json)
      string(APPEND _cpu_backends_json
        "${_cpu_backend_separator}\"${_cpu_backend_json}\"")
      set(_cpu_backend_separator ",")
    endforeach()
    string(APPEND _cpu_backends_json "]")
    foreach(_name library implementation role profile variant language)
      _gpu_suite_json_escape("${_${_name}}" _${_name}_json)
    endforeach()
    string(APPEND _entries
      "${_separator}{\"backend_variant\":\"${_variant_json}\","
      "\"build_profile\":\"${_profile_json}\","
      "\"compiler_language\":\"${_language_json}\","
      "\"cpu_backends\":${_cpu_backends_json},"
      "\"executable_path\":\"$<TARGET_FILE:${_target}>\","
      "\"executable_role\":\"${_role_json}\","
      "\"implementation\":\"${_implementation_json}\","
      "\"library\":\"${_library_json}\","
      "\"target_name\":\"${_target}\"}")
    set(_separator ",")
  endforeach()
  set(_input "${CMAKE_BINARY_DIR}/partial-manifest-input.json")
  set(_output "${CMAKE_BINARY_DIR}/partial-manifest.json")
  file(GENERATE OUTPUT "${_input}"
    CONTENT
      "{\"build_metadata_path\":\"${GPU_SUITE_BUILD_METADATA_FILE}\",\"entries\":[${_entries}]}\n")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/generate_partial_manifest.py"
            "${_input}" "${_output}"
    DEPENDS ${_targets} "${GPU_SUITE_BUILD_METADATA_FILE}"
            "${CMAKE_SOURCE_DIR}/tools/generate_partial_manifest.py"
    VERBATIM)
  add_custom_target(gpu_suite_partial_manifest DEPENDS "${_output}")
  gpu_suite_record_status("Partial executable manifest" TRUE
                          "target gpu_suite_partial_manifest")
endfunction()
