cmake_minimum_required(VERSION 3.20)
project(gpu_suite_probe_fixture LANGUAGES C)

if(NOT DEFINED GPU_SUITE_ROOT OR NOT DEFINED PROBE_LIBRARY_FILE)
  message(FATAL_ERROR "GPU_SUITE_ROOT and PROBE_LIBRARY_FILE are required")
endif()

list(APPEND CMAKE_MODULE_PATH "${GPU_SUITE_ROOT}/cmake")
include(GpuSuiteProbe)

set(_probe_source [=[
#include <probe_api.h>
int main(void) { return probe_api_symbol(1) == 2 ? 0 : 1; }
]=])

gpu_suite_check_c_link(GPU_SUITE_HEADER_ONLY_LINKED
  SOURCE "${_probe_source}"
  INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(GPU_SUITE_HEADER_ONLY_LINKED)
  message(FATAL_ERROR "header-only probe unexpectedly linked")
endif()

gpu_suite_check_c_link(GPU_SUITE_REAL_SYMBOL_LINKED
  SOURCE "${_probe_source}"
  INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include"
  LIBRARIES "${PROBE_LIBRARY_FILE}")
if(NOT GPU_SUITE_REAL_SYMBOL_LINKED)
  message(FATAL_ERROR "real-symbol probe did not compile and link")
endif()
