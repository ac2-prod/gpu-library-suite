add_library(gpu_suite_common STATIC
  src/checked.c
  src/benchmark.c
  src/clock.c
  src/json.c
  src/cli.c
  src/result.c
  src/metadata.c)

add_library(gpu_suite::common ALIAS gpu_suite_common)

target_include_directories(gpu_suite_common
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GPU_SUITE_GENERATED_INCLUDE_DIR}")

target_compile_definitions(gpu_suite_common
  PRIVATE _POSIX_C_SOURCE=200809L)

set_target_properties(gpu_suite_common PROPERTIES
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(gpu_suite_common PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

gpu_suite_record_status("Common C benchmark library" TRUE
                        "strict C17, _POSIX_C_SOURCE=200809L")
