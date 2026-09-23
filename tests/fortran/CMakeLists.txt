add_executable(test_fortran_helpers test_helpers.f90)
target_link_libraries(test_fortran_helpers PRIVATE gpu_suite_fortran_matrices)
target_compile_options(test_fortran_helpers PRIVATE ${GPU_SUITE_FORTRAN_SAFE_OPTIONS})
add_test(NAME fortran_helpers COMMAND test_fortran_helpers)
find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME fortran_tools
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_SOURCE_DIR}/tests/python" -p test_fortran.py)
  set_tests_properties(fortran_tools PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_SOURCE_DIR}/tools;PYTHONPYCACHEPREFIX=${CMAKE_BINARY_DIR}/pycache;GPU_SUITE_FORTRAN_BUILD=${CMAKE_BINARY_DIR}")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    add_test(NAME fortran_controlled_providers
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/check_fortran_providers.py"
              --source "${CMAKE_SOURCE_DIR}" --output "${CMAKE_BINARY_DIR}/controlled-providers"
              --compiler "${CMAKE_Fortran_COMPILER}" --cmake "${CMAKE_COMMAND}")
    set_tests_properties(fortran_controlled_providers PROPERTIES ENVIRONMENT
      "PYTHONPATH=${CMAKE_SOURCE_DIR}/tools;PYTHONPYCACHEPREFIX=${CMAKE_BINARY_DIR}/pycache")
  endif()
endif()
