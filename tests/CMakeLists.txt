find_package(Python3 3.9 COMPONENTS Interpreter QUIET)

function(gpu_suite_add_c_test name source)
  add_executable(${name} "c/${source}")
  target_link_libraries(${name} PRIVATE gpu_suite::common)
  target_compile_definitions(${name} PRIVATE _POSIX_C_SOURCE=200809L)
  set_target_properties(${name} PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${name} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

gpu_suite_add_c_test(test_checked test_checked.c)
gpu_suite_add_c_test(test_clock test_clock.c)
gpu_suite_add_c_test(test_json test_json.c)
gpu_suite_add_c_test(test_cli test_cli.c)
gpu_suite_add_c_test(test_result test_result.c)

add_executable(test_cpp_compat c/test_cpp_compat.cpp)
target_link_libraries(test_cpp_compat PRIVATE gpu_suite::common)
set_target_properties(test_cpp_compat PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(test_cpp_compat PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
add_test(NAME test_cpp_compat COMMAND test_cpp_compat)

add_executable(test_verification_nonfinite c/test_verification_nonfinite.cpp)
target_link_libraries(test_verification_nonfinite PRIVATE gpu_suite::common)
target_include_directories(test_verification_nonfinite PRIVATE
  "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/cufft/benchmarks"
  "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/curand/benchmarks"
  "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/cusolver/benchmarks"
  "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/cusparse/benchmarks"
  "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/thrust/benchmarks")
set_target_properties(test_verification_nonfinite PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(test_verification_nonfinite PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
add_test(NAME test_verification_nonfinite COMMAND test_verification_nonfinite)

set(_gpu_suite_fixture_root "${CMAKE_BINARY_DIR}/test-fixtures")
set(_gpu_suite_fixture_pycache "${_gpu_suite_fixture_root}/pycache")
set(_gpu_suite_clean_provider_environment
    "${CMAKE_COMMAND}" -E env
    --unset=FFTW_ROOT
    --unset=FFTW3_ROOT
    --unset=FFTW3F_ROOT
    --unset=FFTW3f_ROOT
    --unset=CMAKE_PREFIX_PATH
    --unset=CMAKE_INCLUDE_PATH
    --unset=CMAKE_LIBRARY_PATH
    --unset=PKG_CONFIG_PATH
    --unset=PKG_CONFIG_LIBDIR
    --unset=MKLROOT
    --unset=ONEAPI_ROOT
    --unset=CPATH
    --unset=C_INCLUDE_PATH
    --unset=CPLUS_INCLUDE_PATH
    --unset=LIBRARY_PATH)
set(_gpu_suite_unavailable_provider_root
    "${_gpu_suite_fixture_root}/unavailable-provider")
set(_gpu_suite_disable_cpu_provider_options
    "-DGPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR=${_gpu_suite_unavailable_provider_root}/include"
    "-DGPU_SUITE_ONEMKL_CBLAS_LIBRARY=${_gpu_suite_unavailable_provider_root}/lib/libmkl_rt.a"
    "-DGPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR=${_gpu_suite_unavailable_provider_root}/include"
    "-DGPU_SUITE_ONEMKL_LAPACKE_LIBRARY=${_gpu_suite_unavailable_provider_root}/lib/libmkl_rt.a"
    "-DGPU_SUITE_MKL_SPARSE_INCLUDE_DIR=${_gpu_suite_unavailable_provider_root}/include"
    "-DGPU_SUITE_MKL_SPARSE_LIBRARY=${_gpu_suite_unavailable_provider_root}/lib/libmkl_rt.a")

add_library(probe_fixture_symbol STATIC cmake/probe_symbol.c)
target_include_directories(probe_fixture_symbol PRIVATE cmake/probe/include)
set_target_properties(probe_fixture_symbol PROPERTIES
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)

add_library(fake_fftw3f STATIC fixtures/fftw/fake_fftw3f.c)
target_include_directories(fake_fftw3f PUBLIC fixtures/fftw/include)
set_target_properties(fake_fftw3f PROPERTIES
  OUTPUT_NAME fftw3f
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)
add_library(fake_fftw3f_threads STATIC fixtures/fftw/fake_fftw3f_threads.c)
target_include_directories(fake_fftw3f_threads PUBLIC fixtures/fftw/include)
set_target_properties(fake_fftw3f_threads PROPERTIES
  OUTPUT_NAME fftw3f_threads
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)

add_library(fake_cpu_providers STATIC
  fixtures/cpu_libs/fake_cpu_providers.c)
target_include_directories(fake_cpu_providers PUBLIC
  fixtures/cpu_libs/include)
set_target_properties(fake_cpu_providers PROPERTIES
  OUTPUT_NAME mkl_rt
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)

add_library(fake_openblas STATIC
  fixtures/cpu_libs/fake_cpu_providers.c)
target_include_directories(fake_openblas PUBLIC
  fixtures/cpu_libs/include)
set_target_properties(fake_openblas PROPERTIES
  OUTPUT_NAME openblas
  C_STANDARD 17
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO)

add_test(
  NAME cmake_compile_link_probe
  COMMAND "${CMAKE_COMMAND}"
          -S "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe"
          -B "${_gpu_suite_fixture_root}/cmake-probe"
          "-DGPU_SUITE_ROOT=${CMAKE_SOURCE_DIR}"
          "-DPROBE_LIBRARY_FILE=$<TARGET_FILE:probe_fixture_symbol>")

set(_fftw_positive_build
    "${_gpu_suite_fixture_root}/fftw-positive")
add_test(
  NAME cmake_fftw_positive_configure
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_fftw_positive_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          ${_gpu_suite_disable_cpu_provider_options}
          "-DFFTW3f_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw/include"
          "-DFFTW3f_LIBRARY=$<TARGET_FILE:fake_fftw3f>"
          "-DFFTW3f_THREADS_LIBRARY=$<TARGET_FILE:fake_fftw3f_threads>")
add_test(
  NAME cmake_fftw_positive_build
  COMMAND "${CMAKE_COMMAND}" --build "${_fftw_positive_build}"
          --target fft_cpu fft_cpu_bench gpu_suite_partial_manifest --parallel 2)
set_tests_properties(cmake_fftw_positive_build PROPERTIES
  DEPENDS cmake_fftw_positive_configure)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME fftw_fixture_partial_manifest
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/check_partial_manifest.py"
            "${_fftw_positive_build}/partial-manifest.json"
            fft_cpu fft_cpu_bench rand_cpu rand_cpu_bench
            reduce_cpu reduce_cpu_bench)
  set_tests_properties(fftw_fixture_partial_manifest PROPERTIES
    DEPENDS cmake_fftw_positive_build
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/tools;PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}")
endif()
add_test(
  NAME fftw_fixture_serial_compute
  COMMAND "${_fftw_positive_build}/nvidia/c-cpp/cufft/fft_cpu_bench"
          --size 8 --batch 2 --warmup 1 --repeat 2 --trials 2
          --scope compute --verify true --output - --format jsonl
          --cpu-backend cpu-fftw-serial --cpu-threads 48
          --cpu-threads-effective 1 --cpu-backend-role reference
          --series-role auxiliary --cpu-parallelism serial)
set_tests_properties(fftw_fixture_serial_compute PROPERTIES
  DEPENDS cmake_fftw_positive_build)
add_test(
  NAME fftw_fixture_threaded_end_to_end
  COMMAND "${_fftw_positive_build}/nvidia/c-cpp/cufft/fft_cpu_bench"
          --size 8 --batch 2 --warmup 1 --repeat 1 --trials 2
          --scope end-to-end --verify true --output - --format jsonl
          --cpu-backend cpu-fftw-threaded --cpu-threads 4
          --cpu-threads-effective 4 --cpu-backend-role production
          --series-role primary --cpu-parallelism threaded)
set_tests_properties(fftw_fixture_threaded_end_to_end PROPERTIES
  DEPENDS cmake_fftw_positive_build)

set(_fftw_negative_build
    "${_gpu_suite_fixture_root}/fftw-negative")
add_test(
  NAME cmake_fftw_header_only_negative
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_fftw_negative_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          ${_gpu_suite_disable_cpu_provider_options}
          "-DFFTW3f_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw/include"
          "-DFFTW3f_LIBRARY=$<TARGET_FILE:probe_fixture_symbol>"
          "-DFFTW3f_THREADS_LIBRARY=$<TARGET_FILE:fake_fftw3f_threads>")
add_test(
  NAME cmake_fftw_header_only_has_no_target
  COMMAND "${CMAKE_COMMAND}" --build "${_fftw_negative_build}"
          --target fft_cpu)
set_tests_properties(cmake_fftw_header_only_has_no_target PROPERTIES
  DEPENDS cmake_fftw_header_only_negative
  WILL_FAIL TRUE)

set(_cpu_provider_positive_build
    "${_gpu_suite_fixture_root}/cpu-provider-positive")
add_test(
  NAME cmake_cpu_provider_positive_configure
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_cpu_provider_positive_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=TRUE
          -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL
          "-DGPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_CBLAS_LIBRARY=$<TARGET_FILE:fake_cpu_providers>"
          "-DGPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_LAPACKE_LIBRARY=$<TARGET_FILE:fake_cpu_providers>"
          "-DGPU_SUITE_MKL_SPARSE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_MKL_SPARSE_LIBRARY=$<TARGET_FILE:fake_cpu_providers>")
add_test(
  NAME cmake_cpu_provider_positive_build
  COMMAND "${CMAKE_COMMAND}" --build "${_cpu_provider_positive_build}"
          --target blas_cpu blas_cpu_bench sparse_cpu sparse_cpu_bench
                   solver_cpu solver_cpu_bench gpu_suite_partial_manifest
          --parallel 2)
set_tests_properties(cmake_cpu_provider_positive_build PROPERTIES
  DEPENDS cmake_cpu_provider_positive_configure)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME cpu_provider_fixture_partial_manifest
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/check_partial_manifest.py"
            "${_cpu_provider_positive_build}/partial-manifest.json"
            blas_cpu blas_cpu_bench sparse_cpu sparse_cpu_bench
            solver_cpu solver_cpu_bench rand_cpu rand_cpu_bench
            reduce_cpu reduce_cpu_bench)
  set_tests_properties(cpu_provider_fixture_partial_manifest PROPERTIES
    DEPENDS cmake_cpu_provider_positive_build
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_SOURCE_DIR}/tools;PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}")
endif()

add_test(
  NAME cblas_fixture_compute
  COMMAND "${_cpu_provider_positive_build}/nvidia/c-cpp/cublas/blas_cpu_bench"
          --size 8 --warmup 1 --repeat 2 --trials 2 --scope compute
          --verify true --output - --format jsonl --cpu-backend cpu-onemkl
          --cpu-threads 48 --cpu-backend-role production
          --series-role primary --cpu-parallelism threaded)
set_tests_properties(cblas_fixture_compute PROPERTIES
  DEPENDS cmake_cpu_provider_positive_build)
add_test(
  NAME mkl_sparse_fixture_compute
  COMMAND "${_cpu_provider_positive_build}/nvidia/c-cpp/cusparse/sparse_cpu_bench"
          --size 16 --warmup 1 --repeat 2 --trials 2 --scope compute
          --verify true --output - --format jsonl --cpu-backend cpu-onemkl
          --cpu-threads 48 --cpu-backend-role production
          --series-role primary --cpu-parallelism threaded)
set_tests_properties(mkl_sparse_fixture_compute PROPERTIES
  DEPENDS cmake_cpu_provider_positive_build)
add_test(
  NAME lapacke_fixture_end_to_end
  COMMAND "${_cpu_provider_positive_build}/nvidia/c-cpp/cusolver/solver_cpu_bench"
          --size 8 --nrhs 2 --warmup 1 --repeat 1 --trials 2
          --scope end-to-end --verify true --output - --format jsonl
          --cpu-backend cpu-onemkl --cpu-threads 48
          --cpu-backend-role production --series-role primary
          --cpu-parallelism threaded)
set_tests_properties(lapacke_fixture_end_to_end PROPERTIES
  DEPENDS cmake_cpu_provider_positive_build)

set(_cpu_provider_negative_build
    "${_gpu_suite_fixture_root}/cpu-provider-negative")
add_test(
  NAME cmake_cpu_provider_symbol_negative
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_cpu_provider_negative_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=TRUE
          -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL
          "-DGPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_CBLAS_LIBRARY=$<TARGET_FILE:probe_fixture_symbol>"
          "-DGPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_LAPACKE_LIBRARY=$<TARGET_FILE:probe_fixture_symbol>"
          "-DGPU_SUITE_MKL_SPARSE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_MKL_SPARSE_LIBRARY=$<TARGET_FILE:probe_fixture_symbol>")
foreach(_missing_cpu_target blas_cpu sparse_cpu solver_cpu)
  add_test(
    NAME "cpu_provider_negative_no_${_missing_cpu_target}"
    COMMAND "${CMAKE_COMMAND}" --build "${_cpu_provider_negative_build}"
            --target "${_missing_cpu_target}")
  set_tests_properties("cpu_provider_negative_no_${_missing_cpu_target}"
    PROPERTIES
      DEPENDS cmake_cpu_provider_symbol_negative
      WILL_FAIL TRUE)
endforeach()

set(_cpu_provider_mixed_build
    "${_gpu_suite_fixture_root}/cpu-provider-mixed")
add_test(
  NAME cmake_cpu_provider_mixed_rejected
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_cpu_provider_mixed_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=TRUE
          -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE
          "-DGPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_CBLAS_LIBRARY=$<TARGET_FILE:fake_openblas>"
          "-DGPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_LAPACKE_LIBRARY=$<TARGET_FILE:fake_openblas>")
add_test(
  NAME cmake_cpu_provider_mixed_no_blas_target
  COMMAND "${CMAKE_COMMAND}" --build "${_cpu_provider_mixed_build}"
          --target blas_cpu)
set_tests_properties(cmake_cpu_provider_mixed_no_blas_target PROPERTIES
  DEPENDS cmake_cpu_provider_mixed_rejected
  WILL_FAIL TRUE)
add_test(
  NAME cmake_cpu_provider_mixed_no_solver_target
  COMMAND "${CMAKE_COMMAND}" --build "${_cpu_provider_mixed_build}"
          --target solver_cpu)
set_tests_properties(cmake_cpu_provider_mixed_no_solver_target PROPERTIES
  DEPENDS cmake_cpu_provider_mixed_rejected
  WILL_FAIL TRUE)

set(_cpu_provider_switch_build
    "${_gpu_suite_fixture_root}/cpu-provider-switch")
add_test(
  NAME cmake_cpu_provider_switch_onemkl
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_cpu_provider_switch_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=TRUE
          -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
          -DGPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE
          "-DGPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_CBLAS_LIBRARY=$<TARGET_FILE:fake_cpu_providers>"
          "-DGPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_ONEMKL_LAPACKE_LIBRARY=$<TARGET_FILE:fake_cpu_providers>")
add_test(
  NAME cmake_cpu_provider_switch_openblas
  COMMAND ${_gpu_suite_clean_provider_environment}
          "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_cpu_provider_switch_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DGPU_SUITE_BUILD_CPU=ON
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=OFF
          -DGPU_SUITE_BUILD_TESTS=OFF
          -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=TRUE
          -DGPU_SUITE_CPU_BLAS_BACKEND=OPENBLAS
          -DGPU_SUITE_CPU_LAPACK_BACKEND=OPENBLAS
          -DGPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE
          "-DGPU_SUITE_OPENBLAS_CBLAS_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_OPENBLAS_CBLAS_LIBRARY=$<TARGET_FILE:fake_openblas>"
          "-DGPU_SUITE_OPENBLAS_LAPACKE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
          "-DGPU_SUITE_OPENBLAS_LAPACKE_LIBRARY=$<TARGET_FILE:fake_openblas>")
set_tests_properties(cmake_cpu_provider_switch_openblas PROPERTIES
  DEPENDS cmake_cpu_provider_switch_onemkl)
add_test(
  NAME cmake_cpu_provider_switch_build
  COMMAND "${CMAKE_COMMAND}" --build "${_cpu_provider_switch_build}"
          --target blas_cpu solver_cpu --parallel 2)
set_tests_properties(cmake_cpu_provider_switch_build PROPERTIES
  DEPENDS cmake_cpu_provider_switch_openblas)

if(TARGET rand_cpu_bench)
  add_test(
    NAME rand_cpu_fixture_compute
    COMMAND $<TARGET_FILE:rand_cpu_bench>
            --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute
            --verify true --output - --format jsonl
            --cpu-backend cpu-std-random-serial --cpu-threads 48
            --cpu-threads-effective 1 --cpu-backend-role production
            --series-role primary --cpu-parallelism serial)
endif()
if(TARGET reduce_cpu_bench)
  add_test(
    NAME reduce_cpu_fixture_end_to_end
    COMMAND $<TARGET_FILE:reduce_cpu_bench>
            --size 4096 --warmup 1 --repeat 1 --trials 2 --scope end-to-end
            --verify true --output - --format jsonl
            --cpu-backend cpu-stl-serial --cpu-threads 48
            --cpu-threads-effective 1 --cpu-backend-role production
            --series-role primary --cpu-parallelism serial)
endif()

foreach(_cufft_cpp_source
    examples/fft_gpu.cu
    examples/openacc_cufft.cpp
    benchmarks/fft_gpu_bench.cu
    benchmarks/openacc_cufft_bench.cpp)
  get_filename_component(_cufft_cpp_stem "${_cufft_cpp_source}" NAME_WE)
  add_test(
    NAME "cufft_syntax_${_cufft_cpp_stem}"
    COMMAND "${CMAKE_CXX_COMPILER}"
            -x c++ -std=c++17 -fsyntax-only -Wno-unknown-pragmas
            -DGPU_SUITE_HAVE_CUDA_RUNTIME_METADATA=1
            -DGPU_SUITE_CUDA_LIBRARY_CUFFT=1
            "-I${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cuda"
            "-I${CMAKE_SOURCE_DIR}/common/c-cpp/include"
            "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/cufft/${_cufft_cpp_source}")
endforeach()

foreach(_phase3_library cublas cusparse cusolver curand thrust)
  if(_phase3_library STREQUAL "cublas")
    set(_phase3_stem blas)
  elseif(_phase3_library STREQUAL "cusparse")
    set(_phase3_stem sparse)
  elseif(_phase3_library STREQUAL "cusolver")
    set(_phase3_stem solver)
  elseif(_phase3_library STREQUAL "curand")
    set(_phase3_stem rand)
  else()
    set(_phase3_stem reduce)
  endif()
  string(TOUPPER "${_phase3_library}" _phase3_metadata_macro)
  foreach(_phase3_source
      "examples/${_phase3_stem}_gpu.cu"
      "examples/openacc_${_phase3_library}.cpp"
      "benchmarks/${_phase3_stem}_gpu_bench.cu"
      "benchmarks/openacc_${_phase3_library}_bench.cpp")
    get_filename_component(_phase3_source_stem "${_phase3_source}" NAME_WE)
    add_test(
      NAME "phase3_syntax_${_phase3_library}_${_phase3_source_stem}"
      COMMAND "${CMAKE_CXX_COMPILER}"
              -x c++ -std=c++17 -fsyntax-only -Wno-unknown-pragmas
              -DGPU_SUITE_HAVE_CUDA_RUNTIME_METADATA=1
              "-DGPU_SUITE_CUDA_LIBRARY_${_phase3_metadata_macro}=1"
              "-I${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cuda"
              "-I${CMAKE_SOURCE_DIR}/common/c-cpp/include"
              "-I${CMAKE_SOURCE_DIR}/nvidia/c-cpp/${_phase3_library}/benchmarks"
              "${CMAKE_SOURCE_DIR}/nvidia/c-cpp/${_phase3_library}/${_phase3_source}")
  endforeach()
endforeach()

foreach(_cuda_metadata_library cufft cublas cusparse cusolver curand thrust)
  string(TOUPPER "${_cuda_metadata_library}" _cuda_metadata_macro)
  set(_cuda_metadata_target "test_cuda_metadata_${_cuda_metadata_library}")
  add_executable(${_cuda_metadata_target} c/test_cuda_metadata.cpp)
  target_link_libraries(${_cuda_metadata_target} PRIVATE gpu_suite::common)
  target_include_directories(${_cuda_metadata_target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cuda")
  target_compile_definitions(${_cuda_metadata_target} PRIVATE
    "GPU_SUITE_CUDA_LIBRARY_${_cuda_metadata_macro}=1")
  set_target_properties(${_cuda_metadata_target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  add_test(NAME ${_cuda_metadata_target} COMMAND ${_cuda_metadata_target})
endforeach()

set(_openacc_only_tests_build
    "${_gpu_suite_fixture_root}/openacc-only-tests")
add_test(
  NAME cmake_openacc_only_tests_generate
  COMMAND "${CMAKE_COMMAND}"
          -S "${CMAKE_SOURCE_DIR}"
          -B "${_openacc_only_tests_build}"
          -G "${CMAKE_GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_DISABLE_FIND_PACKAGE_OpenACC=TRUE
          -DGPU_SUITE_BUILD_CPU=OFF
          -DGPU_SUITE_BUILD_CUDA=OFF
          -DGPU_SUITE_BUILD_OPENACC=ON
          -DGPU_SUITE_BUILD_TESTS=ON)
set_tests_properties(cmake_openacc_only_tests_generate PROPERTIES TIMEOUT 300)

foreach(_pegasus_shell
    build_cpu_cuda.sh
    build_openacc.sh
    run_node.sh
    run_benchmarks.pbs.in)
  string(REPLACE "." "_" _pegasus_shell_test "${_pegasus_shell}")
  add_test(
    NAME "pegasus_bash_syntax_${_pegasus_shell_test}"
    COMMAND bash -n
            "${CMAKE_SOURCE_DIR}/jobs/pegasus/${_pegasus_shell}")
endforeach()

find_program(SHELLCHECK_EXECUTABLE shellcheck)
if(SHELLCHECK_EXECUTABLE)
  foreach(_pegasus_shell build_cpu_cuda.sh build_openacc.sh run_node.sh)
    string(REPLACE "." "_" _pegasus_shell_test "${_pegasus_shell}")
    add_test(
      NAME "pegasus_shellcheck_${_pegasus_shell_test}"
      COMMAND "${SHELLCHECK_EXECUTABLE}"
              "${CMAKE_SOURCE_DIR}/jobs/pegasus/${_pegasus_shell}")
  endforeach()
else()
  message(STATUS
    "ShellCheck tests not registered: shellcheck is not already available")
endif()

set(_gpu_suite_nested_cmake_tests
    cmake_compile_link_probe
    cmake_fftw_positive_configure
    cmake_fftw_positive_build
    cmake_fftw_header_only_negative
    cmake_fftw_header_only_has_no_target
    cmake_cpu_provider_positive_configure
    cmake_cpu_provider_positive_build
    cmake_cpu_provider_symbol_negative
    cpu_provider_negative_no_blas_cpu
    cpu_provider_negative_no_sparse_cpu
    cpu_provider_negative_no_solver_cpu
    cmake_cpu_provider_mixed_rejected
    cmake_cpu_provider_mixed_no_blas_target
    cmake_cpu_provider_mixed_no_solver_target
    cmake_cpu_provider_switch_onemkl
    cmake_cpu_provider_switch_openblas
    cmake_cpu_provider_switch_build
    cmake_openacc_only_tests_generate)
set(_gpu_suite_provider_fixture_configure_tests
    cmake_fftw_positive_configure
    cmake_fftw_header_only_negative
    cmake_cpu_provider_positive_configure
    cmake_cpu_provider_symbol_negative
    cmake_cpu_provider_mixed_rejected
    cmake_cpu_provider_switch_onemkl
    cmake_cpu_provider_switch_openblas)
set(_gpu_suite_provider_contamination_environment
    "FFTW_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw"
    "MKLROOT=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs"
    "CMAKE_PREFIX_PATH=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw:${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs"
    "CMAKE_INCLUDE_PATH=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw/include:${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
    "CMAKE_LIBRARY_PATH=$<TARGET_FILE_DIR:fake_cpu_providers>"
    "PKG_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/provider-environment/pkgconfig"
    "CPATH=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fftw/include:${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cpu_libs/include"
    "LIBRARY_PATH=$<TARGET_FILE_DIR:fake_cpu_providers>")
set_tests_properties(${_gpu_suite_nested_cmake_tests} PROPERTIES
  RESOURCE_LOCK gpu_suite_nested_cmake)
set_tests_properties(${_gpu_suite_provider_fixture_configure_tests} PROPERTIES
  ENVIRONMENT "${_gpu_suite_provider_contamination_environment}")
set_tests_properties(
  cmake_fftw_positive_build
  cmake_cpu_provider_positive_build
  cmake_cpu_provider_switch_build
  PROPERTIES PROCESSORS 2)

if(Python3_Interpreter_FOUND)
  add_test(
    NAME cmake_dependency_root_discovery
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/check_dependency_discovery.py"
            --cmake "${CMAKE_COMMAND}"
            --generator "${CMAKE_GENERATOR}"
            --source "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependency_roots"
            --repository-root "${CMAKE_SOURCE_DIR}"
            --fixture-root "${_gpu_suite_fixture_root}/dependency-roots"
            --headers-root "${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
            --mkl-library "$<TARGET_FILE:fake_cpu_providers>"
            --fftw-library "$<TARGET_FILE:fake_fftw3f>"
            --fftw-threads-library "$<TARGET_FILE:fake_fftw3f_threads>")
  set_tests_properties(cmake_dependency_root_discovery PROPERTIES
    ENVIRONMENT "PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}"
    RESOURCE_LOCK gpu_suite_nested_cmake
    TIMEOUT 300)

  add_test(
    NAME python_unittest
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/python" -v)
  set(_gpu_suite_python_test_environment
      "PYTHONPATH=${CMAKE_SOURCE_DIR}/tools"
      "PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}"
      "GPU_SUITE_BUILD_METADATA_FILE=${GPU_SUITE_BUILD_METADATA_FILE}"
      "GPU_SUITE_BUILD_METADATA_HEADER=${GPU_SUITE_GENERATED_INCLUDE_DIR}/gpu_suite/build_metadata.h"
      "GPU_SUITE_FFTW_FIXTURE_BENCH=${_fftw_positive_build}/nvidia/c-cpp/cufft/fft_cpu_bench"
      "GPU_SUITE_FFTW_FIXTURE_MANIFEST=${_fftw_positive_build}/partial-manifest.json"
      "GPU_SUITE_FFTW_FIXTURE_METADATA=${_fftw_positive_build}/build-metadata.json"
      "GPU_SUITE_CPU_PROVIDER_FIXTURE_MANIFEST=${_cpu_provider_positive_build}/partial-manifest.json"
      "GPU_SUITE_CPU_PROVIDER_FIXTURE_METADATA=${_cpu_provider_positive_build}/build-metadata.json"
      "GPU_SUITE_CBLAS_FIXTURE_BENCH=${_cpu_provider_positive_build}/nvidia/c-cpp/cublas/blas_cpu_bench"
      "GPU_SUITE_SPARSE_FIXTURE_BENCH=${_cpu_provider_positive_build}/nvidia/c-cpp/cusparse/sparse_cpu_bench"
      "GPU_SUITE_SOLVER_FIXTURE_BENCH=${_cpu_provider_positive_build}/nvidia/c-cpp/cusolver/solver_cpu_bench"
      "GPU_SUITE_CMAKE_COMMAND=${CMAKE_COMMAND}")
  if(TARGET rand_cpu_bench)
    list(APPEND _gpu_suite_python_test_environment
      "GPU_SUITE_RAND_CPU_BENCH=$<TARGET_FILE:rand_cpu_bench>")
  endif()
  if(TARGET reduce_cpu_bench)
    list(APPEND _gpu_suite_python_test_environment
      "GPU_SUITE_REDUCE_CPU_BENCH=$<TARGET_FILE:reduce_cpu_bench>")
  endif()
  set_tests_properties(python_unittest PROPERTIES
    DEPENDS "cmake_fftw_positive_build;cmake_cpu_provider_positive_build"
    ENVIRONMENT "${_gpu_suite_python_test_environment}")

  add_test(
    NAME python39_compat
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/check_python39.py"
            "${CMAKE_SOURCE_DIR}")
  set_tests_properties(python39_compat PROPERTIES
    ENVIRONMENT
      "PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}")

  add_test(
    NAME non_git_source_build
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/check_non_git_build.py"
            --cmake "${CMAKE_COMMAND}"
            --source "${CMAKE_SOURCE_DIR}")
  set_tests_properties(non_git_source_build PROPERTIES
    ENVIRONMENT "PYTHONPYCACHEPREFIX=${_gpu_suite_fixture_pycache}"
    PROCESSORS 2
    RESOURCE_LOCK gpu_suite_nested_cmake
    RUN_SERIAL TRUE
    TIMEOUT 900)
else()
  message(STATUS "Python tests disabled: Python 3.9+ interpreter not found")
endif()
