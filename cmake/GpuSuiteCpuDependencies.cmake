include(GpuSuiteProbe)

function(_gpu_suite_cpu_header_preamble output)
  set(_preamble "")
  if(GPU_SUITE_VENDOR STREQUAL "amd")
    # Match the standalone AMD CPU sources before any system/provider header.
    # Keep a caller's higher feature level; diagnose an incompatible lower one.
    set(_preamble "#ifndef _POSIX_C_SOURCE\n#define _POSIX_C_SOURCE 200809L\n#elif _POSIX_C_SOURCE < 200809L\n#error \"AMD CPU library headers require _POSIX_C_SOURCE >= 200809L\"\n#endif\n")
  endif()
  set(${output} "${_preamble}" PARENT_SCOPE)
endfunction()

# AOCL Sparse is an explicitly selected AMD LP64 provider, never a fallback.
function(gpu_suite_find_aocl_sparse)
  set(GPU_SUITE_AOCL_SPARSE_FOUND FALSE PARENT_SCOPE)
  set(GPU_SUITE_AOCL_SPARSE_REASON "AOCL Sparse requires AMD and explicit AOCLSPARSE selection" PARENT_SCOPE)
  if(NOT GPU_SUITE_VENDOR STREQUAL "amd" OR
      NOT GPU_SUITE_CPU_SPARSE_BACKEND STREQUAL "AOCLSPARSE")
    return()
  endif()
  find_path(GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR NAMES aoclsparse.h
    HINTS ENV AOCL_ROOT PATH_SUFFIXES include include_LP64)
  find_library(GPU_SUITE_AOCL_SPARSE_LIBRARY NAMES aoclsparse
    HINTS ENV AOCL_ROOT PATH_SUFFIXES lib lib64 lib_LP64)
  if(NOT GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR OR
      NOT EXISTS "${GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR}/aoclsparse.h" OR
      NOT GPU_SUITE_AOCL_SPARSE_LIBRARY)
    set(GPU_SUITE_AOCL_SPARSE_REASON "AOCL Sparse aoclsparse.h or aoclsparse library was not found" PARENT_SCOPE)
    return()
  endif()
  _gpu_suite_generic_link_files(_libraries _invalid "${GPU_SUITE_AOCL_SPARSE_LIBRARY}"
    GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES)
  if(_invalid)
    set(GPU_SUITE_AOCL_SPARSE_REASON "${_invalid}" PARENT_SCOPE)
    return()
  endif()
  # A 32-bit header cannot detect a mismatched 64-bit library's C symbols.
  # Reject paths explicitly advertising that incompatible ABI as well.
  foreach(_path IN ITEMS "${GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR}" ${_libraries})
    file(REAL_PATH "${_path}" _resolved)
    string(TOUPPER "${_path};${_resolved}" _abi_paths)
    if(_abi_paths MATCHES "ILP64")
      set(GPU_SUITE_AOCL_SPARSE_REASON "AOCL Sparse ILP64 paths are incompatible with LP64 indices" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  _gpu_suite_cpu_header_preamble(_preamble)
  set(_source [=[
#include <limits.h>
#include <aoclsparse.h>
_Static_assert(sizeof(aoclsparse_int) * CHAR_BIT == 32, "AOCL Sparse requires LP64 32-bit indices");
_Static_assert(_Generic((aoclsparse_int)0, int: 1, default: 0), "AOCL Sparse index type must match int");
int main(void) {
  aoclsparse_matrix matrix = 0;
  aoclsparse_mat_descr descr = 0;
  int row[2] = {0, 1}, col[1] = {0};
  double value[1] = {2}, x[1] = {3}, y[1] = {0}, alpha = 1, beta = 0;
  const char *version = aoclsparse_get_version();
  if (!version || !*version) return 1;
  if (aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success ||
      aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero) != aoclsparse_status_success ||
      aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general) != aoclsparse_status_success ||
      aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit) != aoclsparse_status_success ||
      aoclsparse_create_dcsr(&matrix, aoclsparse_index_base_zero, 1, 1, 1, row, col, value) != aoclsparse_status_success ||
      aoclsparse_set_mv_hint(matrix, aoclsparse_operation_none, descr, 1) != aoclsparse_status_success ||
      aoclsparse_optimize(matrix) != aoclsparse_status_success ||
      aoclsparse_dmv(aoclsparse_operation_none, &alpha, matrix, descr, x, &beta, y) != aoclsparse_status_success)
    return 2;
  if (aoclsparse_destroy(&matrix) != aoclsparse_status_success) return 3;
  return aoclsparse_destroy_mat_descr(descr) == aoclsparse_status_success ? 0 : 4;
}
]=])
  _gpu_suite_system_link_libraries(_system_libraries)
  gpu_suite_check_c_link(GPU_SUITE_AOCL_SPARSE_LINKS
    SOURCE "${_preamble}${_source}" INCLUDES "${GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR}"
    LIBRARIES ${_libraries} ${_system_libraries})
  if(NOT GPU_SUITE_AOCL_SPARSE_LINKS)
    set(GPU_SUITE_AOCL_SPARSE_REASON "AOCL Sparse LP64 descriptor/create/hint/optimize/SpMV/destroy/version API did not compile and link" PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::AOCLSparse)
    add_library(GpuSuite::AOCLSparse INTERFACE IMPORTED)
    set_target_properties(GpuSuite::AOCLSparse PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${_libraries};${_system_libraries}")
  endif()
  set(GPU_SUITE_AOCL_SPARSE_FOUND TRUE PARENT_SCOPE)
  set(GPU_SUITE_AOCL_SPARSE_REASON "" PARENT_SCOPE)
endfunction()

function(_gpu_suite_system_link_libraries output)
  find_package(Threads QUIET)
  set(_libraries "")
  if(Threads_FOUND)
    list(APPEND _libraries Threads::Threads)
  endif()
  if(CMAKE_DL_LIBS)
    list(APPEND _libraries "${CMAKE_DL_LIBS}")
  endif()
  if(UNIX AND NOT APPLE)
    list(APPEND _libraries m)
  endif()
  set(${output} "${_libraries}" PARENT_SCOPE)
endfunction()

# Generic providers are selected by API, not a vendor filename. Additional
# inputs are explicit files in link order, never flags or fallback providers.
function(_gpu_suite_generic_link_files output reason library extra_variable)
  set(${extra_variable} "" CACHE STRING "Ordered absolute extra library files for this generic CPU provider")
  set(_files "${library}")
  if(NOT "${${extra_variable}}" STREQUAL "")
    list(APPEND _files "${${extra_variable}}")
  endif()
  foreach(_file IN LISTS _files)
    if(NOT IS_ABSOLUTE "${_file}" OR NOT EXISTS "${_file}" OR IS_DIRECTORY "${_file}")
      set(${output} "" PARENT_SCOPE)
      set(${reason} "Generic provider primary and extra libraries must be existing absolute files: ${_file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${output} "${_files}" PARENT_SCOPE)
  set(${reason} "" PARENT_SCOPE)
endfunction()

function(gpu_suite_find_cblas)
  set(GPU_SUITE_CBLAS_FOUND FALSE PARENT_SCOPE)
  set(GPU_SUITE_CBLAS_REASON "CBLAS provider probe was not run" PARENT_SCOPE)
  set(_provider "${GPU_SUITE_CPU_BLAS_BACKEND}")
  if(NOT _provider MATCHES "^(ONEMKL|OPENBLAS|GENERIC_CBLAS)$")
    set(GPU_SUITE_CBLAS_REASON "invalid GPU_SUITE_CPU_BLAS_BACKEND"
        PARENT_SCOPE)
    return()
  endif()
  if(_provider STREQUAL "ONEMKL")
    set(_include_var GPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR)
    set(_library_var GPU_SUITE_ONEMKL_CBLAS_LIBRARY)
    set(_header mkl_cblas.h)
    set(_library_names mkl_rt)
    set(_library_pattern "(^|lib)mkl_rt([.].*)?$")
  elseif(_provider STREQUAL "OPENBLAS")
    set(_include_var GPU_SUITE_OPENBLAS_CBLAS_INCLUDE_DIR)
    set(_library_var GPU_SUITE_OPENBLAS_CBLAS_LIBRARY)
    set(_header cblas.h)
    set(_library_names openblas)
    set(_library_pattern "openblas")
  else()
    set(_include_var GPU_SUITE_GENERIC_CBLAS_INCLUDE_DIR)
    set(_library_var GPU_SUITE_GENERIC_CBLAS_LIBRARY)
    set(_header cblas.h)
    set(_library_names cblas)
    set(_library_pattern "(^|lib)cblas([.].*)?$")
  endif()
  if(_provider STREQUAL "ONEMKL")
    find_path(${_include_var} NAMES "${_header}"
      HINTS ENV MKLROOT
      PATH_SUFFIXES include)
    find_library(${_library_var} NAMES ${_library_names}
      HINTS ENV MKLROOT
      PATH_SUFFIXES lib lib64)
  else()
    find_path(${_include_var} NAMES "${_header}")
    find_library(${_library_var} NAMES ${_library_names})
  endif()
  set(_include_dir "${${_include_var}}")
  set(_library "${${_library_var}}")
  if(NOT _include_dir OR NOT _library OR
     NOT EXISTS "${_include_dir}/${_header}")
    set(GPU_SUITE_CBLAS_REASON
        "${_provider} CBLAS header or provider library was not found"
        PARENT_SCOPE)
    return()
  endif()
  get_filename_component(_library_filename "${_library}" NAME)
  if(NOT _provider STREQUAL "GENERIC_CBLAS" AND
     NOT _library_filename MATCHES "${_library_pattern}")
    set(GPU_SUITE_CBLAS_REASON
        "${_provider} CBLAS library path belongs to a different provider"
        PARENT_SCOPE)
    return()
  endif()
  set(_provider_libraries "${_library}")
  if(_provider STREQUAL "GENERIC_CBLAS")
    _gpu_suite_generic_link_files(_provider_libraries _invalid_library "${_library}"
      GPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES)
    if(_invalid_library)
      set(GPU_SUITE_CBLAS_REASON "${_invalid_library}" PARENT_SCOPE)
      return()
    endif()
  endif()
  _gpu_suite_system_link_libraries(_system_libraries)
  _gpu_suite_cpu_header_preamble(_header_preamble)
  set(_source "${_header_preamble}#include <${_header}>\nint main(void) { double a[1]={1.0}, b[1]={1.0}, c[1]={0.0}; cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0,a,1,b,1,0.0,c,1); return c[0] == 1.0 ? 0 : 1; }")
  gpu_suite_check_c_link(GPU_SUITE_CBLAS_LINKS
    SOURCE "${_source}"
    INCLUDES "${_include_dir}"
    LIBRARIES ${_provider_libraries} ${_system_libraries})
  if(NOT GPU_SUITE_CBLAS_LINKS)
    set(GPU_SUITE_CBLAS_REASON
        "${_provider} cblas_dgemm did not compile and link" PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::CBLAS)
    add_library(GpuSuite::CBLAS INTERFACE IMPORTED)
    set_target_properties(GpuSuite::CBLAS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_include_dir}"
      INTERFACE_LINK_LIBRARIES
        "${_provider_libraries};${_system_libraries}"
      INTERFACE_COMPILE_DEFINITIONS
        "GPU_SUITE_CBLAS_HEADER=\"${_header}\"")
  endif()
  set(GPU_SUITE_CBLAS_PROVIDER "${_provider}" PARENT_SCOPE)
  set(GPU_SUITE_CBLAS_FOUND TRUE PARENT_SCOPE)
  set(GPU_SUITE_CBLAS_REASON "" PARENT_SCOPE)
endfunction()

function(gpu_suite_find_lapacke)
  set(GPU_SUITE_LAPACKE_FOUND FALSE PARENT_SCOPE)
  set(GPU_SUITE_LAPACKE_REASON "LAPACKE provider probe was not run"
      PARENT_SCOPE)
  set(_provider "${GPU_SUITE_CPU_LAPACK_BACKEND}")
  if(NOT _provider MATCHES "^(ONEMKL|OPENBLAS|GENERIC_LAPACKE)$")
    set(GPU_SUITE_LAPACKE_REASON "invalid GPU_SUITE_CPU_LAPACK_BACKEND"
        PARENT_SCOPE)
    return()
  endif()
  if(_provider STREQUAL "ONEMKL")
    set(_include_var GPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR)
    set(_library_var GPU_SUITE_ONEMKL_LAPACKE_LIBRARY)
    set(_header mkl_lapacke.h)
    set(_library_names mkl_rt)
    set(_library_pattern "(^|lib)mkl_rt([.].*)?$")
  elseif(_provider STREQUAL "OPENBLAS")
    set(_include_var GPU_SUITE_OPENBLAS_LAPACKE_INCLUDE_DIR)
    set(_library_var GPU_SUITE_OPENBLAS_LAPACKE_LIBRARY)
    set(_header lapacke.h)
    set(_library_names openblas)
    set(_library_pattern "openblas")
  else()
    set(_include_var GPU_SUITE_GENERIC_LAPACKE_INCLUDE_DIR)
    set(_library_var GPU_SUITE_GENERIC_LAPACKE_LIBRARY)
    set(_header lapacke.h)
    set(_library_names lapacke)
    set(_library_pattern "(^|lib)lapacke([.].*)?$")
  endif()
  if(_provider STREQUAL "ONEMKL")
    find_path(${_include_var} NAMES "${_header}"
      HINTS ENV MKLROOT
      PATH_SUFFIXES include)
    find_library(${_library_var} NAMES ${_library_names}
      HINTS ENV MKLROOT
      PATH_SUFFIXES lib lib64)
  else()
    find_path(${_include_var} NAMES "${_header}")
    find_library(${_library_var} NAMES ${_library_names})
  endif()
  set(_include_dir "${${_include_var}}")
  set(_library "${${_library_var}}")
  if(NOT _include_dir OR NOT _library OR
     NOT EXISTS "${_include_dir}/${_header}")
    set(GPU_SUITE_LAPACKE_REASON
        "${_provider} LAPACKE header or provider library was not found"
        PARENT_SCOPE)
    return()
  endif()
  get_filename_component(_library_filename "${_library}" NAME)
  if(NOT _provider STREQUAL "GENERIC_LAPACKE" AND
     NOT _library_filename MATCHES "${_library_pattern}")
    set(GPU_SUITE_LAPACKE_REASON
        "${_provider} LAPACKE library path belongs to a different provider"
        PARENT_SCOPE)
    return()
  endif()
  set(_provider_libraries "${_library}")
  if(_provider STREQUAL "GENERIC_LAPACKE")
    _gpu_suite_generic_link_files(_provider_libraries _invalid_library "${_library}"
      GPU_SUITE_GENERIC_LAPACKE_EXTRA_LIBRARIES)
    if(_invalid_library)
      set(GPU_SUITE_LAPACKE_REASON "${_invalid_library}" PARENT_SCOPE)
      return()
    endif()
  endif()
  _gpu_suite_system_link_libraries(_system_libraries)
  _gpu_suite_cpu_header_preamble(_header_preamble)
  set(_source "${_header_preamble}#include <${_header}>\nint main(void) { double a[4]={2.0,1.0,1.0,2.0}, b[2]={3.0,3.0}; lapack_int ipiv[2]; lapack_int info1=LAPACKE_dgesv(LAPACK_COL_MAJOR,2,1,a,2,ipiv,b,2); a[0]=2.0;a[1]=1.0;a[2]=1.0;a[3]=2.0;b[0]=3.0;b[1]=3.0; lapack_int info2=LAPACKE_dgetrf(LAPACK_COL_MAJOR,2,2,a,2,ipiv); lapack_int info3=LAPACKE_dgetrs(LAPACK_COL_MAJOR,'N',2,1,a,2,ipiv,b,2); return (int)(info1+info2+info3); }")
  gpu_suite_check_c_link(GPU_SUITE_LAPACKE_LINKS
    SOURCE "${_source}"
    INCLUDES "${_include_dir}"
    LIBRARIES ${_provider_libraries} ${_system_libraries})
  if(NOT GPU_SUITE_LAPACKE_LINKS)
    set(GPU_SUITE_LAPACKE_REASON
        "${_provider} LAPACKE_dgesv/dgetrf/dgetrs did not compile and link"
        PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::LAPACKE)
    add_library(GpuSuite::LAPACKE INTERFACE IMPORTED)
    set_target_properties(GpuSuite::LAPACKE PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_include_dir}"
      INTERFACE_LINK_LIBRARIES
        "${_provider_libraries};${_system_libraries}"
      INTERFACE_COMPILE_DEFINITIONS
        "GPU_SUITE_LAPACKE_HEADER=\"${_header}\"")
  endif()
  set(GPU_SUITE_LAPACKE_PROVIDER "${_provider}" PARENT_SCOPE)
  set(GPU_SUITE_LAPACKE_FOUND TRUE PARENT_SCOPE)
  set(GPU_SUITE_LAPACKE_REASON "" PARENT_SCOPE)
endfunction()

function(gpu_suite_find_onemkl_sparse)
  set(GPU_SUITE_MKL_SPARSE_FOUND FALSE PARENT_SCOPE)
  set(GPU_SUITE_MKL_SPARSE_REASON "oneMKL Sparse probe was not run"
      PARENT_SCOPE)
  if(NOT GPU_SUITE_CPU_SPARSE_BACKEND STREQUAL "ONEMKL")
    set(GPU_SUITE_MKL_SPARSE_REASON
        "GPU_SUITE_CPU_SPARSE_BACKEND is not ONEMKL" PARENT_SCOPE)
    return()
  endif()
  find_path(GPU_SUITE_MKL_SPARSE_INCLUDE_DIR NAMES mkl_spblas.h
    HINTS ENV MKLROOT
    PATH_SUFFIXES include)
  find_library(GPU_SUITE_MKL_SPARSE_LIBRARY NAMES mkl_rt
    HINTS ENV MKLROOT
    PATH_SUFFIXES lib lib64)
  if(NOT GPU_SUITE_MKL_SPARSE_INCLUDE_DIR OR
     NOT GPU_SUITE_MKL_SPARSE_LIBRARY)
    set(GPU_SUITE_MKL_SPARSE_REASON
        "oneMKL mkl_spblas.h or mkl_rt was not found" PARENT_SCOPE)
    return()
  endif()
  _gpu_suite_system_link_libraries(_system_libraries)
  set(_source [=[
#include <mkl_spblas.h>
int main(void) {
  MKL_INT row_start[1]={0}, row_end[1]={1}, col[1]={0};
  double value[1]={2.0}, x[1]={3.0}, y[1]={0.0};
  sparse_matrix_t matrix = 0;
  struct matrix_descr descriptor;
  descriptor.type = SPARSE_MATRIX_TYPE_GENERAL;
  descriptor.mode = SPARSE_FILL_MODE_FULL;
  descriptor.diag = SPARSE_DIAG_NON_UNIT;
  if (mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, 1, 1,
      row_start, row_end, col, value) != SPARSE_STATUS_SUCCESS) return 1;
  if (mkl_sparse_set_mv_hint(matrix, SPARSE_OPERATION_NON_TRANSPOSE,
      descriptor, 1) != SPARSE_STATUS_SUCCESS) return 2;
  if (mkl_sparse_optimize(matrix) != SPARSE_STATUS_SUCCESS) return 3;
  if (mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix,
      descriptor, x, 0.0, y) != SPARSE_STATUS_SUCCESS) return 4;
  return mkl_sparse_destroy(matrix) == SPARSE_STATUS_SUCCESS ? 0 : 5;
}
]=])
  gpu_suite_check_c_link(GPU_SUITE_MKL_SPARSE_LINKS
    SOURCE "${_source}"
    INCLUDES "${GPU_SUITE_MKL_SPARSE_INCLUDE_DIR}"
    LIBRARIES "${GPU_SUITE_MKL_SPARSE_LIBRARY}" ${_system_libraries})
  if(NOT GPU_SUITE_MKL_SPARSE_LINKS)
    set(GPU_SUITE_MKL_SPARSE_REASON
        "oneMKL descriptor/create/hint/optimize/SpMV/destroy API did not compile and link"
        PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::MKLSparse)
    add_library(GpuSuite::MKLSparse INTERFACE IMPORTED)
    set_target_properties(GpuSuite::MKLSparse PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GPU_SUITE_MKL_SPARSE_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES
        "${GPU_SUITE_MKL_SPARSE_LIBRARY};${_system_libraries}")
  endif()
  set(GPU_SUITE_MKL_SPARSE_FOUND TRUE PARENT_SCOPE)
  set(GPU_SUITE_MKL_SPARSE_REASON "" PARENT_SCOPE)
endfunction()
