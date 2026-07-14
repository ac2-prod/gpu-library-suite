include(GpuSuiteProbe)

function(_gpu_suite_provider_library_names provider kind output)
  if(provider STREQUAL "ONEMKL")
    set(_names mkl_rt)
  elseif(provider STREQUAL "OPENBLAS")
    set(_names openblas)
  elseif(kind STREQUAL "CBLAS")
    set(_names cblas openblas blas)
  else()
    set(_names lapacke openblas)
  endif()
  set(${output} "${_names}" PARENT_SCOPE)
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

function(gpu_suite_find_cblas)
  set(GPU_SUITE_CBLAS_FOUND FALSE PARENT_SCOPE)
  set(GPU_SUITE_CBLAS_REASON "CBLAS provider probe was not run" PARENT_SCOPE)
  set(_provider "${GPU_SUITE_CPU_BLAS_BACKEND}")
  if(NOT _provider MATCHES "^(ONEMKL|OPENBLAS|GENERIC_CBLAS)$")
    set(GPU_SUITE_CBLAS_REASON "invalid GPU_SUITE_CPU_BLAS_BACKEND"
        PARENT_SCOPE)
    return()
  endif()
  find_path(GPU_SUITE_CBLAS_INCLUDE_DIR NAMES cblas.h mkl_cblas.h)
  _gpu_suite_provider_library_names("${_provider}" CBLAS _library_names)
  find_library(GPU_SUITE_CBLAS_LIBRARY NAMES ${_library_names})
  if(NOT GPU_SUITE_CBLAS_INCLUDE_DIR OR NOT GPU_SUITE_CBLAS_LIBRARY)
    set(GPU_SUITE_CBLAS_REASON
        "${_provider} CBLAS header or provider library was not found"
        PARENT_SCOPE)
    return()
  endif()
  if(EXISTS "${GPU_SUITE_CBLAS_INCLUDE_DIR}/cblas.h")
    set(_header cblas.h)
  else()
    set(_header mkl_cblas.h)
  endif()
  _gpu_suite_system_link_libraries(_system_libraries)
  set(_source "#include <${_header}>\nint main(void) { double a[1]={1.0}, b[1]={1.0}, c[1]={0.0}; cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0,a,1,b,1,0.0,c,1); return c[0] == 1.0 ? 0 : 1; }")
  gpu_suite_check_c_link(GPU_SUITE_CBLAS_LINKS
    SOURCE "${_source}"
    INCLUDES "${GPU_SUITE_CBLAS_INCLUDE_DIR}"
    LIBRARIES "${GPU_SUITE_CBLAS_LIBRARY}" ${_system_libraries})
  if(NOT GPU_SUITE_CBLAS_LINKS)
    set(GPU_SUITE_CBLAS_REASON
        "${_provider} cblas_dgemm did not compile and link" PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::CBLAS)
    add_library(GpuSuite::CBLAS INTERFACE IMPORTED)
    set_target_properties(GpuSuite::CBLAS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GPU_SUITE_CBLAS_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES
        "${GPU_SUITE_CBLAS_LIBRARY};${_system_libraries}"
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
  find_path(GPU_SUITE_LAPACKE_INCLUDE_DIR NAMES lapacke.h mkl_lapacke.h)
  _gpu_suite_provider_library_names("${_provider}" LAPACKE _library_names)
  find_library(GPU_SUITE_LAPACKE_LIBRARY NAMES ${_library_names})
  if(NOT GPU_SUITE_LAPACKE_INCLUDE_DIR OR NOT GPU_SUITE_LAPACKE_LIBRARY)
    set(GPU_SUITE_LAPACKE_REASON
        "${_provider} LAPACKE header or provider library was not found"
        PARENT_SCOPE)
    return()
  endif()
  if(EXISTS "${GPU_SUITE_LAPACKE_INCLUDE_DIR}/lapacke.h")
    set(_header lapacke.h)
  else()
    set(_header mkl_lapacke.h)
  endif()
  _gpu_suite_system_link_libraries(_system_libraries)
  set(_source "#include <${_header}>\nint main(void) { double a[4]={2.0,1.0,1.0,2.0}, b[2]={3.0,3.0}; lapack_int ipiv[2]; lapack_int info1=LAPACKE_dgesv(LAPACK_COL_MAJOR,2,1,a,2,ipiv,b,2); a[0]=2.0;a[1]=1.0;a[2]=1.0;a[3]=2.0;b[0]=3.0;b[1]=3.0; lapack_int info2=LAPACKE_dgetrf(LAPACK_COL_MAJOR,2,2,a,2,ipiv); lapack_int info3=LAPACKE_dgetrs(LAPACK_COL_MAJOR,'N',2,1,a,2,ipiv,b,2); return (int)(info1+info2+info3); }")
  gpu_suite_check_c_link(GPU_SUITE_LAPACKE_LINKS
    SOURCE "${_source}"
    INCLUDES "${GPU_SUITE_LAPACKE_INCLUDE_DIR}"
    LIBRARIES "${GPU_SUITE_LAPACKE_LIBRARY}" ${_system_libraries})
  if(NOT GPU_SUITE_LAPACKE_LINKS)
    set(GPU_SUITE_LAPACKE_REASON
        "${_provider} LAPACKE_dgesv/dgetrf/dgetrs did not compile and link"
        PARENT_SCOPE)
    return()
  endif()
  if(NOT TARGET GpuSuite::LAPACKE)
    add_library(GpuSuite::LAPACKE INTERFACE IMPORTED)
    set_target_properties(GpuSuite::LAPACKE PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GPU_SUITE_LAPACKE_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES
        "${GPU_SUITE_LAPACKE_LIBRARY};${_system_libraries}"
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
  find_path(GPU_SUITE_MKL_SPARSE_INCLUDE_DIR NAMES mkl_spblas.h)
  find_library(GPU_SUITE_MKL_SPARSE_LIBRARY NAMES mkl_rt)
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
