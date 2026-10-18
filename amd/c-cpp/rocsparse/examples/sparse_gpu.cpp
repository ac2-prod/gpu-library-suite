#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int make_poisson2d_csr(int nx, int ny, int *row, int *col, double *val) {
  int off = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int p = iy * nx + ix;
      row[p] = off;
      if (iy > 0) {
        col[off] = p - nx;
        val[off++] = -1;
      }
      if (ix > 0) {
        col[off] = p - 1;
        val[off++] = -1;
      }
      col[off] = p;
      val[off++] = 4;
      if (ix + 1 < nx) {
        col[off] = p + 1;
        val[off++] = -1;
      }
      if (iy + 1 < ny) {
        col[off] = p + nx;
        val[off++] = -1;
      }
    }
  row[nx * ny] = off;
  return off;
}

int main() {
  const int nx = 1024, ny = 1024, n = nx * ny, nnz = 5 * n - 2 * nx - 2 * ny;
  const double alpha = 1, beta = 1;
  std::vector<int> row(n + 1), col(nnz);
  std::vector<double> val(nnz), x(n, 1), y(n, 1);
  if (make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data()) != nnz)
    return EXIT_FAILURE;
  int *dr = nullptr, *dc = nullptr;
  double *dv = nullptr, *dx = nullptr, *dy = nullptr;
  void *workspace = nullptr;
  size_t workspace_size = 0;
  rocsparse_handle handle = nullptr;
  rocsparse_spmat_descr matrix = nullptr;
  rocsparse_dnvec_descr vecx = nullptr, vecy = nullptr;
  int status = EXIT_FAILURE;
  if (hipMalloc((void **)&dr, row.size() * sizeof(int)) != hipSuccess ||
      hipMalloc((void **)&dc, col.size() * sizeof(int)) != hipSuccess ||
      hipMalloc((void **)&dv, val.size() * sizeof(double)) != hipSuccess ||
      hipMalloc((void **)&dx, x.size() * sizeof(double)) != hipSuccess ||
      hipMalloc((void **)&dy, y.size() * sizeof(double)) != hipSuccess)
    goto cleanup;
  if (hipMemcpy(dr, row.data(), row.size() * sizeof(int),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dc, col.data(), col.size() * sizeof(int),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dv, val.data(), val.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dx, x.data(), x.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dy, y.data(), y.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess)
    goto cleanup;
  if (rocsparse_create_handle(&handle) != rocsparse_status_success ||
      rocsparse_create_csr_descr(&matrix, n, n, nnz, dr, dc, dv, rocsparse_indextype_i32,
                        rocsparse_indextype_i32, rocsparse_index_base_zero,
                        rocsparse_datatype_f64_r) != rocsparse_status_success ||
      rocsparse_create_dnvec_descr(&vecx, n, dx, rocsparse_datatype_f64_r) !=
          rocsparse_status_success ||
      rocsparse_create_dnvec_descr(&vecy, n, dy, rocsparse_datatype_f64_r) !=
          rocsparse_status_success ||
      rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix, vecx, &beta, vecy,
          rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
          rocsparse_spmv_stage_buffer_size, &workspace_size, nullptr) != rocsparse_status_success)
    goto cleanup;
  if (workspace_size > 0 &&
      hipMalloc(&workspace, workspace_size) != hipSuccess)
    goto cleanup;
  if (
      rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix, vecx, &beta, vecy,
          rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
          rocsparse_spmv_stage_preprocess, &workspace_size, workspace) != rocsparse_status_success ||
      rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix, vecx, &beta, vecy,
          rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive, rocsparse_spmv_stage_compute,
          &workspace_size, workspace) != rocsparse_status_success ||
      hipMemcpy(y.data(), dy, y.size() * sizeof(double),
                 hipMemcpyDeviceToHost) != hipSuccess)
    goto cleanup;
  {
    double max_error = 0;
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix) {
        int neighbors =
            4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
        max_error =
            (std::isfinite(y[iy * nx + ix]) ? std::max(max_error, std::fabs(y[iy * nx + ix] - (5.0 - neighbors))) : INFINITY);
      }
    std::printf("rocSPARSE Poisson SpMV complete; max error = %.17g\n",
                max_error);
    status = max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (workspace)
    if (hipFree(workspace) != hipSuccess) status = EXIT_FAILURE;
  if (vecy)
    if (rocsparse_destroy_dnvec_descr(vecy) != rocsparse_status_success) status = EXIT_FAILURE;
  if (vecx)
    if (rocsparse_destroy_dnvec_descr(vecx) != rocsparse_status_success) status = EXIT_FAILURE;
  if (matrix)
    if (rocsparse_destroy_spmat_descr(matrix) != rocsparse_status_success) status = EXIT_FAILURE;
  if (handle)
    if (rocsparse_destroy_handle(handle) != rocsparse_status_success) status = EXIT_FAILURE;
  if (dy)
    if (hipFree(dy) != hipSuccess) status = EXIT_FAILURE;
  if (dx)
    if (hipFree(dx) != hipSuccess) status = EXIT_FAILURE;
  if (dv)
    if (hipFree(dv) != hipSuccess) status = EXIT_FAILURE;
  if (dc)
    if (hipFree(dc) != hipSuccess) status = EXIT_FAILURE;
  if (dr)
    if (hipFree(dr) != hipSuccess) status = EXIT_FAILURE;
  return status;
}
