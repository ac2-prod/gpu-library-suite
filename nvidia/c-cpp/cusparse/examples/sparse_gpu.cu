#include <cuda_runtime.h>
#include <cusparse.h>

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
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  cusparseDnVecDescr_t vecx = nullptr, vecy = nullptr;
  int status = EXIT_FAILURE;
  if (cudaMalloc((void **)&dr, row.size() * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&dc, col.size() * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&dv, val.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc((void **)&dx, x.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc((void **)&dy, y.size() * sizeof(double)) != cudaSuccess)
    goto cleanup;
  if (cudaMemcpy(dr, row.data(), row.size() * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dc, col.data(), col.size() * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dv, val.data(), val.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dx, x.data(), x.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dy, y.data(), y.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    goto cleanup;
  if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS ||
      cusparseCreateCsr(&matrix, n, n, nnz, dr, dc, dv, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
      cusparseCreateDnVec(&vecx, n, dx, CUDA_R_64F) !=
          CUSPARSE_STATUS_SUCCESS ||
      cusparseCreateDnVec(&vecy, n, dy, CUDA_R_64F) !=
          CUSPARSE_STATUS_SUCCESS ||
      cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              matrix, vecx, &beta, vecy, CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              &workspace_size) != CUSPARSE_STATUS_SUCCESS)
    goto cleanup;
  if (workspace_size > 0 &&
      cudaMalloc(&workspace, workspace_size) != cudaSuccess)
    goto cleanup;
  if (
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix,
                   vecx, &beta, vecy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                   workspace) != CUSPARSE_STATUS_SUCCESS ||
      cudaMemcpy(y.data(), dy, y.size() * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    goto cleanup;
  {
    double max_error = 0;
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix) {
        int neighbors =
            4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
        max_error =
            std::max(max_error, std::fabs(y[iy * nx + ix] - (5.0 - neighbors)));
      }
    std::printf("cuSPARSE Poisson SpMV complete; max error = %.17g\n",
                max_error);
    status = max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (workspace)
    cudaFree(workspace);
  if (vecy)
    cusparseDestroyDnVec(vecy);
  if (vecx)
    cusparseDestroyDnVec(vecx);
  if (matrix)
    cusparseDestroySpMat(matrix);
  if (handle)
    cusparseDestroy(handle);
  if (dy)
    cudaFree(dy);
  if (dx)
    cudaFree(dx);
  if (dv)
    cudaFree(dv);
  if (dc)
    cudaFree(dc);
  if (dr)
    cudaFree(dr);
  return status;
}
