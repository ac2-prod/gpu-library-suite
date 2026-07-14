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
  make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data());
  int *rp = row.data(), *cp = col.data();
  double *vp = val.data(), *xp = x.data(), *yp = y.data();
  size_t rs = row.size(), cs = col.size(), vs = val.size(), xs = x.size(),
         ys = y.size();
  cusparseHandle_t handle = nullptr;
  cusparseCreate(&handle);
  bool ok = true;
#pragma acc data copyin(rp[0 : rs], cp[0 : cs], vp[0 : vs], xp[0 : xs])        \
    copy(yp[0 : ys])
  {
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t vecx = nullptr, vecy = nullptr;
    void *workspace = nullptr;
    size_t workspace_size = 0;
#pragma acc host_data use_device(rp, cp, vp, xp, yp)
    {
      ok = cusparseCreateCsr(&matrix, n, n, nnz, rp, cp, vp, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                             CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
           cusparseCreateDnVec(&vecx, n, xp, CUDA_R_64F) ==
               CUSPARSE_STATUS_SUCCESS &&
           cusparseCreateDnVec(&vecy, n, yp, CUDA_R_64F) ==
               CUSPARSE_STATUS_SUCCESS;
    }
    ok = ok && cusparseSpMV_bufferSize(
                   handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix,
                   vecx, &beta, vecy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                   &workspace_size) == CUSPARSE_STATUS_SUCCESS;
    if (ok)
      ok = cudaMalloc(&workspace, workspace_size) == cudaSuccess;
    if (ok)
      ok =
          cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix,
                       vecx, &beta, vecy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                       workspace) == CUSPARSE_STATUS_SUCCESS &&
          cudaDeviceSynchronize() == cudaSuccess;
    if (workspace)
      cudaFree(workspace);
    if (vecy)
      cusparseDestroyDnVec(vecy);
    if (vecx)
      cusparseDestroyDnVec(vecx);
    if (matrix)
      cusparseDestroySpMat(matrix);
  }
  ok = ok && cusparseDestroy(handle) == CUSPARSE_STATUS_SUCCESS;
  double max_error = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int neighbors =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      max_error =
          std::max(max_error, std::fabs(y[iy * nx + ix] - (5.0 - neighbors)));
    }
  std::printf("OpenACC-managed cuSPARSE SpMV complete; max error = %.17g\n",
              max_error);
  return ok && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}
