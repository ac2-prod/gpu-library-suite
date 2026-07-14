#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void make_dense_system(int n, int nrhs, double *a, double *b) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? n + 1.0 : 1.0;
  for (int r = 0; r < nrhs; ++r)
    for (int i = 0; i < n; ++i)
      b[i + (size_t)r * n] = 2.0 * n;
}

int main() {
  const int n = 1024, nrhs = 16;
  std::vector<double> a((size_t)n * n), b((size_t)n * nrhs);
  make_dense_system(n, nrhs, a.data(), b.data());
  double *da = nullptr, *db = nullptr, *work = nullptr;
  int *piv = nullptr, *info = nullptr;
  int lwork = 0;
  cusolverDnHandle_t handle = nullptr;
  int status = EXIT_FAILURE;
  if (cudaMalloc((void **)&da, a.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc((void **)&db, b.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc((void **)&piv, n * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&info, sizeof(int)) != cudaSuccess)
    goto cleanup;
  if (cudaMemcpy(da, a.data(), a.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(db, b.data(), b.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS ||
      cusolverDnDgetrf_bufferSize(handle, n, n, da, n, &lwork) !=
          CUSOLVER_STATUS_SUCCESS ||
      cudaMalloc((void **)&work, (size_t)lwork * sizeof(double)) !=
          cudaSuccess ||
      cusolverDnDgetrf(handle, n, n, da, n, work, piv, info) !=
          CUSOLVER_STATUS_SUCCESS ||
      cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs, da, n, piv, db, n, info) !=
          CUSOLVER_STATUS_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(b.data(), db, b.size() * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    goto cleanup;
  {
    int host_info = 0;
    cudaMemcpy(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost);
    double max_error = 0;
    for (double x : b)
      max_error = std::max(max_error, std::fabs(x - 1));
    std::printf("cuSOLVER getrf/getrs info=%d, max error=%.17g\n", host_info,
                max_error);
    status = host_info == 0 && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (work)
    cudaFree(work);
  if (info)
    cudaFree(info);
  if (piv)
    cudaFree(piv);
  if (db)
    cudaFree(db);
  if (da)
    cudaFree(da);
  if (handle)
    cusolverDnDestroy(handle);
  return status;
}
