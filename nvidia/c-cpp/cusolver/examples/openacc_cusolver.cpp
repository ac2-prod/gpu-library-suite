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
  std::vector<int> piv(n), getrf_info(1, -1), getrs_info(1, -1);
  make_dense_system(n, nrhs, a.data(), b.data());
  double *ap = a.data(), *bp = b.data();
  int *pp = piv.data(), *getrf_ip = getrf_info.data();
  int *getrs_ip = getrs_info.data();
  size_t ac = a.size(), bc = b.size();
  cusolverDnHandle_t handle = nullptr;
  if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
    std::fprintf(stderr, "cusolverDnCreate failed\n");
    return EXIT_FAILURE;
  }
  int lwork = 0;
  bool ok = true;
#pragma acc data copy(ap[0 : ac], bp[0 : bc]) create(pp[0 : n],              \
                                                     getrf_ip[0 : 1],         \
                                                     getrs_ip[0 : 1])
  {
#pragma acc host_data use_device(ap)
    {
      ok = cusolverDnDgetrf_bufferSize(handle, n, n, ap, n, &lwork) ==
           CUSOLVER_STATUS_SUCCESS;
    }
    double *device_work = nullptr;
    if (ok)
      ok = cudaMalloc((void **)&device_work, (size_t)lwork * sizeof(double)) ==
           cudaSuccess;
#pragma acc host_data use_device(ap, bp, pp, getrf_ip, getrs_ip)
    {
      if (ok)
        ok = cusolverDnDgetrf(handle, n, n, ap, n, device_work, pp,
                             getrf_ip) ==
                 CUSOLVER_STATUS_SUCCESS &&
             cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs, ap, n, pp, bp, n,
                              getrs_ip) == CUSOLVER_STATUS_SUCCESS;
    }
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
#pragma acc update self(getrf_ip[0 : 1], getrs_ip[0 : 1])
    if (device_work)
      cudaFree(device_work);
  }
  ok = ok && cusolverDnDestroy(handle) == CUSOLVER_STATUS_SUCCESS;
  double max_error = 0;
  for (double x : b)
    max_error = std::max(max_error, std::fabs(x - 1));
  std::printf("OpenACC-managed cuSOLVER solve complete; getrf info=%d, "
              "getrs info=%d, max error=%.17g\n",
              getrf_info[0], getrs_info[0], max_error);
  return ok && getrf_info[0] == 0 && getrs_info[0] == 0 &&
                 max_error <= 1e-12
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
