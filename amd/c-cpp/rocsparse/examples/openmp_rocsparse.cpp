#include <hip/hip_runtime.h>
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif
#include <rocsparse/rocsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool mapped_device(const void *pointer) {
  hipPointerAttribute_t attributes{};
  int device = -1;
  return hipGetDevice(&device) == hipSuccess &&
         hipPointerGetAttributes(&attributes, pointer) == hipSuccess &&
         attributes.type == hipMemoryTypeDevice && attributes.device == device;
}

static bool prove_offload() {
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices()) {
    std::fprintf(stderr, "OpenMP target device unavailable\n");
    return false;
  }
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) std::fprintf(stderr, "OpenMP host fallback rejected\n");
  return initial == 0;
}

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
  if (hipSetDevice(0) != hipSuccess || !prove_offload()) return EXIT_FAILURE;
  const int nx = 1024, ny = 1024, n = nx * ny, nnz = 5 * n - 2 * nx - 2 * ny;
  const double alpha = 1, beta = 1;
  std::vector<int> row(n + 1), col(nnz);
  std::vector<double> val(nnz), x(n, 1), y(n, 1);
  make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data());
  int *rp = row.data(), *cp = col.data();
  double *vp = val.data(), *xp = x.data(), *yp = y.data();
  size_t rs = row.size(), cs = col.size(), vs = val.size(), xs = x.size(),
         ys = y.size();
  rocsparse_handle handle = nullptr;
  if (rocsparse_create_handle(&handle) != rocsparse_status_success) {
    std::fprintf(stderr, "rocsparse_create_handle failed\n");
    return EXIT_FAILURE;
  }
  bool ok = true;
#pragma omp target data map(to: rp[0 : rs], cp[0 : cs], vp[0 : vs], \
    xp[0 : xs]) map(tofrom: yp[0 : ys])
  {
    rocsparse_spmat_descr matrix = nullptr;
    rocsparse_dnvec_descr vecx = nullptr, vecy = nullptr;
    rocsparse_spmv_descr spmv = nullptr;
    const auto alg = rocsparse_spmv_alg_csr_rowsplit;
    const auto op = rocsparse_operation_none;
    const auto datatype = rocsparse_datatype_f64_r;
    void *workspace = nullptr;
    size_t workspace_size = 0;
#pragma omp target data use_device_ptr(rp, cp, vp, xp, yp)
    {
      ok = ok && (mapped_device(rp) && mapped_device(cp) && mapped_device(vp) && mapped_device(xp) && mapped_device(yp));
      ok = ok && rocsparse_create_csr_descr(&matrix, n, n, nnz, rp, cp, vp, rocsparse_indextype_i32,
                             rocsparse_indextype_i32, rocsparse_index_base_zero,
                             rocsparse_datatype_f64_r) == rocsparse_status_success &&
           rocsparse_create_dnvec_descr(&vecx, n, xp, rocsparse_datatype_f64_r) ==
               rocsparse_status_success &&
           rocsparse_create_dnvec_descr(&vecy, n, yp, rocsparse_datatype_f64_r) ==
               rocsparse_status_success;
    }
    ok = ok && rocsparse_create_spmv_descr(&spmv) == rocsparse_status_success &&
         rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_alg,
             &alg, sizeof(alg), nullptr) == rocsparse_status_success &&
         rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_operation,
             &op, sizeof(op), nullptr) == rocsparse_status_success &&
         rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_scalar_datatype,
             &datatype, sizeof(datatype), nullptr) == rocsparse_status_success &&
         rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_compute_datatype,
             &datatype, sizeof(datatype), nullptr) == rocsparse_status_success &&
         rocsparse_v2_spmv_buffer_size(handle, spmv, matrix, vecx, vecy,
             rocsparse_v2_spmv_stage_analysis, &workspace_size, nullptr) == rocsparse_status_success;
    if (ok && workspace_size > 0)
      ok = hipMalloc(&workspace, workspace_size) == hipSuccess;
    if (ok)
      ok = rocsparse_v2_spmv(handle, spmv, &alpha, matrix, vecx, &beta, vecy,
          rocsparse_v2_spmv_stage_analysis, workspace_size, workspace, nullptr) == rocsparse_status_success;
    if (workspace) {
      ok = hipFree(workspace) == hipSuccess && ok;
      workspace = nullptr;
    }
    ok = ok && rocsparse_v2_spmv_buffer_size(handle, spmv, matrix, vecx, vecy,
        rocsparse_v2_spmv_stage_compute, &workspace_size, nullptr) == rocsparse_status_success;
    if (ok && workspace_size > 0)
      ok = hipMalloc(&workspace, workspace_size) == hipSuccess;
    if (ok)
      ok = rocsparse_v2_spmv(handle, spmv, &alpha, matrix, vecx, &beta, vecy,
          rocsparse_v2_spmv_stage_compute, workspace_size, workspace, nullptr) == rocsparse_status_success;
    ok = hipDeviceSynchronize() == hipSuccess && ok;
    if (workspace)
      ok = hipFree(workspace) == hipSuccess && ok;
    if (spmv)
      ok = rocsparse_destroy_spmv_descr(spmv) == rocsparse_status_success && ok;
    if (vecy)
      ok = rocsparse_destroy_dnvec_descr(vecy) == rocsparse_status_success && ok;
    if (vecx)
      ok = rocsparse_destroy_dnvec_descr(vecx) == rocsparse_status_success && ok;
    if (matrix)
      ok = rocsparse_destroy_spmat_descr(matrix) == rocsparse_status_success && ok;
  }
  ok = rocsparse_destroy_handle(handle) == rocsparse_status_success && ok;
  double max_error = 0;
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int neighbors =
          4 - (ix == 0) - (ix + 1 == nx) - (iy == 0) - (iy + 1 == ny);
      max_error =
          (std::isfinite(y[iy * nx + ix]) ? std::max(max_error, std::fabs(y[iy * nx + ix] - (5.0 - neighbors))) : INFINITY);
    }
  std::printf("OpenMP-managed rocSPARSE SpMV complete; max error = %.17g\n",
              max_error);
  return ok && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}
