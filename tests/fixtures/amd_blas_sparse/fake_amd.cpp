// Synthetic host numerical and failure fixture. No GPU or OpenMP runtime calls.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

static bool requested(const char *key) { return std::getenv(key) != nullptr; }
static bool operation_failure(int count) {
  const char *after = std::getenv("FAKE_FAIL_AFTER");
  return requested("FAKE_LIBRARY_FAIL") || (after && count > std::atoi(after));
}
static void corrupt(double *output) {
  const char *s = std::getenv("GPU_SUITE_TEST_NONFINITE");
  if (s) *output = std::strcmp(s, "nan") == 0 ? NAN : INFINITY;
}
hipError_t hipMalloc(void **p, size_t n) {
  if (requested("FAKE_ALLOC_FAIL")) return static_cast<hipError_t>(1);
  *p = std::malloc(n ? n : 1);
  return *p ? hipSuccess : static_cast<hipError_t>(1);
}
hipError_t hipFree(void *p) { std::free(p); return hipSuccess; }
hipError_t hipMemcpy(void *d, const void *s, size_t n, hipMemcpyKind) {
  std::memcpy(d, s, n); return hipSuccess;
}
hipError_t hipDeviceSynchronize() { return hipSuccess; }
hipError_t hipSetDevice(int) { return hipSuccess; }
hipError_t hipGetDevice(int *d) { *d = 0; return hipSuccess; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *a, const void *) {
  a->type = hipMemoryTypeDevice;
  a->device = requested("FAKE_DEVICE_MISMATCH") ? 1 : 0;
  return hipSuccess;
}
const char *hipGetErrorString(hipError_t) { return "synthetic HIP failure"; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
  std::strcpy(p->name, "host fixture, not GPU"); return hipSuccess;
}
hipError_t hipRuntimeGetVersion(int *v) { *v = 70200000; return hipSuccess; }
hipError_t hipDriverGetVersion(int *v) { *v = 70200000; return hipSuccess; }
extern "C" int omp_get_num_devices() { return 1; }
extern "C" int omp_get_default_device() { return 0; }
extern "C" int omp_is_initial_device() { return requested("FAKE_HOST_FALLBACK"); }

struct fake_blas_handle {};
rocblas_status rocblas_create_handle(rocblas_handle *h) {
  *h = new fake_blas_handle; return rocblas_status_success;
}
rocblas_status rocblas_destroy_handle(rocblas_handle h) {
  delete h; return rocblas_status_success;
}
rocblas_status rocblas_get_version_string(char *s, size_t n) {
  const char value[] = "fixture-rocblas";
  if (n < sizeof(value)) return rocblas_status_internal_error;
  std::memcpy(s, value, sizeof(value)); return rocblas_status_success;
}
rocblas_status rocblas_dgemm(rocblas_handle, rocblas_operation,
    rocblas_operation, int m, int n, int k, const double *alpha,
    const double *a, int lda, const double *b, int ldb,
    const double *beta, double *c, int ldc) {
  static int calls = 0;
  if (operation_failure(++calls)) return rocblas_status_internal_error;
  for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
    double sum = 0;
    for (int r = 0; r < k; ++r) sum += a[i + r*lda] * b[r + j*ldb];
    c[i + j*ldc] = *alpha * sum + *beta * c[i + j*ldc];
  }
  corrupt(c); return rocblas_status_success;
}
struct fake_sparse_handle {};
struct fake_matrix { int n; int *row, *column; double *value; bool analyzed; };
struct fake_vector { int n; double *data; };
struct fake_spmv {
  unsigned inputs = 0;
  bool analysis_queried = false, analyzed = false, compute_queried = false;
};
rocsparse_status rocsparse_create_handle(rocsparse_handle *h) {
  *h = new fake_sparse_handle; return rocsparse_status_success;
}
rocsparse_status rocsparse_destroy_handle(rocsparse_handle h) {
  delete h; return rocsparse_status_success;
}
rocsparse_status rocsparse_get_version(rocsparse_handle, int *v) {
  *v = 40200; return rocsparse_status_success;
}
rocsparse_status rocsparse_get_git_rev(rocsparse_handle, char *s) {
  std::strcpy(s, "fixture"); return rocsparse_status_success;
}
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr *a, int64_t m,
    int64_t, int64_t, void *r, void *c, void *v, rocsparse_indextype,
    rocsparse_indextype, rocsparse_index_base, rocsparse_datatype) {
  *a = new fake_matrix{static_cast<int>(m), static_cast<int *>(r),
      static_cast<int *>(c), static_cast<double *>(v), false};
  return rocsparse_status_success;
}
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr *v, int64_t n,
    void *d, rocsparse_datatype) {
  *v = new fake_vector{static_cast<int>(n), static_cast<double *>(d)};
  return rocsparse_status_success;
}
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_spmat_descr a) {
  delete a; return rocsparse_status_success;
}
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_dnvec_descr v) {
  delete v; return rocsparse_status_success;
}
rocsparse_status rocsparse_create_spmv_descr(rocsparse_spmv_descr *d) {
  *d = new fake_spmv; return rocsparse_status_success;
}
rocsparse_status rocsparse_destroy_spmv_descr(rocsparse_spmv_descr d) {
  delete d; return rocsparse_status_success;
}
rocsparse_status rocsparse_spmv_set_input(rocsparse_handle, rocsparse_spmv_descr d,
    rocsparse_spmv_input input, const void *value, size_t bytes, rocsparse_error *) {
  if (!d || !value || input < 0 || input > 3 || bytes != sizeof(int) ||
      *static_cast<const int *>(value) != 0) return rocsparse_status_internal_error;
  d->inputs |= 1u << input;
  return rocsparse_status_success;
}
rocsparse_status rocsparse_v2_spmv_buffer_size(rocsparse_handle, rocsparse_spmv_descr d,
    rocsparse_const_spmat_descr, rocsparse_const_dnvec_descr,
    rocsparse_const_dnvec_descr, rocsparse_v2_spmv_stage stage, size_t *bytes, rocsparse_error *) {
  if (!d || d->inputs != 15) return rocsparse_status_internal_error;
  if (stage == rocsparse_v2_spmv_stage_analysis) {
    d->analysis_queried = true; *bytes = 64;
  } else {
    if (!d->analyzed) return rocsparse_status_internal_error;
    d->compute_queried = true; *bytes = 128;
  }
  return rocsparse_status_success;
}
rocsparse_status rocsparse_v2_spmv(rocsparse_handle, rocsparse_spmv_descr d,
    const void *alpha, rocsparse_const_spmat_descr a,
    rocsparse_const_dnvec_descr x, const void *beta, rocsparse_dnvec_descr y,
    rocsparse_v2_spmv_stage stage, size_t bytes, void *workspace, rocsparse_error *) {
  if (!d || d->inputs != 15 || !workspace) return rocsparse_status_internal_error;
  if (stage == rocsparse_v2_spmv_stage_analysis) {
    if (!d->analysis_queried || bytes != 64) return rocsparse_status_internal_error;
    d->analyzed = true;
    return rocsparse_status_success;
  }
  static int calls = 0;
  if (!d->analyzed || !d->compute_queried || bytes != 128 ||
      operation_failure(++calls)) return rocsparse_status_internal_error;
  for (int i = 0; i < a->n; ++i) {
    double sum = 0;
    for (int j = a->row[i]; j < a->row[i+1]; ++j)
      sum += a->value[j] * x->data[a->column[j]];
    y->data[i] = *static_cast<const double *>(alpha) * sum +
                *static_cast<const double *>(beta) * y->data[i];
  }
  corrupt(y->data); return rocsparse_status_success;
}
