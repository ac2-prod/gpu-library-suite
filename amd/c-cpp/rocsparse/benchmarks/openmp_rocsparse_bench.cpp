#include <cstdint>
#include "gpu_suite/benchmark.hpp"
#include "sparse_bench_common.hpp"

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
#include <exception>
#include <new>
#include <stdexcept>
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

static bool hip_success(hipError_t status, const char *api) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, hipGetErrorString(status));
  return false;
}

static bool rocsparse_success(rocsparse_status status, const char *api) {
  if (status == rocsparse_status_success)
    return true;
  std::fprintf(stderr, "%s: rocSPARSE status %d\n", api,
               static_cast<int>(status));
  return false;
}

static uint64_t root(uint64_t v) {
  uint64_t r = (uint64_t)std::sqrt((long double)v);
  return r;
}
static int make_poisson2d_csr(int nx, int ny, int *r, int *c, double *v) {
  int o = 0;
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      int p = y * nx + x;
      r[p] = o;
      if (y > 0) {
        c[o] = p - nx;
        v[o++] = -1;
      }
      if (x > 0) {
        c[o] = p - 1;
        v[o++] = -1;
      }
      c[o] = p;
      v[o++] = 4;
      if (x + 1 < nx) {
        c[o] = p + 1;
        v[o++] = -1;
      }
      if (y + 1 < ny) {
        c[o] = p + nx;
        v[o++] = -1;
      }
    }
  r[nx * ny] = o;
  return o;
}

static bool run_end_to_end_once(int n, int nnz, int *row_offsets,
                                size_t row_count, int *column_indices,
                                size_t column_count, double *matrix_values,
                                size_t value_count, double *x, size_t x_count,
                                double *y, size_t y_count, double alpha,
                                double beta, double *elapsed, char *error,
                                size_t error_size, bool first_repeat,
                                bool last_repeat,
                                char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
                                char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY]) {
  rocsparse_handle handle = nullptr;
  rocsparse_spmat_descr matrix = nullptr;
  rocsparse_dnvec_descr vector_x = nullptr;
  rocsparse_dnvec_descr vector_y = nullptr;
  void *workspace = nullptr;
  size_t workspace_size = 0;
  struct timespec start;
  struct timespec end;
  bool ok = (first_repeat
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK &&
            rocsparse_success(rocsparse_create_handle(&handle), "rocsparse_create_handle");

#pragma omp target data map(to: row_offsets[0 : row_count], column_indices[0 : column_count], \
    matrix_values[0 : value_count], x[0 : x_count]) map(tofrom: y[0 : y_count])
  {
#pragma omp target data use_device_ptr(row_offsets, column_indices, matrix_values, \
                                       x, y)
    {
      ok = ok && (mapped_device(row_offsets) && mapped_device(column_indices) && mapped_device(matrix_values) && mapped_device(x) && mapped_device(y));
      if (ok) {
        ok = rocsparse_success(
                 rocsparse_create_csr_descr(&matrix, n, n, nnz, row_offsets,
                                   column_indices, matrix_values,
                                   rocsparse_indextype_i32, rocsparse_indextype_i32,
                                   rocsparse_index_base_zero, rocsparse_datatype_f64_r),
                 "rocsparse_create_csr_descr") &&
             rocsparse_success(rocsparse_create_dnvec_descr(&vector_x, n, x, rocsparse_datatype_f64_r),
                              "rocsparse_create_dnvec_descr x") &&
             rocsparse_success(rocsparse_create_dnvec_descr(&vector_y, n, y, rocsparse_datatype_f64_r),
                              "rocsparse_create_dnvec_descr y");
      }
    }
    if (ok) {
      ok = rocsparse_success(
               rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix, vector_x, &beta,
                   vector_y, rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_buffer_size, &workspace_size, nullptr),
               "rocsparse_spmv buffer_size") &&
           (workspace_size == 0 ||
            hip_success(hipMalloc(&workspace, workspace_size),
                         "hipMalloc rocSPARSE workspace")) &&
           rocsparse_success(
               rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix, vector_x, &beta,
                   vector_y, rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_preprocess, &workspace_size, workspace),
                   "rocsparse_spmv preprocess") &&
           rocsparse_success(rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matrix,
               vector_x, &beta, vector_y, rocsparse_datatype_f64_r,
               rocsparse_spmv_alg_csr_adaptive, rocsparse_spmv_stage_compute, &workspace_size,
               workspace),
               "rocsparse_spmv compute") &&
           hip_success(hipDeviceSynchronize(),
                        "end-to-end synchronize");
    }
  }
  ok = ok &&
       (last_repeat
            ? gpu_suite_measurement_end(&end, end_timestamp, error, error_size)
            : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
  if (ok)
    *elapsed = gpu_suite_clock_elapsed(&start, &end);

  if (workspace)
    ok = hip_success(hipFree(workspace), "hipFree cleanup") && ok;
  if (vector_y)
    ok = rocsparse_success(rocsparse_destroy_dnvec_descr(vector_y), "rocsparse_destroy_dnvec_descr cleanup") && ok;
  if (vector_x)
    ok = rocsparse_success(rocsparse_destroy_dnvec_descr(vector_x), "rocsparse_destroy_dnvec_descr cleanup") && ok;
  if (matrix)
    ok = rocsparse_success(rocsparse_destroy_spmat_descr(matrix), "rocsparse_destroy_spmat_descr cleanup") && ok;
  if (handle)
    ok = rocsparse_success(rocsparse_destroy_handle(handle), "rocsparse_destroy_handle cleanup") && ok;
  return ok;
}
// Descriptor creation follows data entry and device-address translation; only
// workspace uses hipMalloc.
int main(int argc, char **argv) {
  gpu_suite_options o;
  char e[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_ROCSPARSE,
                         GPU_SUITE_IMPLEMENTATION_OPENMP);
  auto p = gpu_suite_options_parse(&o, argc, argv, e, sizeof(e));
  if (p == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCSPARSE);
    return 0;
  }
  if (p != GPU_SUITE_PARSE_OK)
    return 1;
  gpu_suite::ResultWriter w;
  if (!w.open(o))
    return 1;
  const std::uint64_t nx_value = o.size_set ? root(o.size) : o.nx;
  const std::uint64_t ny_value = o.size_set ? root(o.size) : o.ny;
  int nx = 0, ny = 0, n = 0, nnz = 0;
  std::size_t row_count = 0;
  std::size_t row_bytes = 0, column_bytes = 0, value_bytes = 0,
              vector_bytes = 0;
  if (!gpu_suite_checked_poisson2d_dimensions(
          nx_value, ny_value, &nx, &ny, &n, &nnz, &row_count) ||
      !gpu_suite_checked_bytes(row_count, sizeof(int), &row_bytes) ||
      !gpu_suite_checked_bytes(static_cast<std::size_t>(nnz), sizeof(int),
                               &column_bytes) ||
      !gpu_suite_checked_bytes(static_cast<std::size_t>(nnz), sizeof(double),
                               &value_bytes) ||
      !gpu_suite_checked_bytes(static_cast<std::size_t>(n), sizeof(double),
                               &vector_bytes)) {
    std::fprintf(stderr, "CSR dimensions or byte counts exceed supported range\n");
    gpu_suite::emit_unmeasured(
        w, o, 0, false, "prerequisite",
        "CSR dimensions exceed supported integer or byte range");
    (void)w.close();
    return 1;
  }
  (void)row_bytes;
  (void)column_bytes;
  (void)value_bytes;
  (void)vector_bytes;
  bool fatal_failure = false;
  bool any_failure = false;
  try {
  std::vector<int> row(row_count), col(static_cast<std::size_t>(nnz));
  std::vector<double> val(static_cast<std::size_t>(nnz));
  std::vector<double> x(static_cast<std::size_t>(n), 1.0);
  std::vector<double> y(static_cast<std::size_t>(n), 1.0);
  make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data());
  int *rp = row.data(), *cp = col.data();
  double *vp = val.data(), *xp = x.data(), *yp = y.data();
  size_t rs = row.size(), cs = col.size(), vs = val.size(), xs = x.size(),
         ys = y.size();

  if (!hip_success(hipSetDevice(o.device), "hipSetDevice") || !prove_offload()) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "hipSetDevice failed");
    w.close();
    return 1;
  }

  if (o.scope == GPU_SUITE_SCOPE_END_TO_END) {
    for (int warmup = 0; warmup < o.warmup && !fatal_failure; ++warmup) {
      std::fill(y.begin(), y.end(), 1.0);
      double ignored_elapsed = 0.0;
      char ignored_start[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char ignored_end[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      fatal_failure =
          !run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                               o.alpha, o.beta, &ignored_elapsed, e, sizeof(e),
                               false, false, ignored_start, ignored_end);
    }
    if (fatal_failure) {
      gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                                 "OpenMP rocSPARSE warmup failed");
      any_failure = true;
    }
    for (int trial = 0; trial < o.trials && !fatal_failure; ++trial) {
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, o, trial)) {
        fatal_failure = true;
        any_failure = true;
        break;
      }
      char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      double elapsed_total = 0.0;
      bool ok = true;
      for (int repeat = 0; repeat < o.repeat && ok; ++repeat) {
        std::fill(y.begin(), y.end(), 1.0);
        double elapsed = 0.0;
        ok = run_end_to_end_once(n, nnz, rp, rs, cp, cs, vp, vs, xp, xs, yp, ys,
                                 o.alpha, o.beta, &elapsed, e, sizeof(e),
                                 repeat == 0, repeat + 1 == o.repeat,
                                 start_timestamp, end_timestamp);
        if (ok)
          elapsed_total += elapsed;
      }
      if (ok) {
        result.measurement_start_timestamp = start_timestamp;
        result.measurement_end_timestamp = end_timestamp;
        result.elapsed_total_sec =
            gpu_suite_optional_double_value(elapsed_total);
        result.elapsed_sec =
            gpu_suite_optional_double_value(elapsed_total / o.repeat);
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_rocsparse::set_spmv_verification(result, o, y, nx, ny, 1);
        const bool constructed =
            outcome != gpu_suite::VerificationOutcome::construction_error;
        const bool pass = outcome == gpu_suite::VerificationOutcome::pass;
        result.attempted = true;
        result.failure_origin =
            pass ? nullptr : (constructed ? "verification" : "benchmark");
        if (!constructed)
          result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
        result.status = pass ? "success" : "failure";
        result.message =
            pass ? ""
                 : (constructed ? "OpenMP SpMV verification failed"
                                : "verification result construction failed");
        any_failure = any_failure || !pass;
        ok = ok && constructed;
      } else {
        result.attempted = true;
        result.failure_origin = "benchmark";
        result.verification_status = "skipped";
        result.exit_code = gpu_suite_optional_int_value(1);
        result.status = "failure";
        result.message = "OpenMP end-to-end SpMV failed";
        fatal_failure = true;
        any_failure = true;
      }
      if (!w.write(result)) {
        fatal_failure = true;
        any_failure = true;
      }
      gpu_suite_result_destroy(&result);
      fatal_failure = fatal_failure || !ok;
      if (fatal_failure) {
        gpu_suite::emit_unmeasured(w, o, trial + 1, false, "prior-failure",
                                   "prior OpenMP rocSPARSE failure");
      }
    }
    if (!w.close())
      any_failure = true;
    return any_failure ? 1 : 0;
  }

  rocsparse_handle h = nullptr;
  fatal_failure = !rocsparse_success(rocsparse_create_handle(&h), "rocsparse_create_handle");
  int next_trial = 0;
#pragma omp target data map(to: rp[0 : rs], cp[0 : cs], vp[0 : vs], \
    xp[0 : xs]) map(tofrom: yp[0 : ys])
  {
    rocsparse_spmat_descr a = nullptr;
    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    void *work = nullptr;
    size_t ws = 0;
#pragma omp target data use_device_ptr(rp, cp, vp, xp, yp)
    {
      fatal_failure = fatal_failure || !(mapped_device(rp) && mapped_device(cp) && mapped_device(vp) && mapped_device(xp) && mapped_device(yp));
      if (!fatal_failure) {
        fatal_failure =
            !rocsparse_success(
                rocsparse_create_csr_descr(&a, n, n, nnz, rp, cp, vp,
                                  rocsparse_indextype_i32, rocsparse_indextype_i32,
                                  rocsparse_index_base_zero, rocsparse_datatype_f64_r),
                "rocsparse_create_csr_descr") ||
            !rocsparse_success(rocsparse_create_dnvec_descr(&vx, n, xp, rocsparse_datatype_f64_r),
                              "rocsparse_create_dnvec_descr x") ||
            !rocsparse_success(rocsparse_create_dnvec_descr(&vy, n, yp, rocsparse_datatype_f64_r),
                              "rocsparse_create_dnvec_descr y");
      }
    }
    if (!fatal_failure)
      fatal_failure =
          !rocsparse_success(
              rocsparse_spmv(h, rocsparse_operation_none, &o.alpha, a, vx, &o.beta, vy,
                  rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
                  rocsparse_spmv_stage_buffer_size, &ws, nullptr),
              "rocsparse_spmv buffer_size") ||
          (ws != 0 && !hip_success(hipMalloc(&work, ws),
                                    "hipMalloc rocSPARSE workspace"));
    if (!fatal_failure)
      fatal_failure = !rocsparse_success(rocsparse_spmv(h, rocsparse_operation_none,
          &o.alpha, a, vx, &o.beta, vy, rocsparse_datatype_f64_r,
          rocsparse_spmv_alg_csr_adaptive, rocsparse_spmv_stage_preprocess,
          &ws, work), "rocsparse_spmv preprocess");
    for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm)
      fatal_failure = !rocsparse_success(
          rocsparse_spmv(h, rocsparse_operation_none, &o.alpha, a, vx, &o.beta, vy,
              rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
              rocsparse_spmv_stage_compute, &ws, work),
          "warmup rocsparse_spmv compute");
    if (!fatal_failure)
      fatal_failure = !hip_success(hipDeviceSynchronize(),
                                    "warmup synchronize");
    for (int t = 0; t < o.trials && !fatal_failure; ++t) {
      std::fill(y.begin(), y.end(), 1);
#pragma omp target update to(yp[0 : ys])
      gpu_suite_result r;
      if (!gpu_suite::initialize_result(r, o, t)) {
        fatal_failure = true;
        any_failure = true;
        break;
      }
      char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
           et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
      struct timespec ts, te;
      bool ok = hip_success(hipDeviceSynchronize(),
                             "OpenMP update device y") &&
                gpu_suite_measurement_start(st, &ts, e, sizeof(e)) ==
                    GPU_SUITE_OK;
      for (int rep = 0; rep < o.repeat && ok; ++rep)
        ok = rocsparse_success(
            rocsparse_spmv(h, rocsparse_operation_none, &o.alpha, a, vx, &o.beta, vy,
                rocsparse_datatype_f64_r, rocsparse_spmv_alg_csr_adaptive,
                rocsparse_spmv_stage_compute, &ws, work),
            "rocsparse_spmv compute");
      ok = ok && hip_success(hipDeviceSynchronize(),
                              "post-timing synchronize") &&
           gpu_suite_measurement_end(&te, et, e, sizeof(e)) == GPU_SUITE_OK;
#pragma omp target update from(yp[0 : ys])
      ok = ok && hip_success(hipDeviceSynchronize(),
                              "OpenMP update self y");
      bool pass = false;
      if (ok) {
        const double elapsed = gpu_suite_clock_elapsed(&ts, &te);
        r.measurement_start_timestamp = st;
        r.measurement_end_timestamp = et;
        r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
        const gpu_suite::VerificationOutcome outcome =
            gpu_suite_rocsparse::set_spmv_verification(r, o, y, nx, ny,
                                                      o.repeat);
        const bool constructed =
            outcome != gpu_suite::VerificationOutcome::construction_error;
        pass = outcome == gpu_suite::VerificationOutcome::pass;
        if (!constructed) {
          r.verification_status = "skipped";
          ok = false;
        }
      } else {
        r.verification_status = "skipped";
      }
      r.attempted = true;
      r.failure_origin = pass ? nullptr : (ok ? "verification" : "benchmark");
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? "" : "OpenMP SpMV failed";
      if (!w.write(r)) {
        fatal_failure = true;
        any_failure = true;
      }
      gpu_suite_result_destroy(&r);
      next_trial = t + 1;
      any_failure = any_failure || !pass;
      if (!ok)
        fatal_failure = true;
      if (fatal_failure) {
        gpu_suite::emit_unmeasured(w, o, next_trial, false, "prior-failure",
                                   "prior OpenMP rocSPARSE failure");
        break;
      }
    }
    if (work)
      any_failure = !hip_success(hipFree(work), "hipFree cleanup") || any_failure;
    if (vy)
      any_failure = !rocsparse_success(rocsparse_destroy_dnvec_descr(vy), "rocsparse_destroy_dnvec_descr cleanup") || any_failure;
    if (vx)
      any_failure = !rocsparse_success(rocsparse_destroy_dnvec_descr(vx), "rocsparse_destroy_dnvec_descr cleanup") || any_failure;
    if (a)
      any_failure = !rocsparse_success(rocsparse_destroy_spmat_descr(a), "rocsparse_destroy_spmat_descr cleanup") || any_failure;
  }
  if (fatal_failure && next_trial == 0) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "OpenMP rocSPARSE setup/warmup failed");
  }
  if (h)
    any_failure = !rocsparse_success(rocsparse_destroy_handle(h), "rocsparse_destroy_handle cleanup") || any_failure;
  } catch (const std::length_error &) {
    std::fprintf(stderr,
                 "OpenMP rocSPARSE host vector length is unsupported\n");
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host vector length exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "OpenMP rocSPARSE host allocation failed\n");
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "OpenMP rocSPARSE host allocation failed: %s\n",
                 exception.what());
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation or vector length failed");
    any_failure = true;
  }
  if (!w.close())
    any_failure = true;
  if (fatal_failure)
    any_failure = true;
  return any_failure ? 1 : 0;
}
