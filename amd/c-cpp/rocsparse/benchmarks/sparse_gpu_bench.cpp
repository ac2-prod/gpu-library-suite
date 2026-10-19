#include <cstdint>
#include "gpu_suite/benchmark.hpp"
#include "sparse_bench_common.hpp"

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

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
static uint64_t root(uint64_t v) {
  uint64_t r = (uint64_t)std::sqrt((long double)v);
  while ((r + 1) <= v / (r + 1))
    ++r;
  while (r > v / r)
    --r;
  return r;
}
struct Context {
  rocsparse_handle h = nullptr;
  rocsparse_spmat_descr a = nullptr;
  rocsparse_dnvec_descr x = nullptr, y = nullptr;
  rocsparse_spmv_descr spmv = nullptr;
  int *dr = nullptr, *dc = nullptr;
  double *dv = nullptr, *dx = nullptr, *dy = nullptr;
  void *work = nullptr;
  size_t ws = 0;
};
static bool hip_ok(hipError_t status, const char *api) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", api, hipGetErrorString(status));
  return false;
}
static bool sparse_ok(rocsparse_status status, const char *api) {
  if (status == rocsparse_status_success)
    return true;
  std::fprintf(stderr, "%s: rocSPARSE status %d\n", api, (int)status);
  return false;
}
static bool prepare_spmv(rocsparse_handle handle, rocsparse_spmat_descr matrix,
                         rocsparse_dnvec_descr x, rocsparse_dnvec_descr y,
                         double alpha, double beta, rocsparse_spmv_descr &spmv,
                         void *&workspace, size_t &workspace_size) {
  const auto alg = rocsparse_spmv_alg_csr_rowsplit;
  const auto op = rocsparse_operation_none;
  const auto datatype = rocsparse_datatype_f64_r;
  if (!sparse_ok(rocsparse_create_spmv_descr(&spmv), "rocsparse_create_spmv_descr") ||
      !sparse_ok(rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_alg,
          &alg, sizeof(alg), nullptr), "rocsparse_spmv_set_input alg") ||
      !sparse_ok(rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_operation,
          &op, sizeof(op), nullptr), "rocsparse_spmv_set_input operation") ||
      !sparse_ok(rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_scalar_datatype,
          &datatype, sizeof(datatype), nullptr), "rocsparse_spmv_set_input scalar") ||
      !sparse_ok(rocsparse_spmv_set_input(handle, spmv, rocsparse_spmv_input_compute_datatype,
          &datatype, sizeof(datatype), nullptr), "rocsparse_spmv_set_input compute") ||
      !sparse_ok(rocsparse_v2_spmv_buffer_size(handle, spmv, matrix, x, y,
          rocsparse_v2_spmv_stage_analysis, &workspace_size, nullptr),
          "rocsparse_v2_spmv_buffer_size analysis"))
    return false;
  if (workspace_size && !hip_ok(hipMalloc(&workspace, workspace_size), "analysis workspace"))
    return false;
  if (!sparse_ok(rocsparse_v2_spmv(handle, spmv, &alpha, matrix, x, &beta, y,
          rocsparse_v2_spmv_stage_analysis, workspace_size, workspace, nullptr),
          "rocsparse_v2_spmv analysis"))
    return false;
  if (workspace && !hip_ok(hipFree(workspace), "free analysis workspace"))
    return false;
  workspace = nullptr;
  if (!sparse_ok(rocsparse_v2_spmv_buffer_size(handle, spmv, matrix, x, y,
          rocsparse_v2_spmv_stage_compute, &workspace_size, nullptr),
          "rocsparse_v2_spmv_buffer_size compute"))
    return false;
  return workspace_size == 0 || hip_ok(hipMalloc(&workspace, workspace_size), "compute workspace");
}

static bool destroy(Context &q) {
  bool ok = true;
  if (q.work)
    ok = hip_ok(hipFree(q.work), "hipFree cleanup") && ok;
  if (q.spmv)
    ok = sparse_ok(rocsparse_destroy_spmv_descr(q.spmv), "rocsparse_destroy_spmv_descr cleanup") && ok;
  if (q.y)
    ok = sparse_ok(rocsparse_destroy_dnvec_descr(q.y), "rocsparse_destroy_dnvec_descr cleanup") && ok;
  if (q.x)
    ok = sparse_ok(rocsparse_destroy_dnvec_descr(q.x), "rocsparse_destroy_dnvec_descr cleanup") && ok;
  if (q.a)
    ok = sparse_ok(rocsparse_destroy_spmat_descr(q.a), "rocsparse_destroy_spmat_descr cleanup") && ok;
  if (q.h)
    ok = sparse_ok(rocsparse_destroy_handle(q.h), "rocsparse_destroy_handle cleanup") && ok;
  if (q.dy)
    ok = hip_ok(hipFree(q.dy), "hipFree cleanup") && ok;
  if (q.dx)
    ok = hip_ok(hipFree(q.dx), "hipFree cleanup") && ok;
  if (q.dv)
    ok = hip_ok(hipFree(q.dv), "hipFree cleanup") && ok;
  if (q.dc)
    ok = hip_ok(hipFree(q.dc), "hipFree cleanup") && ok;
  if (q.dr)
    ok = hip_ok(hipFree(q.dr), "hipFree cleanup") && ok;
  q = Context{};
  return ok;
}
static bool create(Context &q, int n, int nnz, const std::vector<int> &r,
                   const std::vector<int> &c, const std::vector<double> &v,
                   const std::vector<double> &x, const std::vector<double> &y,
                   std::size_t row_bytes, std::size_t column_bytes,
                   std::size_t value_bytes, std::size_t vector_bytes,
                   double alpha, double beta) {
  if (!hip_ok(hipMalloc((void **)&q.dr, row_bytes),
               "hipMalloc CSR row") ||
      !hip_ok(hipMalloc((void **)&q.dc, column_bytes),
               "hipMalloc CSR column") ||
      !hip_ok(hipMalloc((void **)&q.dv, value_bytes),
               "hipMalloc CSR value") ||
      !hip_ok(hipMalloc((void **)&q.dx, vector_bytes),
               "hipMalloc x") ||
      !hip_ok(hipMalloc((void **)&q.dy, vector_bytes),
               "hipMalloc y") ||
      !hip_ok(hipMemcpy(q.dr, r.data(), row_bytes,
                          hipMemcpyHostToDevice),
               "hipMemcpy CSR row") ||
      !hip_ok(hipMemcpy(q.dc, c.data(), column_bytes,
                          hipMemcpyHostToDevice),
               "hipMemcpy CSR column") ||
      !hip_ok(hipMemcpy(q.dv, v.data(), value_bytes,
                          hipMemcpyHostToDevice),
               "hipMemcpy CSR value") ||
      !hip_ok(hipMemcpy(q.dx, x.data(), vector_bytes,
                          hipMemcpyHostToDevice),
               "hipMemcpy x") ||
      !hip_ok(hipMemcpy(q.dy, y.data(), vector_bytes,
                          hipMemcpyHostToDevice),
               "hipMemcpy y") ||
      !sparse_ok(rocsparse_create_handle(&q.h), "rocsparse_create_handle") ||
      !sparse_ok(rocsparse_create_csr_descr(&q.a, n, n, nnz, q.dr, q.dc, q.dv,
                                   rocsparse_indextype_i32, rocsparse_indextype_i32,
                                   rocsparse_index_base_zero, rocsparse_datatype_f64_r),
                 "rocsparse_create_csr_descr") ||
      !sparse_ok(rocsparse_create_dnvec_descr(&q.x, n, q.dx, rocsparse_datatype_f64_r),
                 "rocsparse_create_dnvec_descr x") ||
      !sparse_ok(rocsparse_create_dnvec_descr(&q.y, n, q.dy, rocsparse_datatype_f64_r),
                 "rocsparse_create_dnvec_descr y"))
    return false;
  return prepare_spmv(q.h, q.a, q.x, q.y, alpha, beta, q.spmv, q.work, q.ws);
}
static bool apply(Context &q, double a, double b) {
  return sparse_ok(rocsparse_v2_spmv(q.h, q.spmv, &a, q.a, q.x, &b, q.y,
      rocsparse_v2_spmv_stage_compute, q.ws, q.work, nullptr),
                   "rocsparse_v2_spmv compute");
}

static bool run_end_to_end_pipeline(
    const gpu_suite_options &options,
    int n, int nnz, const std::vector<int> &row,
    const std::vector<int> &column, const std::vector<double> &values,
    const std::vector<double> &x, std::vector<double> &y,
    std::size_t row_bytes, std::size_t column_bytes, std::size_t value_bytes,
    std::size_t vector_bytes, double alpha, double beta, int repeat_index,
    bool measure, char start_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY],
    char end_timestamp[GPU_SUITE_TIMESTAMP_CAPACITY], char *error,
    std::size_t error_size, double *elapsed_total) {
  Context context;
  struct timespec start;
  struct timespec end;
  bool ok = !measure ||
            (repeat_index == 0
                 ? gpu_suite_measurement_start(start_timestamp, &start, error,
                                               error_size)
                 : gpu_suite_clock_now(&start, error, error_size)) ==
                GPU_SUITE_OK;
  ok = ok && create(context, n, nnz, row, column, values, x, y, row_bytes,
                    column_bytes, value_bytes, vector_bytes, alpha, beta) &&
       apply(context, alpha, beta) &&
       hip_ok(hipDeviceSynchronize(), "end-to-end synchronize") &&
       hip_ok(hipMemcpy(y.data(), context.dy, vector_bytes,
                          hipMemcpyDeviceToHost),
               "copy end-to-end y");
  if (ok && measure) {
    ok = (repeat_index + 1 == options.repeat
              ? gpu_suite_measurement_end(&end, end_timestamp, error,
                                          error_size)
              : gpu_suite_clock_now(&end, error, error_size)) == GPU_SUITE_OK;
    if (ok)
      *elapsed_total += gpu_suite_clock_elapsed(&start, &end);
  }
  return destroy(context) && ok;
}
int main(int argc, char **argv) {
  gpu_suite_options o;
  char e[256] = {0};
  gpu_suite_options_init(&o, GPU_SUITE_BENCHMARK_ROCSPARSE,
                         GPU_SUITE_IMPLEMENTATION_HIP);
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
  bool any_failure = false;
  try {
  std::vector<int> row(row_count), col(static_cast<std::size_t>(nnz));
  std::vector<double> val(static_cast<std::size_t>(nnz));
  std::vector<double> x(static_cast<std::size_t>(n), 1.0);
  std::vector<double> y(static_cast<std::size_t>(n), 1.0);
  make_poisson2d_csr(nx, ny, row.data(), col.data(), val.data());
  Context q;
  bool fatal_failure = !hip_ok(hipSetDevice(o.device), "hipSetDevice");
  if (!fatal_failure && o.scope == GPU_SUITE_SCOPE_COMPUTE) {
    fatal_failure = !create(q, n, nnz, row, col, val, x, y, row_bytes,
                            column_bytes, value_bytes, vector_bytes, o.alpha,
                            o.beta);
    for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm)
      fatal_failure = !apply(q, o.alpha, o.beta);
    if (!fatal_failure)
      fatal_failure = !hip_ok(hipDeviceSynchronize(), "warmup synchronize");
  } else if (!fatal_failure) {
    for (int warm = 0; warm < o.warmup && !fatal_failure; ++warm) {
      std::fill(y.begin(), y.end(), 1.0);
      fatal_failure = !run_end_to_end_pipeline(
          o, n, nnz, row, col, val, x, y, row_bytes, column_bytes,
          value_bytes, vector_bytes, o.alpha, o.beta, 0, false, nullptr,
          nullptr, e, sizeof(e), nullptr);
    }
  }
  if (fatal_failure) {
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "rocSPARSE setup/warmup failed");
    any_failure = true;
  }
  for (int t = 0; t < o.trials && !fatal_failure; ++t) {
    gpu_suite_result r;
    if (!gpu_suite::initialize_result(r, o, t)) {
      any_failure = true;
      fatal_failure = true;
      break;
    }
    std::fill(y.begin(), y.end(), 1);
    char st[GPU_SUITE_TIMESTAMP_CAPACITY] = {0},
         et[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
    struct timespec ts, te;
    double elapsed = 0;
    bool ok = true;
    if (o.scope == GPU_SUITE_SCOPE_COMPUTE) {
      ok = hip_ok(hipMemcpy(q.dy, y.data(), vector_bytes,
                              hipMemcpyHostToDevice),
                   "restore y") &&
           hip_ok(hipDeviceSynchronize(), "pre-timing synchronize") &&
           gpu_suite_measurement_start(st, &ts, e, sizeof(e)) == GPU_SUITE_OK;
      for (int rep = 0; rep < o.repeat && ok; ++rep)
        ok = apply(q, o.alpha, o.beta);
      ok = ok && hip_ok(hipDeviceSynchronize(), "post-timing synchronize") &&
           gpu_suite_measurement_end(&te, et, e, sizeof(e)) == GPU_SUITE_OK;
      if (ok)
        elapsed = gpu_suite_clock_elapsed(&ts, &te);
      if (ok)
        ok = hip_ok(hipMemcpy(y.data(), q.dy, vector_bytes,
                               hipMemcpyDeviceToHost),
                     "copy result y");
    } else {
      for (int rep = 0; rep < o.repeat && ok; ++rep) {
        std::fill(y.begin(), y.end(), 1);
        ok = run_end_to_end_pipeline(
            o, n, nnz, row, col, val, x, y, row_bytes, column_bytes,
            value_bytes, vector_bytes, o.alpha, o.beta, rep, true, st, et, e,
            sizeof(e), &elapsed);
      }
    }
    int updates = o.scope == GPU_SUITE_SCOPE_COMPUTE ? o.repeat : 1;
    if (ok) {
      r.measurement_start_timestamp = st;
      r.measurement_end_timestamp = et;
      r.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
      r.elapsed_sec = gpu_suite_optional_double_value(elapsed / o.repeat);
      const gpu_suite::VerificationOutcome outcome =
          gpu_suite_rocsparse::set_spmv_verification(r, o, y, nx, ny, updates);
      const bool constructed =
          outcome != gpu_suite::VerificationOutcome::construction_error;
      bool pass = outcome == gpu_suite::VerificationOutcome::pass;
      r.attempted = true;
      r.failure_origin =
          pass ? nullptr : (constructed ? "verification" : "benchmark");
      if (!constructed)
        r.verification_status = "skipped";
      r.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      r.status = pass ? "success" : "failure";
      r.message = pass ? ""
                       : (constructed ? "SpMV verification failed"
                                      : "verification result construction failed");
      any_failure = any_failure || !pass;
      ok = ok && constructed;
    } else {
      r.attempted = true;
      r.failure_origin = "benchmark";
      r.verification_status = "skipped";
      r.exit_code = gpu_suite_optional_int_value(1);
      r.status = "failure";
      r.message = "rocSPARSE pipeline failed";
      any_failure = true;
      fatal_failure = true;
    }
    if (!w.write(r)) {
      any_failure = true;
      fatal_failure = true;
    }
    gpu_suite_result_destroy(&r);
    fatal_failure = fatal_failure || !ok;
    if (fatal_failure) {
      gpu_suite::emit_unmeasured(w, o, t + 1, false, "prior-failure",
                                 "prior rocSPARSE failure");
      break;
    }
  }
  if (!destroy(q)) any_failure = true;
  } catch (const std::length_error &) {
    std::fprintf(stderr, "rocSPARSE host vector length is unsupported\n");
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host vector length exceeds max_size");
    any_failure = true;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "rocSPARSE host allocation failed\n");
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation failed");
    any_failure = true;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "rocSPARSE host allocation failed: %s\n",
                 exception.what());
    gpu_suite::emit_unmeasured(w, o, 0, true, "benchmark",
                               "host allocation or vector length failed");
    any_failure = true;
  }
  if (!w.close())
    any_failure = true;
  return any_failure ? 1 : 0;
}
