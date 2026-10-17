#ifndef GPU_SUITE_ROCSOLVER_BENCH_HPP
#define GPU_SUITE_ROCSOLVER_BENCH_HPP
#include "solver_bench_common.hpp"
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target support is required"
#endif
#endif

namespace gpu_suite_rocsolver {
inline bool hip_ok(hipError_t status, const char *operation) {
  if (status == hipSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
  return false;
}
inline bool solver_ok(rocblas_status status, const char *operation) {
  if (status == rocblas_status_success) return true;
  std::fprintf(stderr, "%s: rocBLAS status %d\n", operation, (int)status);
  return false;
}
inline bool query_ok(rocblas_status status) {
  return status == rocblas_status_success || status == rocblas_status_size_increased ||
         status == rocblas_status_size_unchanged;
}
inline void make_system(int n, std::vector<double> &a, std::vector<double> &b) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? n + 1.0 : 1.0;
  std::fill(b.begin(), b.end(), 2.0 * n);
}
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
inline bool mapped_device(const void *p, int device) {
  hipPointerAttribute_t attr{};
  return hip_ok(hipPointerGetAttributes(&attr, p), "OpenMP mapped pointer") &&
         attr.type == hipMemoryTypeDevice && attr.device == device;
}
inline bool prove_offload(int device) {
  if (device < 0 || device >= omp_get_num_devices()) return false;
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) std::fprintf(stderr, "OpenMP host fallback rejected\n");
  if (!initial) std::fprintf(stderr, "OpenMP offload ordinal=%d\n", device);
  return !initial;
}
#endif

// The handle and explicit workspace persist in compute scope. In OpenMP mode
// the data arrays belong exclusively to target mappings, never to hipFree.
struct Context {
  rocblas_handle handle = nullptr;
  void *workspace = nullptr;
  double *da = nullptr, *db = nullptr;
  rocblas_int *dp = nullptr, *di = nullptr;
  double *ha = nullptr, *hb = nullptr;
  rocblas_int host_info = 0, *hp = nullptr;
  size_t ac = 0, bc = 0, ab = 0, bb = 0, pb = 0;
  int n = 0, nrhs = 0, device = 0;
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
  int omp_device = 0;
#endif
  bool mapped = false;
  bool factor_info_known = false, solve_completed = false;
  Context() = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  ~Context() { (void)destroy(); }

  bool create(int dimension, int rhs_count, int requested_device,
              std::vector<double> &a, std::vector<double> &b,
              std::vector<rocblas_int> &p) {
    n = dimension; nrhs = rhs_count; device = requested_device;
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    omp_device = omp_get_default_device();
#endif
    ac = a.size(); bc = b.size(); ab = ac * sizeof(double);
    bb = bc * sizeof(double); pb = p.size() * sizeof(rocblas_int);
    ha = a.data(); hb = b.data(); hp = p.data();
    if (!solver_ok(rocblas_create_handle(&handle), "rocblas_create_handle")) return false;
    if (!solver_ok(rocblas_start_device_memory_size_query(handle), "start workspace query")) return false;
    const auto rf = rocsolver_dgetrf(handle, n, n, nullptr, n, nullptr, nullptr);
    const auto rs = rocsolver_dgetrs(handle, rocblas_operation_none, n, nrhs,
                                    nullptr, n, nullptr, nullptr, n);
    size_t required = 0;
    const auto stop = rocblas_stop_device_memory_size_query(handle, &required);
    if (!query_ok(rf) || !query_ok(rs) || !solver_ok(stop, "stop workspace query")) {
      std::fprintf(stderr, "rocSOLVER workspace query failed\n"); return false;
    }
    // A non-null user workspace prevents hidden allocation in compute scope,
    // including the zero-workspace case (the API treats size zero as automatic).
    required = std::max(required, size_t(1));
    if (!hip_ok(hipMalloc(&workspace, required), "hipMalloc workspace") ||
        !solver_ok(rocblas_set_workspace(handle, workspace, required), "rocblas_set_workspace")) return false;
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    double *a_ptr = ha, *b_ptr = hb;
    rocblas_int *p_ptr = hp, *i_ptr = &host_info;
#pragma omp target enter data device(omp_device) map(alloc: a_ptr[0:ac], b_ptr[0:bc], p_ptr[0:n], i_ptr[0:1])
    mapped = true;
    bool valid = false;
#pragma omp target data device(omp_device) use_device_ptr(a_ptr, b_ptr, p_ptr, i_ptr)
    {
      valid = mapped_device(a_ptr, device) && mapped_device(b_ptr, device) &&
              mapped_device(p_ptr, device) && mapped_device(i_ptr, device);
    }
    if (!valid) return false;
#else
    if (!hip_ok(hipMalloc((void **)&da, ab), "hipMalloc A") ||
        !hip_ok(hipMalloc((void **)&db, bb), "hipMalloc B") ||
        !hip_ok(hipMalloc((void **)&dp, pb), "hipMalloc pivots") ||
        !hip_ok(hipMalloc((void **)&di, sizeof(*di)), "hipMalloc info")) return false;
#endif
    return true;
  }
  bool restore(const std::vector<double> &a0, const std::vector<double> &b0, bool restore_host = true) {
    factor_info_known = false; solve_completed = false; host_info = 0;
    if (restore_host) {
      std::copy(a0.begin(), a0.end(), ha); std::copy(b0.begin(), b0.end(), hb);
      std::fill(hp, hp + n, 0);
    }
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    double *a_ptr = ha, *b_ptr = hb;
    rocblas_int *p_ptr = hp, *i_ptr = &host_info;
#pragma omp target update device(omp_device) to(a_ptr[0:ac], b_ptr[0:bc], p_ptr[0:n], i_ptr[0:1])
    return hip_ok(hipDeviceSynchronize(), "restoration completion");
#else
    return hip_ok(hipMemcpy(da, ha, ab, hipMemcpyHostToDevice), "restore A") &&
           hip_ok(hipMemcpy(db, hb, bb, hipMemcpyHostToDevice), "restore B") &&
           hip_ok(hipMemset(dp, 0, pb), "restore pivots") &&
           hip_ok(hipMemset(di, 0, sizeof(*di)), "restore info") &&
           hip_ok(hipDeviceSynchronize(), "restoration completion");
#endif
  }
  bool solve_device(double *a, double *b, rocblas_int *p, rocblas_int *info) {
    bool ok = solver_ok(rocsolver_dgetrf(handle, n, n, a, n, p, info), "rocsolver_dgetrf") &&
              hip_ok(hipDeviceSynchronize(), "getrf completion") &&
              hip_ok(hipMemcpy(&host_info, info, sizeof(host_info), hipMemcpyDeviceToHost), "getrf info");
    factor_info_known = ok;
    if (!ok || host_info != 0) {
      std::fprintf(stderr, "rocSOLVER factorization failed, info known=%d info=%d\n", (int)ok, (int)host_info);
      return false;
    }
    // getrs has a rocblas_status return, not a LAPACK info argument. Only a
    // successful API call AND completion justify the synthetic getrs_info=0.
    solve_completed = solver_ok(rocsolver_dgetrs(handle, rocblas_operation_none,
        n, nrhs, a, n, p, b, n), "rocsolver_dgetrs") &&
        hip_ok(hipDeviceSynchronize(), "getrs completion");
    return solve_completed;
  }
  bool solve() {
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    double *a_ptr = ha, *b_ptr = hb;
    rocblas_int *p_ptr = hp, *i_ptr = &host_info;
    bool ok = false;
#pragma omp target data device(omp_device) use_device_ptr(a_ptr, b_ptr, p_ptr, i_ptr)
    { ok = solve_device(a_ptr, b_ptr, p_ptr, i_ptr); }
    return ok;
#else
    return solve_device(da, db, dp, di);
#endif
  }
  bool retrieve() {
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    double *b_ptr = hb;
#pragma omp target update device(omp_device) from(b_ptr[0:bc])
    return true;
#else
    return hip_ok(hipMemcpy(hb, db, bb, hipMemcpyDeviceToHost), "retrieve solution");
#endif
  }
  bool destroy() {
    bool ok = true;
    // Complete submitted work before destroying its workspace or mappings,
    // including an error path after an asynchronous library submission.
    if (handle) ok = hip_ok(hipDeviceSynchronize(), "cleanup completion") && ok;
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
    if (mapped) {
      double *a_ptr = ha, *b_ptr = hb;
      rocblas_int *p_ptr = hp, *i_ptr = &host_info;
#pragma omp target exit data device(omp_device) map(delete: a_ptr[0:ac], b_ptr[0:bc], p_ptr[0:n], i_ptr[0:1])
      mapped = false;
    }
#else
    if (di) ok = hip_ok(hipFree(di), "hipFree info") && ok;
    if (dp) ok = hip_ok(hipFree(dp), "hipFree pivots") && ok;
    if (db) ok = hip_ok(hipFree(db), "hipFree B") && ok;
    if (da) ok = hip_ok(hipFree(da), "hipFree A") && ok;
    di = nullptr; dp = nullptr; db = nullptr; da = nullptr;
#endif
    if (handle) {
      ok = solver_ok(rocblas_set_workspace(handle, nullptr, 0), "detach workspace") && ok;
      ok = solver_ok(rocblas_destroy_handle(handle), "rocblas_destroy_handle") && ok;
      handle = nullptr;
    }
    if (workspace) ok = hip_ok(hipFree(workspace), "hipFree workspace") && ok;
    workspace = nullptr;
    return ok;
  }
};

inline int run(int argc, char **argv) {
  gpu_suite_options options;
  char error[256]{};
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
  const auto implementation = GPU_SUITE_IMPLEMENTATION_OPENMP;
#else
  const auto implementation = GPU_SUITE_IMPLEMENTATION_HIP;
#endif
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCSOLVER, implementation);
  const auto parsed = gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCSOLVER); return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK || options.repeat != 1) {
    std::fprintf(stderr, "invalid rocSOLVER options: %s (repeat must be 1)\n", error); return EXIT_FAILURE;
  }
  gpu_suite::ResultWriter writer;
  if (!writer.open(options)) return EXIT_FAILURE;
  int n = 0, nrhs = 0;
  size_t ac = 0, bc = 0, ab = 0, bb = 0, pb = 0;
  if (!gpu_suite_checked_u64_to_int(options.size, &n) ||
      !gpu_suite_checked_u64_to_int(options.nrhs, &nrhs) ||
      !gpu_suite_checked_mul_size((size_t)n, (size_t)n, &ac) ||
      !gpu_suite_checked_mul_size((size_t)n, (size_t)nrhs, &bc) ||
      !gpu_suite_checked_bytes(ac, sizeof(double), &ab) ||
      !gpu_suite_checked_bytes(bc, sizeof(double), &bb) ||
      !gpu_suite_checked_bytes((size_t)n, sizeof(rocblas_int), &pb)) {
    gpu_suite::emit_unmeasured(writer, options, 0, false, "prerequisite", "rocSOLVER dimensions overflow");
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(options.device), "hipSetDevice")
#ifdef GPU_SUITE_ROCSOLVER_OPENMP
      || !prove_offload(omp_get_default_device())
#endif
      ) {
    gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark", "requested GPU/offload device unavailable");
    return EXIT_FAILURE;
  }
  int next_trial = 0;
  bool failed = false;
  try {
    std::vector<double> a0(ac), b0(bc), a(ac), b(bc);
    std::vector<rocblas_int> piv(n);
    make_system(n, a0, b0);
    Context persistent;
    const bool compute = options.scope == GPU_SUITE_SCOPE_COMPUTE;
    if (compute && !persistent.create(n, nrhs, options.device, a, b, piv)) {
      gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark", "rocSOLVER setup failed");
      return EXIT_FAILURE;
    }
    for (int w = 0; w < options.warmup; ++w) {
      Context transient;
      Context &ctx = compute ? persistent : transient;
      bool ok = (compute || ctx.create(n, nrhs, options.device, a, b, piv)) &&
                ctx.restore(a0, b0) && ctx.solve() && ctx.retrieve();
      if (!compute) ok = ctx.destroy() && ok;
      if (!ok) {
        gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark", "rocSOLVER warm-up failed");
        return EXIT_FAILURE;
      }
    }
    for (int trial = 0; trial < options.trials; ++trial) {
      Context transient;
      Context &ctx = compute ? persistent : transient;
      char start_ts[GPU_SUITE_TIMESTAMP_CAPACITY]{}, end_ts[GPU_SUITE_TIMESTAMP_CAPACITY]{};
      struct timespec start{}, end{};
      // Host canonical restoration is always untimed. Compute device restore
      // also precedes timing; end-to-end includes creation, H2D and retrieval.
      std::copy(a0.begin(), a0.end(), a.begin());
      std::copy(b0.begin(), b0.end(), b.begin());
      std::fill(piv.begin(), piv.end(), 0);
      bool ok = !compute || ctx.restore(a0, b0);
      ok = ok && gpu_suite_measurement_start(start_ts, &start, error, sizeof(error)) == GPU_SUITE_OK;
      ok = ok && (compute || (ctx.create(n, nrhs, options.device, a, b, piv) && ctx.restore(a0, b0, false)));
      ok = ok && ctx.solve();
      if (!compute) ok = ok && ctx.retrieve();
      ok = ok && gpu_suite_measurement_end(&end, end_ts, error, sizeof(error)) == GPU_SUITE_OK;
      if (compute) ok = ok && ctx.retrieve();
      if (!compute) ok = ctx.destroy() && ok;
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial)) { failed = true; break; }
      result.getrf_info = ctx.factor_info_known ? gpu_suite_optional_int_value(ctx.host_info) : gpu_suite_optional_int_null();
      result.getrs_info = ctx.solve_completed ? gpu_suite_optional_int_value(0) : gpu_suite_optional_int_null();
      auto verified = gpu_suite::VerificationOutcome::pass;
      if (ok) {
        result.measurement_start_timestamp = start_ts;
        result.measurement_end_timestamp = end_ts;
        const double elapsed = gpu_suite_clock_elapsed(&start, &end);
        result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        result.elapsed_sec = gpu_suite_optional_double_value(elapsed);
        verified = set_solver_verification(result, options, b, a0, b0, n, nrhs);
      } else result.verification_status = "skipped";
      const bool pass = ok && verified == gpu_suite::VerificationOutcome::pass;
      result.attempted = true;
      result.status = pass ? "success" : "failure";
      result.failure_origin = pass ? nullptr :
          (ok && verified == gpu_suite::VerificationOutcome::failure ? "verification" : "benchmark");
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.message = pass ? "" : "rocSOLVER pipeline, LU info or verification failed";
      if (verified == gpu_suite::VerificationOutcome::construction_error) result.verification_status = "skipped";
      const bool written = writer.write(result);
      gpu_suite_result_destroy(&result);
      next_trial = trial + 1;
      failed |= !pass || !written;
      if (!ok || !written || verified == gpu_suite::VerificationOutcome::construction_error) {
        if (written) gpu_suite::emit_unmeasured(writer, options, next_trial, false, "prior-failure", "prior rocSOLVER failure");
        break;
      }
    }
    failed = !persistent.destroy() || failed;
  } catch (const std::bad_alloc &) {
    gpu_suite::emit_unmeasured(writer, options, next_trial, true, "benchmark", "host allocation failed"); failed = true;
  } catch (const std::length_error &) {
    gpu_suite::emit_unmeasured(writer, options, next_trial, true, "benchmark", "host vector size overflow"); failed = true;
  }
  if (!writer.close()) failed = true;
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
} // namespace gpu_suite_rocsolver
#endif
