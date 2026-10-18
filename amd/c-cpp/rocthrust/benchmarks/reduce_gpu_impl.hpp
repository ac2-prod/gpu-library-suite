#ifndef GPU_SUITE_ROCTHRUST_GPU_IMPL_HPP
#define GPU_SUITE_ROCTHRUST_GPU_IMPL_HPP
#include "reduce_bench_common.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <vector>
#ifdef GPU_SUITE_REDUCE_OPENMP
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif
extern "C" int rocthrust_square_sum(const double *, std::size_t, double *) noexcept;
#else
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/transform_reduce.h>
#endif

namespace gpu_suite_rocthrust {
inline bool hip_ok(hipError_t code) {
  if (code == hipSuccess) return true;
  std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(code)); return false;
}
#ifndef GPU_SUITE_REDUCE_OPENMP
struct SquareValue {
  __host__ __device__ double operator()(double value) const { return value * value; }
};
#endif
struct Context {
  double *host = nullptr;
  double *device_values = nullptr;
  std::size_t count = 0;
  int omp_device = 0;
  bool mapped = false;
  bool create(std::vector<double> &values, const gpu_suite_options &options) {
    host = values.data(); count = values.size();
#ifdef GPU_SUITE_REDUCE_OPENMP
    omp_device = omp_get_default_device();
    double *data = host;
#pragma omp target enter data device(omp_device) map(to: data[0:count])
    mapped = true;
    bool valid = false;
#pragma omp target data device(omp_device) use_device_ptr(data)
    {
      hipPointerAttribute_t attributes{};
      valid = hip_ok(hipPointerGetAttributes(&attributes, data)) &&
              attributes.type == hipMemoryTypeDevice && attributes.device == options.device;
    }
    if (!valid) std::fprintf(stderr, "OpenMP allocation is not on requested HIP device\n");
    return valid;
#else
    (void)options;
    return hip_ok(hipMalloc(reinterpret_cast<void **>(&device_values), count * sizeof(double))) &&
           hip_ok(hipMemcpy(device_values, host, count * sizeof(double), hipMemcpyHostToDevice));
#endif
  }
  bool execute(double &result) {
#ifdef GPU_SUITE_REDUCE_OPENMP
    double *data = host;
    int status = 1;
#pragma omp target data device(omp_device) use_device_ptr(data)
    { status = rocthrust_square_sum(data, count, &result); }
    return status == 0;
#else
    try {
      const auto first = thrust::device_pointer_cast(device_values);
      // Synchronous HIP policy includes reduction and host scalar retrieval.
      result = thrust::transform_reduce(thrust::hip_rocprim::par, first, first + count,
                                        SquareValue{}, 0.0, thrust::plus<double>());
      return true;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "rocThrust: %s\n", error.what()); return false;
    } catch (...) {
      std::fprintf(stderr, "rocThrust: unknown exception\n"); return false;
    }
#endif
  }
  bool destroy() {
#ifdef GPU_SUITE_REDUCE_OPENMP
    if (mapped) {
      double *data = host;
#pragma omp target exit data device(omp_device) map(delete: data[0:count])
      mapped = false;
    }
    return true;
#else
    bool ok = true;
    if (device_values != nullptr) {
      ok = hip_ok(hipFree(device_values)); device_values = nullptr;
    }
    return ok;
#endif
  }
  ~Context() { (void)destroy(); }
};
inline bool prove_device(const gpu_suite_options &options) {
  if (!hip_ok(hipSetDevice(options.device))) return false;
#ifdef GPU_SUITE_REDUCE_OPENMP
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices()) return false;
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) {
    std::fprintf(stderr, "OpenMP host fallback rejected\n"); return false;
  }
#endif
  return true;
}

inline int run(int argc, char **argv) {
  gpu_suite_options options;
  char error[256]{};
#ifdef GPU_SUITE_REDUCE_OPENMP
  constexpr auto implementation = GPU_SUITE_IMPLEMENTATION_OPENMP;
#else
  constexpr auto implementation = GPU_SUITE_IMPLEMENTATION_HIP;
#endif
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCTHRUST, implementation);
  const auto parsed = gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCTHRUST); return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error); return EXIT_FAILURE;
  }
  gpu_suite::ResultWriter writer;
  if (!writer.open(options)) return EXIT_FAILURE;
  std::size_t count = 0, bytes = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_bytes(count, sizeof(double), &bytes) ||
      count > std::vector<double>().max_size() ||
      count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      !prove_device(options)) {
    (void)gpu_suite::emit_unmeasured(writer, options, 0, false, "prerequisite",
                                    "rocThrust dimensions or HIP/OpenMP device unavailable");
    return EXIT_FAILURE;
  }
  bool failed = false;
  int next_trial = 0;
  try {
    std::vector<double> values(count, 1.0); // Immutable; warm-up does not modify input.
    Context persistent;
    double value = 0.0;
    bool ready = true;
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      ready = persistent.create(values, options);
      for (int i = 0; i < options.warmup && ready; ++i) ready = persistent.execute(value);
    } else {
      for (int i = 0; i < options.warmup && ready; ++i) {
        Context context;
        ready = context.create(values, options) && context.execute(value);
        if (!context.destroy()) ready = false;
      }
    }
    value = 0.0;
    if (!ready) {
      (void)gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                      "rocThrust preparation or warm-up failed");
      return EXIT_FAILURE;
    }
    for (int trial = 0; trial < options.trials; ++trial) {
      next_trial = trial;
      gpu_suite_result result;
      if (!gpu_suite::initialize_result(result, options, trial)) return EXIT_FAILURE;
      char start_stamp[GPU_SUITE_TIMESTAMP_CAPACITY]{}, end_stamp[GPU_SUITE_TIMESTAMP_CAPACITY]{};
      struct timespec start{}, end{};
      double elapsed = 0.0;
      bool ok = true;
      value = 0.0;
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok = gpu_suite_measurement_start(start_stamp, &start, error, sizeof(error)) == GPU_SUITE_OK;
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) ok = persistent.execute(value);
        ok = ok && gpu_suite_measurement_end(&end, end_stamp, error, sizeof(error)) == GPU_SUITE_OK;
        if (ok) elapsed = gpu_suite_clock_elapsed(&start, &end);
      } else {
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          Context context;
          ok = (repeat == 0 ? gpu_suite_measurement_start(start_stamp, &start, error, sizeof(error))
                            : gpu_suite_clock_now(&start, error, sizeof(error))) == GPU_SUITE_OK &&
               context.create(values, options) && context.execute(value);
          if (ok) ok = (repeat + 1 == options.repeat
              ? gpu_suite_measurement_end(&end, end_stamp, error, sizeof(error))
              : gpu_suite_clock_now(&end, error, sizeof(error))) == GPU_SUITE_OK;
          if (ok) elapsed += gpu_suite_clock_elapsed(&start, &end);
          if (!context.destroy()) ok = false; // Explicit cleanup is untimed.
        }
      }
      auto verified = gpu_suite::VerificationOutcome::failure;
      if (ok) {
        result.measurement_start_timestamp = start_stamp;
        result.measurement_end_timestamp = end_stamp;
        result.elapsed_total_sec = gpu_suite_optional_double_value(elapsed);
        result.elapsed_sec = gpu_suite_optional_double_value(elapsed / options.repeat);
        verified = set_reduction_verification(result, options, value, count);
      }
      const bool pass = ok && verified == gpu_suite::VerificationOutcome::pass;
      const bool constructed = verified != gpu_suite::VerificationOutcome::construction_error;
      result.attempted = true;
      result.status = pass ? "success" : "failure";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.failure_origin = pass ? nullptr : (ok && constructed ? "verification" : "benchmark");
      if (!ok || !constructed) result.verification_status = "skipped";
      result.message = pass ? "" : "rocThrust reduction, timing or verification failed";
      const bool written = writer.write(result);
      gpu_suite_result_destroy(&result);
      if (!written) return EXIT_FAILURE;
      next_trial = trial + 1;
      failed = failed || !pass;
      if (!ok || !constructed) {
        (void)gpu_suite::emit_unmeasured(writer, options, trial + 1, false, "prior-failure",
                                        "prior rocThrust execution failure"); break;
      }
    }
    if (!persistent.destroy()) failed = true;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "rocThrust: %s\n", exception.what());
    (void)gpu_suite::emit_unmeasured(writer, options, next_trial, true, "benchmark",
                                    "host allocation or rocThrust exception"); failed = true;
  }
  if (!writer.close()) failed = true;
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
} // namespace gpu_suite_rocthrust
#endif
