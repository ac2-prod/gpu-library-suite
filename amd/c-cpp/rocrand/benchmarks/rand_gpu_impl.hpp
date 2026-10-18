#ifndef GPU_SUITE_ROCRAND_GPU_IMPL_HPP
#define GPU_SUITE_ROCRAND_GPU_IMPL_HPP
#include "rand_bench_common.hpp"
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
#ifdef GPU_SUITE_RAND_OPENMP
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif
#endif

namespace gpu_suite_rocrand {
inline bool hip_ok(hipError_t code) {
  if (code == hipSuccess) return true;
  std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(code)); return false;
}
inline bool rand_ok(rocrand_status code) {
  if (code == ROCRAND_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "rocRAND status %d\n", static_cast<int>(code)); return false;
}
struct Context {
  rocrand_generator generator = nullptr;
  double *host = nullptr;
  double *device_values = nullptr;
  std::size_t count = 0;
  int omp_device = 0;
  bool mapped = false;
  bool initialize(const gpu_suite_options &options) {
    // set_seed and set_offset invalidate rocRAND's state. Initialize explicitly
    // so compute timing does not include deferred generator initialization.
    return rand_ok(rocrand_set_seed(generator, options.seed)) &&
           rand_ok(rocrand_set_offset(generator, options.offset)) &&
           rand_ok(rocrand_set_ordering(generator, ROCRAND_ORDERING_PSEUDO_DEFAULT)) &&
           rand_ok(rocrand_initialize_generator(generator)) &&
           hip_ok(hipDeviceSynchronize());
  }
  bool create(std::vector<double> &values, const gpu_suite_options &options) {
    host = values.data(); count = values.size();
    if (!rand_ok(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT)))
      return false;
#ifdef GPU_SUITE_RAND_OPENMP
    omp_device = omp_get_default_device();
    double *data = host;
#pragma omp target enter data device(omp_device) map(alloc: data[0:count])
    mapped = true;
    bool valid = false;
#pragma omp target data device(omp_device) use_device_ptr(data)
    {
      hipPointerAttribute_t attributes{};
      valid = hip_ok(hipPointerGetAttributes(&attributes, data)) &&
              attributes.type == hipMemoryTypeDevice && attributes.device == options.device;
    }
    if (!valid) {
      std::fprintf(stderr, "OpenMP allocation is not on requested HIP device\n");
      return false;
    }
#else
    if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&device_values), count * sizeof(double))))
      return false;
#endif
    return initialize(options);
  }
  bool generate(int repeat) {
    bool ok = true;
#ifdef GPU_SUITE_RAND_OPENMP
    double *data = host;
#pragma omp target data device(omp_device) use_device_ptr(data)
    {
      for (int i = 0; i < repeat && ok; ++i)
        ok = rand_ok(rocrand_generate_uniform_double(generator, data, count));
    }
#else
    for (int i = 0; i < repeat && ok; ++i)
      ok = rand_ok(rocrand_generate_uniform_double(generator, device_values, count));
#endif
    // One completion boundary for the sequence, not an extra sync per repeat.
    const bool completed = hip_ok(hipDeviceSynchronize());
    return ok && completed;
  }
  bool retrieve() {
#ifdef GPU_SUITE_RAND_OPENMP
    double *data = host;
#pragma omp target update device(omp_device) from(data[0:count])
    return true;
#else
    return hip_ok(hipMemcpy(host, device_values, count * sizeof(double), hipMemcpyDeviceToHost));
#endif
  }
  bool destroy() {
    bool ok = true;
    if (generator != nullptr) {
      ok = rand_ok(rocrand_destroy_generator(generator)); generator = nullptr;
    }
#ifdef GPU_SUITE_RAND_OPENMP
    if (mapped) {
      double *data = host;
#pragma omp target exit data device(omp_device) map(delete: data[0:count])
      mapped = false;
    }
#else
    if (device_values != nullptr) {
      if (!hip_ok(hipFree(device_values))) ok = false;
      device_values = nullptr;
    }
#endif
    return ok;
  }
  ~Context() { (void)destroy(); }
};

inline bool prove_device(const gpu_suite_options &options) {
  if (!hip_ok(hipSetDevice(options.device))) return false;
#ifdef GPU_SUITE_RAND_OPENMP
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
#ifdef GPU_SUITE_RAND_OPENMP
  constexpr auto implementation = GPU_SUITE_IMPLEMENTATION_OPENMP;
#else
  constexpr auto implementation = GPU_SUITE_IMPLEMENTATION_HIP;
#endif
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCRAND, implementation);
  const auto parsed = gpu_suite_options_parse(&options, argc, argv, error, sizeof(error));
  if (parsed == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], GPU_SUITE_BENCHMARK_ROCRAND); return EXIT_SUCCESS;
  }
  if (parsed != GPU_SUITE_PARSE_OK) {
    std::fprintf(stderr, "%s\n", error); return EXIT_FAILURE;
  }
  gpu_suite::ResultWriter writer;
  if (!writer.open(options)) return EXIT_FAILURE;
  std::size_t count = 0, bytes = 0;
  if (!gpu_suite_checked_u64_to_size(options.size, &count) ||
      !gpu_suite_checked_bytes(count, sizeof(double), &bytes) ||
      count > std::vector<double>().max_size() || !prove_device(options)) {
    (void)gpu_suite::emit_unmeasured(writer, options, 0, false, "prerequisite",
                                    "rocRAND dimensions or HIP/OpenMP device unavailable");
    return EXIT_FAILURE;
  }
  bool failed = false;
  int next_trial = 0;
  try {
    std::vector<double> values(count);
    Context persistent;
    bool ready = true;
    if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
      ready = persistent.create(values, options);
      if (ready && options.warmup > 0) ready = persistent.generate(options.warmup);
      if (ready) ready = persistent.initialize(options); // Discard warm-up state.
    } else {
      for (int i = 0; i < options.warmup && ready; ++i) {
        Context context;
        ready = context.create(values, options) && context.generate(1) && context.retrieve();
        if (!context.destroy()) ready = false;
      }
    }
    if (!ready) {
      (void)gpu_suite::emit_unmeasured(writer, options, 0, true, "benchmark",
                                      "rocRAND preparation or warm-up failed");
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
      if (options.scope == GPU_SUITE_SCOPE_COMPUTE) {
        ok = persistent.initialize(options) &&
             gpu_suite_measurement_start(start_stamp, &start, error, sizeof(error)) == GPU_SUITE_OK &&
             persistent.generate(options.repeat) &&
             gpu_suite_measurement_end(&end, end_stamp, error, sizeof(error)) == GPU_SUITE_OK;
        if (ok) { elapsed = gpu_suite_clock_elapsed(&start, &end); ok = persistent.retrieve(); }
      } else {
        for (int repeat = 0; repeat < options.repeat && ok; ++repeat) {
          Context context;
          ok = (repeat == 0 ? gpu_suite_measurement_start(start_stamp, &start, error, sizeof(error))
                            : gpu_suite_clock_now(&start, error, sizeof(error))) == GPU_SUITE_OK &&
               context.create(values, options) && context.generate(1) && context.retrieve();
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
        verified = set_random_verification(result, options, values, "ROCRAND_RNG_PSEUDO_DEFAULT", "(0,1]");
      }
      const bool pass = ok && verified == gpu_suite::VerificationOutcome::pass;
      const bool constructed = verified != gpu_suite::VerificationOutcome::construction_error;
      result.attempted = true;
      result.status = pass ? "success" : "failure";
      result.exit_code = gpu_suite_optional_int_value(pass ? 0 : 1);
      result.failure_origin = pass ? nullptr : (ok && constructed ? "verification" : "benchmark");
      if (!ok || !constructed) result.verification_status = "skipped";
      result.message = pass ? "" : "rocRAND generation, timing or verification failed";
      const bool written = writer.write(result);
      gpu_suite_result_destroy(&result);
      if (!written) return EXIT_FAILURE;
      next_trial = trial + 1;
      failed = failed || !pass;
      if (!ok || !constructed) {
        (void)gpu_suite::emit_unmeasured(writer, options, trial + 1, false, "prior-failure",
                                        "prior rocRAND execution failure"); break;
      }
    }
    if (!persistent.destroy()) failed = true;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "rocRAND: %s\n", exception.what());
    (void)gpu_suite::emit_unmeasured(writer, options, next_trial, true, "benchmark",
                                    "host allocation or rocRAND exception"); failed = true;
  }
  if (!writer.close()) failed = true;
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
} // namespace gpu_suite_rocrand
#endif
