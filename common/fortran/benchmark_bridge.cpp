// C ABI for common CLI/timing/serialization only. All workloads are Fortran.
#include "gpu_suite/benchmark.h"
#include "gpu_suite/clock.h"
#include "gpu_suite/fortran_metadata.h"
#ifdef GPU_SUITE_HAVE_CUDA_RUNTIME_METADATA
#include "gpu_suite/cuda_metadata.hpp"
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <string>
#include <vector>

#ifdef GPU_SUITE_FORTRAN_FFTW
extern "C" { extern const char fftwf_version[]; }
#endif
#ifdef GPU_SUITE_FORTRAN_MKL
#include <mkl_service.h>
#endif

namespace {
gpu_suite_options options;
gpu_suite_benchmark_writer writer{};
gpu_suite_result row{};
std::vector<std::string> arguments;
char error[1024]{}, start_stamp[GPU_SUITE_TIMESTAMP_CAPACITY]{}, end_stamp[GPU_SUITE_TIMESTAMP_CAPACITY]{};
char library_version[256]{};
timespec start_time{}, end_time{};
double elapsed = 0;
int current_trial = 0;
bool opened = false, row_live = false;
void check(int status) {
  if (status != GPU_SUITE_OK) {
    std::fprintf(stderr, "Fortran benchmark common support: %s (status %d)\n", error, status);
    std::exit(2);
  }
}
void initialize_row(int trial) {
  if (row_live) gpu_suite_result_destroy(&row);
  check(gpu_suite_benchmark_result_init(&row, &options, trial, error, sizeof error));
  row_live = true;
  row.compiler = GPU_SUITE_FORTRAN_COMPILER;
  row.compiler_version = GPU_SUITE_FORTRAN_COMPILER_VERSION;
  row.global_configure_flags = GPU_SUITE_FORTRAN_FLAGS;
  check(gpu_suite_json_add_string(row.parameters, "source_language", "fortran"));
  if (options.benchmark == GPU_SUITE_BENCHMARK_CURAND) {
    check(gpu_suite_json_add_int(row.parameters, "verification_sample_count", static_cast<int64_t>(options.size)));
    const bool cpu = options.implementation == GPU_SUITE_IMPLEMENTATION_CPU;
    check(gpu_suite_json_add_string(row.parameters, cpu ? "cpu_engine" : "generator_algorithm",
          cpu ? "Fortran random_number" : "CURAND_RNG_PSEUDO_DEFAULT"));
    check(gpu_suite_json_add_string(row.parameters, "distribution_interval", cpu ? "[0,1)" : "(0,1]"));
  }
  if (library_version[0]) row.library_version = library_version;
}
void metric(const char *key, double value) {
  check(std::isfinite(value) ? gpu_suite_json_add_double(row.verification_metrics, key, value)
                            : gpu_suite_json_add_null(row.verification_metrics, key));
}
void tolerance(const char *key, double scale) {
  check(gpu_suite_verification_add_absolute_relative_threshold(
        &row, key, scale, options.abs_tolerance, options.rel_tolerance));
}
gpu_suite_json_value *threshold(const char *key, const char *method) {
  auto *value = gpu_suite_json_object();
  if (!value) check(GPU_SUITE_ERROR_NOMEM);
  check(gpu_suite_json_add_string(value, "method", method));
  check(gpu_suite_json_object_set(row.verification_thresholds, key, value));
  return value;
}
}

extern "C" void gpu_suite_f_initialize(int benchmark, int implementation) {
  gpu_suite_options_init(&options, static_cast<gpu_suite_benchmark_kind>(benchmark),
                         static_cast<gpu_suite_implementation>(implementation));
  if (implementation == 0) {
    std::snprintf(options.cpu_backend, sizeof options.cpu_backend, "%s", GPU_SUITE_FORTRAN_CPU_BACKEND);
    if (benchmark == 4 || benchmark == 5) {
      options.cpu_threads_effective = 1;
      std::snprintf(options.cpu_parallelism, sizeof options.cpu_parallelism, "serial");
    }
  }
}
extern "C" void gpu_suite_f_argument(const char *argument) { arguments.emplace_back(argument); }
extern "C" int gpu_suite_f_parse() {
  std::vector<char *> argv;
  for (auto &argument : arguments) argv.push_back(&argument[0]);
  auto result = gpu_suite_options_parse(&options, static_cast<int>(argv.size()), argv.data(), error, sizeof error);
  if (result == GPU_SUITE_PARSE_HELP) {
    gpu_suite_options_usage(stdout, argv[0], options.benchmark);
    return 1;
  }
  if (result != GPU_SUITE_PARSE_OK) { std::fprintf(stderr, "%s\n", error); return -1; }
  if (std::strcmp(GPU_SUITE_FORTRAN_COMPILER, "NVHPC") == 0) {
    const char *fpu_override = std::getenv("NVCOMPILER_FPU_STATE");
    if (fpu_override && fpu_override[0]) {
      std::fprintf(stderr, "unset NVCOMPILER_FPU_STATE: runtime overrides are not part of the Fortran numerical profile\n");
      return -1;
    }
  }
  if (options.implementation == GPU_SUITE_IMPLEMENTATION_CPU) {
    bool supported = std::strcmp(options.cpu_backend, GPU_SUITE_FORTRAN_CPU_BACKEND) == 0;
#ifdef GPU_SUITE_FORTRAN_FFTW
    supported = std::strcmp(options.cpu_backend, "cpu-fftw-serial") == 0;
#ifdef GPU_SUITE_HAVE_FFTW_THREADS
    supported = supported || std::strcmp(options.cpu_backend, "cpu-fftw-threaded") == 0;
#endif
#endif
    if (!supported) { std::fprintf(stderr, "selected Fortran CPU backend is not compiled in\n"); return -1; }
#ifdef GPU_SUITE_FORTRAN_FFTW
    const bool threaded = std::strcmp(options.cpu_backend, "cpu-fftw-threaded") == 0;
    const int effective = threaded ? options.cpu_threads : 1;
    const char *parallelism = threaded ? "threaded" : "serial";
    if ((options.cpu_threads_effective != 0 && options.cpu_threads_effective != effective) ||
        (std::strcmp(options.cpu_parallelism, "unknown") != 0 &&
         std::strcmp(options.cpu_parallelism, parallelism) != 0)) {
      std::fprintf(stderr, "Fortran FFTW thread metadata conflicts with selected API mode\n");
      return -1;
    }
    options.cpu_threads_effective = effective;
    std::snprintf(options.cpu_parallelism, sizeof options.cpu_parallelism, "%s", parallelism);
#endif
    if ((options.benchmark == 4 || options.benchmark == 5) &&
        (std::strcmp(options.cpu_parallelism, "serial") != 0 || options.cpu_threads_effective != 1)) {
      std::fprintf(stderr, "intrinsic CPU build requires serial / effective=1 (requested threads remain recorded)\n");
      return -1;
    }
  }
  check(gpu_suite_benchmark_writer_open(&writer, &options, error, sizeof error));
  opened = true;
#ifdef GPU_SUITE_FORTRAN_MKL
  mkl_set_num_threads(options.cpu_threads);
  mkl_get_version_string(library_version, sizeof library_version);
#elif defined(GPU_SUITE_FORTRAN_FFTW)
  std::snprintf(library_version, sizeof library_version, "%s", fftwf_version);
#else
  if (options.implementation == GPU_SUITE_IMPLEMENTATION_CPU)
    std::snprintf(library_version, sizeof library_version, "%s", GPU_SUITE_FORTRAN_COMPILER_VERSION);
#endif
  return 0;
}
extern "C" void gpu_suite_f_fail(const char *message) {
  std::fprintf(stderr, "%s\n", message);
  if (opened) {
    for (int trial = current_trial; trial < options.trials; ++trial) {
      initialize_row(trial);
      row.attempted = trial == current_trial;
      row.failure_origin = row.attempted ? "benchmark" : "prior-failure";
      row.status = row.attempted ? "failure" : "skipped";
      row.verification_status = "skipped";
      row.exit_code = row.attempted ? gpu_suite_optional_int_value(1) : gpu_suite_optional_int_null();
      row.message = message;
      check(gpu_suite_benchmark_writer_write(&writer, &row, error, sizeof error));
    }
    check(gpu_suite_benchmark_writer_close(&writer, error, sizeof error));
  }
  std::exit(1);
}
extern "C" int64_t gpu_suite_f_integer(int key) {
  uint64_t value = 0;
  switch (key) {
  case 1: value=options.size; break;
  case 2: value=options.batch; break;
  case 3: value=options.size_set ? options.size : options.m; break;
  case 4: value=options.size_set ? options.size : options.n; break;
  case 5: value=options.size_set ? options.size : options.k; break;
  case 6: case 7: value=options.size_set ? static_cast<uint64_t>(std::sqrt(static_cast<double>(options.size)))
                                      : (key==6 ? options.nx : options.ny); break;
  case 8: value=options.nrhs; break;
  case 9: value=options.seed; break;
  case 10: value=options.offset; break;
  case 11: return options.warmup;
  case 12: return options.repeat;
  case 13: return options.trials;
  case 14: return options.scope;
  case 15: return options.verify;
  case 16: return options.device;
  case 17: return options.cpu_threads;
  case 18: return std::strcmp(options.cpu_backend, "cpu-fftw-threaded") == 0;
  default: gpu_suite_f_fail("unknown integer option");
  }
  if (value > static_cast<uint64_t>(INT64_MAX)) gpu_suite_f_fail("option exceeds Fortran int64 range");
  return static_cast<int64_t>(value);
}
extern "C" double gpu_suite_f_real(int key) {
  switch (key) {
  case 1: return options.alpha; case 2: return options.beta;
  case 3: return options.abs_tolerance; case 4: return options.rel_tolerance;
  case 5: return options.sigma_multiplier; case 6: return options.expected_mean;
  case 7: return options.expected_second_central_moment;
  default: gpu_suite_f_fail("unknown real option"); return 0;
  }
}
extern "C" void gpu_suite_f_trial(int trial) {
  current_trial = trial; elapsed = 0; start_stamp[0]=0; end_stamp[0]=0;
  initialize_row(trial);
#ifdef GPU_SUITE_HAVE_CUDA_RUNTIME_METADATA
  if (!gpu_suite::apply_cuda_runtime_metadata(row, options.device)) gpu_suite_f_fail("CUDA metadata probe failed");
#endif
}
extern "C" void gpu_suite_f_start() {
  char stamp[GPU_SUITE_TIMESTAMP_CAPACITY];
  check(gpu_suite_measurement_start(stamp, &start_time, error, sizeof error));
  if (!start_stamp[0]) std::strcpy(start_stamp, stamp);
}
extern "C" void gpu_suite_f_end() {
  check(gpu_suite_measurement_end(&end_time, end_stamp, error, sizeof error));
  elapsed += gpu_suite_clock_elapsed(&start_time, &end_time);
}
extern "C" int gpu_suite_f_finish(const double *values, double scale, int getrf, int getrs) {
  bool passed = true;
  bool finite = true;
  if (options.verify) {
    check(gpu_suite_verification_reset(&row));
    switch (options.benchmark) {
    case GPU_SUITE_BENCHMARK_CUFFT:
      metric("dc_relative_error",values[0]); metric("non_dc_max_abs_error",values[1]);
      check(gpu_suite_verification_add_upper_bound_threshold(&row,"dc_relative_error","relative-upper-bound",options.rel_tolerance));
      check(gpu_suite_verification_add_upper_bound_threshold(&row,"non_dc_max_abs_error","absolute-upper-bound",options.abs_tolerance));
      row.verification_primary_metric="non_dc_max_abs_error";
      passed=std::isfinite(values[0]) && std::isfinite(values[1]) && values[0]<=options.rel_tolerance && values[1]<=options.abs_tolerance;
      break;
    case GPU_SUITE_BENCHMARK_CUBLAS: case GPU_SUITE_BENCHMARK_CUSPARSE:
      metric("max_abs_error", values[0]); tolerance("max_abs_error", scale);
      row.verification_primary_metric="max_abs_error";
      passed=std::isfinite(values[0]) && values[0]<=options.abs_tolerance+options.rel_tolerance*scale;
      break;
    case GPU_SUITE_BENCHMARK_CUSOLVER:
      metric("solution_relative_error",values[0]); metric("relative_residual",values[1]);
      tolerance("solution_relative_error",1); tolerance("relative_residual",1);
      row.verification_primary_metric="relative_residual";
      passed=std::isfinite(values[0]) && std::isfinite(values[1]) && values[0]<=options.abs_tolerance+options.rel_tolerance && values[1]<=options.abs_tolerance+options.rel_tolerance && getrf==0 && getrs==0;
      break;
    case GPU_SUITE_BENCHMARK_CURAND: {
      metric("observed_min",values[0]); metric("observed_max",values[1]);
      metric("sample_mean",values[2]); metric("second_central_moment_about_half",values[3]);
      auto *range=threshold("observed_range","inclusive-range");
      check(gpu_suite_json_add_double(range,"lower_bound",0)); check(gpu_suite_json_add_double(range,"upper_bound",1));
      check(gpu_suite_json_add_string(range,"backend_interval", options.implementation==0?"[0,1)":"(0,1]"));
      double mean_bound=options.sigma_multiplier*std::sqrt(1.0/(12.0*static_cast<double>(options.size)));
      double moment_bound=options.sigma_multiplier*std::sqrt(1.0/(180.0*static_cast<double>(options.size)));
      auto *mean=threshold("sample_mean","uniform-mean-sigma-bound");
      check(gpu_suite_json_add_double(mean,"expected_mean",options.expected_mean));
      check(gpu_suite_json_add_double(mean,"sigma_multiplier",options.sigma_multiplier));
      check(gpu_suite_json_add_double(mean,"absolute_bound",mean_bound));
      auto *moment=threshold("second_central_moment_about_half","uniform-second-central-moment-sigma-bound");
      check(gpu_suite_json_add_double(moment,"expected_second_central_moment",options.expected_second_central_moment));
      check(gpu_suite_json_add_double(moment,"sigma_multiplier",options.sigma_multiplier));
      check(gpu_suite_json_add_double(moment,"absolute_bound",moment_bound));
      row.verification_primary_metric="sample_mean";
      passed=std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]) && std::isfinite(values[3]) && values[0]>=0 && values[1]<=1 && std::abs(values[2]-options.expected_mean)<=mean_bound && std::abs(values[3]-options.expected_second_central_moment)<=moment_bound;
      break;
    }
    case GPU_SUITE_BENCHMARK_THRUST:
      metric("absolute_error", values[0]); tolerance("absolute_error", scale);
      row.verification_primary_metric="absolute_error";
      passed=std::isfinite(values[0]) && values[0]<=options.abs_tolerance+options.rel_tolerance*scale;
      break;
    }
  }
  if (options.benchmark==GPU_SUITE_BENCHMARK_CUSOLVER) {
    row.getrf_info=gpu_suite_optional_int_value(getrf);
    row.getrs_info=gpu_suite_optional_int_value(getrs);
    passed=passed && getrf==0 && getrs==0;
  }
  row.elapsed_total_sec=gpu_suite_optional_double_value(elapsed);
  row.elapsed_sec=gpu_suite_optional_double_value(elapsed/options.repeat);
  row.measurement_start_timestamp=start_stamp; row.measurement_end_timestamp=end_stamp;
  row.status=passed?"success":"failure";
  row.failure_origin=passed?nullptr:(options.verify?"verification":"benchmark");
  const int count = options.benchmark==GPU_SUITE_BENCHMARK_CURAND ? 4 :
      (options.benchmark==GPU_SUITE_BENCHMARK_CUFFT || options.benchmark==GPU_SUITE_BENCHMARK_CUSOLVER ? 2 : 1);
  for (int i=0; i<count; ++i) finite=finite && std::isfinite(values[i]);
  row.verification_status=options.verify?(passed?"pass":(finite?"failure":"nonfinite")):"skipped";
  row.exit_code=gpu_suite_optional_int_value(passed?0:1);
  row.message=passed?"":"Fortran numerical verification failed";
  check(gpu_suite_benchmark_writer_write(&writer,&row,error,sizeof error));
  ++current_trial;
  return passed?0:1;
}
extern "C" void gpu_suite_f_close(int exit_code) {
  if (row_live) gpu_suite_result_destroy(&row);
  if (opened) check(gpu_suite_benchmark_writer_close(&writer,error,sizeof error));
  std::exit(exit_code);
}
