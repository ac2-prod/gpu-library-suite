#include "gpu_suite/fft_bench_result.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "check failed: %s\n", #condition); std::abort(); } } while (0)

// CPU-only test of the shared verifier; no HIP types, runtime or devices.
struct Complex { float x, y; };
int main() {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.size = 8;
  options.batch = 3;
  options.verify = true;
  options.abs_tolerance = 1.0e-5;
  options.rel_tolerance = 1.0e-5;
  gpu_suite_result result;
  CHECK(gpu_suite_result_init(&result) == GPU_SUITE_OK);
  std::vector<Complex> output(24, {0.0F, 0.0F});
  for (std::size_t batch = 0; batch < 3; ++batch) output[batch * 8].x = 8.0F;
  auto check = [&](const char *status) {
    CHECK(gpu_suite_fft_bench::verify(result, options, output.data(), output.size()) == GPU_SUITE_OK);
    CHECK(std::strcmp(result.verification_status, status) == 0);
  };
  check("pass");
  for (std::size_t batch = 0; batch < 3; ++batch) {
    output[batch * 8].x = 7.0F;
    check("failure");
    output[batch * 8].x = 8.0F;
    output[batch * 8].y = 1.0F;
    check("failure");
    output[batch * 8].y = 0.0F;
    for (std::size_t frequency = 1; frequency < 8; ++frequency) {
      Complex &value = output[batch * 8 + frequency];
      value.x = 1.0F;
      check("failure");
      value.x = 0.0F;
      value.y = -1.0F;
      check("failure");
      value.y = 0.0F;
    }
  }
  for (float special : {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()}) {
    for (std::size_t index = 0; index < output.size(); ++index) {
      Complex saved = output[index];
      output[index].x = special;
      check("nonfinite");
      output[index] = saved;
      output[index].y = special;
      check("nonfinite");
      output[index] = saved;
    }
  }
  check("pass");
  CHECK(gpu_suite_fft_bench::verify(result, options, output.data(), 23) == GPU_SUITE_ERROR_INVALID);
  CHECK(gpu_suite_fft_bench::verify<Complex>(result, options, nullptr, 24) == GPU_SUITE_ERROR_INVALID);
  options.size = 0;
  CHECK(gpu_suite_fft_bench::verify(result, options, output.data(), 24) == GPU_SUITE_ERROR_INVALID);
  gpu_suite_result_destroy(&result);
}
