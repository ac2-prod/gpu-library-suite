#include "gpu_suite/gpu_suite.h"

#include "test_support.h"

int main() {
  gpu_suite_options options{};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.size = 16;
  options.size_set = true;
  CHECK(gpu_suite_options_validate(&options, nullptr, 0) == GPU_SUITE_OK);
  return 0;
}
