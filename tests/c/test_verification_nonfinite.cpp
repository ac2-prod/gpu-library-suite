#include "gpu_suite/benchmark.hpp"
#include "fft_bench_result.hpp"
#include "rand_bench_common.hpp"
#include "reduce_bench_common.hpp"
#include "solver_bench_common.hpp"
#include "sparse_bench_common.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct ComplexValue {
  float x;
  float y;
};

gpu_suite_result make_result() {
  gpu_suite_result result;
  assert(gpu_suite_result_init(&result) == GPU_SUITE_OK);
  return result;
}

void assert_null_metrics(const gpu_suite_result &result) {
  char *serialized = nullptr;
  std::size_t length = 0;
  char error[256] = {0};
  assert(std::strcmp(result.verification_status, "nonfinite") == 0);
  assert(gpu_suite_json_serialize(result.verification_metrics, &serialized,
                                  &length, error,
                                  sizeof(error)) == GPU_SUITE_OK);
  assert(serialized != nullptr);
  assert(length > 0U);
  assert(std::strstr(serialized, "null") != nullptr);
  assert(std::strstr(serialized, "NaN") == nullptr);
  assert(std::strstr(serialized, "Infinity") == nullptr);
  std::free(serialized);
}

void test_fft(double special) {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.verify = true;
  options.size = 2U;
  options.batch = 1U;
  ComplexValue output[2] = {{2.0F, 0.0F},
                            {static_cast<float>(special), 0.0F}};
  gpu_suite_result result = make_result();
  assert(gpu_suite_fft_bench::verify(result, options, output, 2U) ==
         GPU_SUITE_OK);
  assert_null_metrics(result);
  gpu_suite_result_destroy(&result);
}

void test_random(double special) {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CURAND,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.verify = true;
  std::vector<double> values = {0.25, special, 0.75};
  gpu_suite_result result = make_result();
  assert(gpu_suite_curand::set_random_verification(
             result, options, values, "test-generator", "[0,1)") ==
         gpu_suite::VerificationOutcome::failure);
  assert_null_metrics(result);
  gpu_suite_result_destroy(&result);
}

void test_reduction(double special) {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.verify = true;
  gpu_suite_result result = make_result();
  assert(gpu_suite_thrust::set_reduction_verification(
             result, options, special, 4U) ==
         gpu_suite::VerificationOutcome::failure);
  assert_null_metrics(result);
  gpu_suite_result_destroy(&result);
}

void test_sparse(double special) {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSPARSE,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.verify = true;
  std::vector<double> values(4U, 3.0);
  values[1] = special;
  gpu_suite_result result = make_result();
  assert(gpu_suite_cusparse::set_spmv_verification(
             result, options, values, 2, 2, 1) ==
         gpu_suite::VerificationOutcome::failure);
  assert_null_metrics(result);
  gpu_suite_result_destroy(&result);
}

void test_solver(double special) {
  gpu_suite_options options;
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSOLVER,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  options.verify = true;
  std::vector<double> solution = {1.0, special};
  const std::vector<double> matrix = {3.0, 1.0, 1.0, 3.0};
  const std::vector<double> rhs = {4.0, 4.0};
  gpu_suite_result result = make_result();
  assert(gpu_suite_cusolver::set_solver_verification(
             result, options, solution, matrix, rhs, 2, 1) ==
         gpu_suite::VerificationOutcome::failure);
  assert_null_metrics(result);
  gpu_suite_result_destroy(&result);
}

} // namespace

int main() {
  const double values[] = {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
  for (double special : values) {
    test_fft(special);
    test_random(special);
    test_reduction(special);
    test_sparse(special);
    test_solver(special);
  }
  return 0;
}
