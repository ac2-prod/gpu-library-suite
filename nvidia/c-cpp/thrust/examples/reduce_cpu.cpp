#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>

int main() {
  const std::size_t num_elem = std::size_t{1} << 24;
  const std::vector<double> values(num_elem, 1.0);
  double result = 0.0;
#ifdef GPU_SUITE_USE_OPENMP
#pragma omp parallel for reduction(+ : result)
  for (std::size_t index = 0; index < num_elem; ++index) {
    result += values[index] * values[index];
  }
#else
  result = std::transform_reduce(values.begin(), values.end(), 0.0,
                                 std::plus<double>(),
                                 [](double value) { return value * value; });
#endif
  std::printf("sum(values[i]^2) = %.17g (expected %zu)\n", result, num_elem);
  return result == static_cast<double>(num_elem) ? EXIT_SUCCESS : EXIT_FAILURE;
}
