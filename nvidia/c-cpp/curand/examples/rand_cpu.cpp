#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
  const std::size_t num_rand = std::size_t{1} << 24;
  const std::uint64_t seed = 1234;
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  std::vector<double> values(num_rand);
  for (double &value : values) {
    value = distribution(engine);
  }
  const auto bounds = std::minmax_element(values.begin(), values.end());
  std::printf("std::mt19937_64 generated %zu uniform doubles in [0,1); "
              "observed range %.17g to %.17g\n",
              num_rand, *bounds.first, *bounds.second);
  return *bounds.first >= 0.0 && *bounds.second <= 1.0 ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
