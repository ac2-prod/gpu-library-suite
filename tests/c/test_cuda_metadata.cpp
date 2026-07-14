#include "gpu_suite/cuda_metadata.hpp"

int main() {
  gpu_suite_result result{};
  return gpu_suite::apply_cuda_runtime_metadata(result, 0) ? 0 : 1;
}
