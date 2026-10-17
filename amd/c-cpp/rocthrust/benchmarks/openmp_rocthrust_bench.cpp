#define GPU_SUITE_REDUCE_OPENMP 1
#include "reduce_gpu_impl.hpp"
int main(int argc, char **argv) { return gpu_suite_rocthrust::run(argc, argv); }
