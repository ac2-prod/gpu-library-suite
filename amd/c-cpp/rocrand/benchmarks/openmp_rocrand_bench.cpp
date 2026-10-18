#define GPU_SUITE_RAND_OPENMP 1
#include "rand_gpu_impl.hpp"
int main(int argc, char **argv) { return gpu_suite_rocrand::run(argc, argv); }
