#include "gpu_suite/cli.h"

#include "test_support.h"
#include <string.h>

#define ARRAY_COUNT(values) ((int)(sizeof(values) / sizeof((values)[0])))

static gpu_suite_parse_result parse(gpu_suite_options *options, int argc,
                                    char **argv) {
  char error[256] = {0};
  return gpu_suite_options_parse(options, argc, argv, error, sizeof(error));
}

int main(void) {
  gpu_suite_options options;
  char *valid[] = {"fft_cpu_bench",
                   "--size",
                   "256",
                   "--batch",
                   "8",
                   "--warmup",
                   "0",
                   "--repeat",
                   "2",
                   "--trials",
                   "3",
                   "--scope",
                   "end-to-end",
                   "--verify",
                   "false",
                   "--run-id",
                   "run.1",
                   "--implementation-order",
                   "openacc,cuda,cpu",
                   "--cpu-threads",
                   "48",
                   "--cpu-threads-effective",
                   "4",
                   "--cpu-backend-role",
                   "production",
                   "--series-role",
                   "primary",
                   "--cpu-parallelism",
                   "threaded"};
  char *duplicate[] = {"bench", "--size", "16", "--size", "32"};
  char *bad_run[] = {"bench", "--size", "16", "--run-id", "../bad"};
  char *bad_bool[] = {"bench", "--size", "16", "--verify", "True"};
  char *bad_order[] = {"bench", "--size", "16", "--implementation-order",
                       "cpu,cpu,cuda"};
  char *blas_bad[] = {"bench", "--size", "16",  "--m", "16",
                      "--n",   "16",     "--k", "16"};
  char *blas_ok[] = {"bench", "--m", "4", "--n", "5", "--k", "6"};
  char *sparse_bad[] = {"bench", "--size", "15"};
  char *solver_bad[] = {"bench", "--size", "64", "--repeat", "2"};
  char *help[] = {"bench", "--help"};

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(valid), valid) == GPU_SUITE_PARSE_OK);
  CHECK(options.size == 256U && options.batch == 8U);
  CHECK(options.warmup == 0 && options.repeat == 2 && options.trials == 3);
  CHECK(options.scope == GPU_SUITE_SCOPE_END_TO_END && !options.verify);
  CHECK(strcmp(options.run_id, "run.1") == 0);
  CHECK(options.cpu_threads == 48 && options.cpu_threads_effective == 4);
  CHECK(strcmp(options.cpu_backend_role, "production") == 0);
  CHECK(strcmp(options.series_role, "primary") == 0);
  CHECK(strcmp(options.cpu_parallelism, "threaded") == 0);

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(duplicate), duplicate) ==
        GPU_SUITE_PARSE_ERROR);
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(bad_run), bad_run) ==
        GPU_SUITE_PARSE_ERROR);
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(bad_bool), bad_bool) ==
        GPU_SUITE_PARSE_ERROR);
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUFFT,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(bad_order), bad_order) ==
        GPU_SUITE_PARSE_ERROR);

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(ARRAY_COUNT(blas_bad) == 9);
  CHECK(parse(&options, ARRAY_COUNT(blas_bad), blas_bad) ==
        GPU_SUITE_PARSE_ERROR);
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUBLAS,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(blas_ok), blas_ok) == GPU_SUITE_PARSE_OK);

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSPARSE,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(sparse_bad), sparse_bad) ==
        GPU_SUITE_PARSE_ERROR);
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_CUSOLVER,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(solver_bad), solver_bad) ==
        GPU_SUITE_PARSE_ERROR);

  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_THRUST,
                         GPU_SUITE_IMPLEMENTATION_CPU);
  CHECK(parse(&options, ARRAY_COUNT(help), help) == GPU_SUITE_PARSE_HELP);
  CHECK(gpu_suite_run_id_validate("a-b_c.1"));
  CHECK(!gpu_suite_run_id_validate(".hidden"));
  CHECK(!gpu_suite_run_id_validate("a..b"));
  CHECK(!gpu_suite_run_id_validate("a/b"));
  return 0;
}

#undef ARRAY_COUNT
