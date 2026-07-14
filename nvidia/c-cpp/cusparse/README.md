# cuSPARSE C/C++ examples and benchmarks

The examples implement the canonical FP64 five-point Poisson CSR SpMV. The
matrix definition and exact nonzero count are owned by
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md); the
benchmark sweep, repeat-state semantics, and verification rules are owned by
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

The production CPU target uses oneMKL Sparse only after a compile-and-link probe
has exercised descriptor creation, the SpMV hint and optimization calls, SpMV,
and destruction. The separately named reference CSR backend is never a silent
replacement. CUDA and OpenACC targets require actual cuSPARSE symbols.

The canonical source-stem/target/executable names are `sparse_cpu`,
`sparse_gpu`, `openacc_cusparse`, `sparse_cpu_bench`, `sparse_gpu_bench`, and
`openacc_cusparse_bench`.

In OpenACC variants the data region owns CSR and vector storage. Descriptors are
created only after `host_data use_device` translates those addresses. The only
manual CUDA allocation is the cuSPARSE workspace, which is not OpenACC-managed.
