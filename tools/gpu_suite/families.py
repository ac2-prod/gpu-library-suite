"""Library families and mathematical counterparts; never alias artifact identity."""

AMD_COUNTERPARTS = {
    "rocfft": "cufft",
    "rocblas": "cublas",
    "rocsparse": "cusparse",
    "rocsolver": "cusolver",
    "rocrand": "curand",
    "rocthrust": "thrust",
}
AMD_BENCHMARKS = tuple(AMD_COUNTERPARTS)


def mathematical_family(benchmark: str) -> str:
    return AMD_COUNTERPARTS.get(benchmark, benchmark)
