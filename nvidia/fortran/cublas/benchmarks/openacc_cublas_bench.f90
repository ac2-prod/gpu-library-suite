! openacc entry point; computation is in blas_workload.F90 (Fortran).
program openacc_cublas_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(1, 2, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program openacc_cublas_bench
