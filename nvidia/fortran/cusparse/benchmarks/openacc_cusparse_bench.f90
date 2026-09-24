! openacc entry point; computation is in sparse_workload.F90 (Fortran).
program openacc_cusparse_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(2, 2, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program openacc_cusparse_bench
