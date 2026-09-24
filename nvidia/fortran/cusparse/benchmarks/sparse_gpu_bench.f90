! cuda entry point; computation is in sparse_workload.F90 (Fortran).
program sparse_gpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(2, 1, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program sparse_gpu_bench
