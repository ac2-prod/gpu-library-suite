! cuda entry point; computation is in reduce_workload.F90 (Fortran).
program reduce_gpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(5, 1, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program reduce_gpu_bench
