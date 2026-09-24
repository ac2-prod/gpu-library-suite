! cuda entry point; computation is in solver_workload.F90 (Fortran).
program solver_gpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(3, 1, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program solver_gpu_bench
