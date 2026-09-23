! cpu entry point; computation is in rand_workload.F90 (Fortran).
program rand_cpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(4, 0, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program rand_cpu_bench
