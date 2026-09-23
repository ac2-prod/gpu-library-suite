! cpu entry point; computation is in fft_workload.F90 (Fortran).
program fft_cpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(0, 0, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program fft_cpu_bench
