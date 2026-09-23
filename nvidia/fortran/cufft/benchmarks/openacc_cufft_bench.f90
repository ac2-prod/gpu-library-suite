! openacc entry point; computation is in fft_workload.F90 (Fortran).
program openacc_cufft_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(0, 2, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program openacc_cufft_bench
