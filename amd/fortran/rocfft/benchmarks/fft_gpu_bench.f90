! The workload is native Fortran; the shared bridge owns CLI/timing/result output.
program fft_gpu_bench
    use gpu_suite_fortran, only: run_benchmark
    use gpu_suite_workload
    implicit none
    call run_benchmark(0, 1, host_setup, reset, setup, restore, operation, &
                       synchronize, download, cleanup, verify, host_cleanup)
end program fft_gpu_bench
