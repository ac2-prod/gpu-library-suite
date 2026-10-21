module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use hipfort, only: hipSetDevice,hipMalloc,hipFree,hipMemcpy,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use hipfort_rocblas, only: rocblas_create_handle,rocblas_destroy_handle,rocblas_operation_none, &
                              rocblas_dgemm
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
#if GPU_SUITE_IMPL == 2
    interface
        function check_mapped_device(pointer,device) bind(C,name='gpu_suite_f_check_mapped_device') result(status)
            import c_ptr,c_int
            type(c_ptr), value :: pointer
            integer(c_int), value :: device
            integer(c_int) :: status
        end function
    end interface
#endif
    real(c_double), allocatable, target :: mat_A(:,:),mat_B(:,:),mat_C(:,:)
#if GPU_SUITE_IMPL != 0
    type(c_ptr) :: handle=c_null_ptr
#endif
#if GPU_SUITE_IMPL == 1
    type(c_ptr) :: d_A=c_null_ptr,d_B=c_null_ptr,d_C=c_null_ptr
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: count
#if GPU_SUITE_IMPL == 2
        logical :: initial
        integer :: omp_device
#endif
        count=checked_product(o%m,o%k); count=checked_product(o%k,o%n); count=checked_product(o%m,o%n)
#if GPU_SUITE_IMPL != 0
        call check(hipSetDevice(int(o%device,c_int)),'hipSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        omp_device=omp_get_default_device()
        call require(omp_device >= 0 .and. omp_device < omp_get_num_devices(),'OpenMP target unavailable')
        initial=.true.
        !$omp target map(from: initial)
        initial=omp_is_initial_device()
        !$omp end target
        call require(.not. initial,'OpenMP host fallback rejected')
#endif
        allocate(mat_A(o%m,o%k),mat_B(o%k,o%n),mat_C(o%m,o%n),stat=status)
        call check(status,'host allocation')
        mat_A=1; mat_B=1
    end subroutine
    subroutine reset()
        mat_C=1
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_A,size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate A')
        call check(hipMalloc(d_B,size(mat_B,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate B')
        call check(hipMalloc(d_C,size(mat_C,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate C')
        call check(hipMemcpy(d_A,c_loc(mat_A),size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'copy A')
        call check(hipMemcpy(d_B,c_loc(mat_B),size(mat_B,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'copy B')
        if (o%scope /= 0) call restore()
#elif GPU_SUITE_IMPL == 2
        call check(rocblas_create_handle(handle),'rocblas_create_handle')
        !$omp target enter data map(to: mat_A,mat_B,mat_C)
        data_present=.true.
#endif
#if GPU_SUITE_IMPL == 1
        call check(rocblas_create_handle(handle),'rocblas_create_handle')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(d_C,c_loc(mat_C),size(mat_C,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'restore C')
#elif GPU_SUITE_IMPL == 2
        !$omp target update to(mat_C)
#endif
    end subroutine
    subroutine operation(repetitions,timed)
        integer, intent(in) :: repetitions
        logical, intent(in) :: timed
        integer :: repetition
#if GPU_SUITE_IMPL == 2
        ! One address region for the complete sequence, matching C/C++.
        !$omp target data use_device_addr(mat_A,mat_B,mat_C)
        call check(check_mapped_device(c_loc(mat_A),int(o%device,c_int)),'mapped mat_A device')
        call check(check_mapped_device(c_loc(mat_B),int(o%device,c_int)),'mapped mat_B device')
        call check(check_mapped_device(c_loc(mat_C),int(o%device,c_int)),'mapped mat_C device')
#endif
        if (timed) call start_timer()
        do repetition=1,repetitions
#if GPU_SUITE_IMPL == 0
        call dgemm('N','N',o%m,o%n,o%k,o%alpha,mat_A,o%m,mat_B,o%k,o%beta,mat_C,o%m)
#elif GPU_SUITE_IMPL == 1
        call check(rocblas_dgemm(handle,rocblas_operation_none,rocblas_operation_none,o%m,o%n,o%k, &
                                o%alpha,d_A,o%m,d_B,o%k,o%beta,d_C,o%m),'rocblas_dgemm')
#else
        call check(rocblas_dgemm(handle,rocblas_operation_none,rocblas_operation_none,o%m,o%n,o%k, &
                                o%alpha,c_loc(mat_A),o%m,c_loc(mat_B),o%k,o%beta,c_loc(mat_C),o%m),'rocblas_dgemm')
#endif
        end do
#if GPU_SUITE_IMPL == 2
        !$omp end target data
#endif
        ! Library warm-up submissions are completed once by the driver.
        if (timed .or. o%scope /= 0) call synchronize()
        if (timed) call end_timer()
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(hipDeviceSynchronize(),'hipDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(c_loc(mat_C),d_C,size(mat_C,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyDeviceToHost),'retrieve C')
#elif GPU_SUITE_IMPL == 2
        ! The shared driver synchronizes HIP before this OpenMP transfer.
        if (end_to_end) then
            !$omp target exit data map(from: mat_C) map(delete: mat_A,mat_B)
            data_present=.false.
        else
            !$omp target update from(mat_C)
            call check(hipDeviceSynchronize(),'OpenMP result completion')
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        call check(rocblas_destroy_handle(handle),'rocblas_destroy_handle')
        handle=c_null_ptr
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_C),'free C'); call check(hipFree(d_B),'free B'); call check(hipFree(d_A),'free A')
        d_A=c_null_ptr; d_B=c_null_ptr; d_C=c_null_ptr
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$omp target exit data map(delete: mat_A,mat_B,mat_C)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        integer :: i,updates
        real(c_double) :: expected
        metrics=0; getrf=0; getrs=0
        updates=1
        if (o%scope == 0) updates=o%repeat
        expected=1
        do i=1,updates
            expected=o%alpha*real(o%k,c_double)+o%beta*expected
        end do
        call require(ieee_is_finite(expected),'nonfinite analytic DGEMM reference')
        scale=abs(expected)
        if (.not. all(ieee_is_finite(mat_C))) then
            metrics(1)=nan_value()
        else
            metrics(1)=maxval(abs(mat_C-expected))
        end if
    end subroutine
    subroutine host_cleanup()
        deallocate(mat_A,mat_B,mat_C)
    end subroutine
end module gpu_suite_workload
