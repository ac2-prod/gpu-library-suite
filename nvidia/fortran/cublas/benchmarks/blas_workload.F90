module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use cudafor
    use cublas_v2
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
    real(c_double), allocatable :: mat_A(:,:), mat_B(:,:), mat_C(:,:)
#if GPU_SUITE_IMPL == 1
    real(c_double), device, allocatable :: d_A(:,:), d_B(:,:), d_C(:,:)
#endif
#if GPU_SUITE_IMPL != 0
    type(cublasHandle) :: handle
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: count
        count=checked_product(o%m,o%k); count=checked_product(o%k,o%n); count=checked_product(o%m,o%n)
#if GPU_SUITE_IMPL != 0
        call check(cudaSetDevice(o%device),'cudaSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
        allocate(mat_A(o%m,o%k),mat_B(o%k,o%n),mat_C(o%m,o%n),stat=status)
        call check(status,'host allocation')
    end subroutine
    subroutine reset()
        mat_A=1; mat_B=1; mat_C=1
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 1
        integer :: status
        allocate(d_A(o%m,o%k),d_B(o%k,o%n),d_C(o%m,o%n),stat=status)
        call check(status,'device allocation')
        call restore()
#elif GPU_SUITE_IMPL == 2
        !$acc enter data copyin(mat_A,mat_B,mat_C)
        data_present=.true.
#endif
#if GPU_SUITE_IMPL != 0
        call check(cublasCreate(handle),'cublasCreate')
        call check(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH),'cublasSetMathMode')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(d_A,mat_A,size(mat_A,kind=c_int64_t),cudaMemcpyHostToDevice),'copy A')
        call check(cudaMemcpy(d_B,mat_B,size(mat_B,kind=c_int64_t),cudaMemcpyHostToDevice),'copy B')
        call check(cudaMemcpy(d_C,mat_C,size(mat_C,kind=c_int64_t),cudaMemcpyHostToDevice),'copy initial C')
#elif GPU_SUITE_IMPL == 2
        !$acc update device(mat_A,mat_B,mat_C)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        call dgemm('N','N',o%m,o%n,o%k,o%alpha,mat_A,o%m,mat_B,o%k,o%beta,mat_C,o%m)
#elif GPU_SUITE_IMPL == 1
        call check(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,o%m,o%n,o%k, &
                   o%alpha,d_A,o%m,d_B,o%k,o%beta,d_C,o%m),'cublasDgemm')
#else
        integer :: status
        !$acc host_data use_device(mat_A,mat_B,mat_C)
        status=cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,o%m,o%n,o%k, &
                          o%alpha,mat_A,o%m,mat_B,o%k,o%beta,mat_C,o%m)
        !$acc end host_data
        call check(status,'cublasDgemm')
#endif
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(cudaDeviceSynchronize(),'cudaDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(mat_C,d_C,size(mat_C,kind=c_int64_t),cudaMemcpyDeviceToHost),'copy result C')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data copyout(mat_C) delete(mat_A,mat_B)
            data_present=.false.
        else
            !$acc update self(mat_C)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        call check(cublasDestroy(handle),'cublasDestroy')
#endif
#if GPU_SUITE_IMPL == 1
        deallocate(d_A,d_B,d_C)
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$acc exit data delete(mat_A,mat_B,mat_C)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        integer :: i, updates
        real(c_double) :: expected
        getrf=0; getrs=0; metrics=0
        updates=1
        if (o%scope==0) updates=o%repeat
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
