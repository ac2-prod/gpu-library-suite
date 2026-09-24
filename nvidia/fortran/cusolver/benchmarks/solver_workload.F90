module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use cudafor
    use cusolverDn
    use cublas_v2, only: CUBLAS_OP_N
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
    interface
        subroutine make_dense_system(n,nrhs,mat_A,rhs_B)
            integer, intent(in) :: n,nrhs
            real(8), intent(out) :: mat_A(n,n),rhs_B(n,nrhs)
        end subroutine
    end interface
    real(c_double), allocatable :: mat_A(:,:),rhs_B(:,:)
    integer, allocatable :: ipiv(:)
    integer :: info_getrf=-1,info_getrs=-1
#if GPU_SUITE_IMPL != 0
    type(cusolverDnHandle) :: handle
    real(c_double), device, allocatable :: workspace(:)
    integer :: lwork
#endif
#if GPU_SUITE_IMPL == 1
    real(c_double), device, allocatable :: d_A(:,:),d_B(:,:)
    integer, device, allocatable :: d_ipiv(:)
    integer, device :: d_getrf,d_getrs
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: count
        count=checked_product(o%n,o%n); count=checked_product(o%n,o%nrhs)
        call require(o%repeat==1,'solver benchmark requires repeat=1')
#if GPU_SUITE_IMPL != 0
        call check(cudaSetDevice(o%device),'cudaSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
        allocate(mat_A(o%n,o%n),rhs_B(o%n,o%nrhs),ipiv(o%n),stat=status)
        call check(status,'host allocation')
    end subroutine
    subroutine reset()
        call make_dense_system(o%n,o%nrhs,mat_A,rhs_B)
        ipiv=0; info_getrf=-1; info_getrs=-1
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL != 0
        integer :: status
        call check(cusolverDnCreate(handle),'cusolverDnCreate')
#if GPU_SUITE_IMPL == 1
        allocate(d_A(o%n,o%n),d_B(o%n,o%nrhs),d_ipiv(o%n),stat=status)
        call check(status,'device allocation')
        call restore()
        call check(cusolverDnDgetrf_bufferSize(handle,o%n,o%n,d_A,o%n,lwork),'getrf_bufferSize')
#else
        !$acc enter data copyin(mat_A,rhs_B,ipiv,info_getrf,info_getrs)
        data_present=.true.
        !$acc host_data use_device(mat_A)
        status=cusolverDnDgetrf_bufferSize(handle,o%n,o%n,mat_A,o%n,lwork)
        !$acc end host_data
        call check(status,'getrf_bufferSize')
#endif
        call require(lwork>=0,'negative cuSOLVER workspace size')
        allocate(workspace(max(1,lwork)),stat=status)
        call check(status,'cuSOLVER workspace allocation')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(d_A,mat_A,size(mat_A,kind=c_int64_t),cudaMemcpyHostToDevice),'copy A')
        call check(cudaMemcpy(d_B,rhs_B,size(rhs_B,kind=c_int64_t),cudaMemcpyHostToDevice),'copy B')
        call check(cudaMemcpy(d_ipiv,ipiv,o%n,cudaMemcpyHostToDevice),'restore pivots')
        call check(cudaMemcpy(d_getrf,info_getrf,1,cudaMemcpyHostToDevice),'restore getrf info')
        call check(cudaMemcpy(d_getrs,info_getrs,1,cudaMemcpyHostToDevice),'restore getrs info')
#elif GPU_SUITE_IMPL == 2
        !$acc update device(mat_A,rhs_B,ipiv,info_getrf,info_getrs)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        ! Staged benchmark API, unlike the dgesv teaching example.
        call dgetrf(o%n,o%n,mat_A,o%n,ipiv,info_getrf)
        if (info_getrf==0) call dgetrs('N',o%n,o%nrhs,mat_A,o%n,ipiv,rhs_B,o%n,info_getrs)
#elif GPU_SUITE_IMPL == 1
        call check(cusolverDnDgetrf(handle,o%n,o%n,d_A,o%n,workspace,d_ipiv,d_getrf),'cusolverDnDgetrf')
        call check(cusolverDnDgetrs(handle,CUBLAS_OP_N,o%n,o%nrhs,d_A,o%n,d_ipiv,d_B,o%n,d_getrs), &
                   'cusolverDnDgetrs')
#else
        integer :: status
        !$acc host_data use_device(mat_A,rhs_B,ipiv,info_getrf,info_getrs)
        status=cusolverDnDgetrf(handle,o%n,o%n,mat_A,o%n,workspace,ipiv,info_getrf)
        call check(status,'cusolverDnDgetrf')
        status=cusolverDnDgetrs(handle,CUBLAS_OP_N,o%n,o%nrhs,mat_A,o%n,ipiv,rhs_B,o%n,info_getrs)
        call check(status,'cusolverDnDgetrs')
        !$acc end host_data
#endif
        ! Retain both device info values; inspect after synchronization/download.
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(cudaDeviceSynchronize(),'cudaDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(rhs_B,d_B,size(rhs_B,kind=c_int64_t),cudaMemcpyDeviceToHost),'copy solution')
        call check(cudaMemcpy(info_getrf,d_getrf,1,cudaMemcpyDeviceToHost),'copy getrf info')
        call check(cudaMemcpy(info_getrs,d_getrs,1,cudaMemcpyDeviceToHost),'copy getrs info')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data copyout(rhs_B,info_getrf,info_getrs) delete(mat_A,ipiv)
            data_present=.false.
        else
            !$acc update self(rhs_B,info_getrf,info_getrs)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        deallocate(workspace)
        call check(cusolverDnDestroy(handle),'cusolverDnDestroy')
#endif
#if GPU_SUITE_IMPL == 1
        deallocate(d_A,d_B,d_ipiv)
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$acc exit data delete(mat_A,rhs_B,ipiv,info_getrf,info_getrs)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        integer :: rhs
        real(c_double) :: residual,denominator
        getrf=info_getrf; getrs=info_getrs; metrics=0; scale=1
        if (.not. all(ieee_is_finite(rhs_B))) then
            metrics(1:2)=nan_value()
            return
        end if
        metrics(1)=maxval(abs(rhs_B-1.0_c_double))
        do rhs=1,o%nrhs
            ! A = n*I + ones; this evaluates the original residual without
            ! retaining or regenerating a second n*n matrix in the timed path.
            residual=maxval(abs(real(o%n,c_double)*rhs_B(:,rhs)+sum(rhs_B(:,rhs))-2.0_c_double*o%n))
            ! Same denominator as the C/C++ verifier: ||A||inf + ||B||max;
            ! the known reference solution has norm one.
            denominator=4.0_c_double*o%n
            if (.not. ieee_is_finite(residual) .or. .not. ieee_is_finite(denominator)) then
                metrics(2)=nan_value()
                return
            end if
            metrics(2)=max(metrics(2),residual/denominator)
        end do
    end subroutine
    subroutine host_cleanup()
        deallocate(mat_A,rhs_B,ipiv)
    end subroutine
end module gpu_suite_workload
