module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use hipfort, only: hipSetDevice,hipMalloc,hipFree,hipMemcpy,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use hipfort_rocblas, only: rocblas_create_handle,rocblas_destroy_handle,rocblas_operation_none
    use hipfort_rocblas, only: rocblas_start_device_memory_size_query,rocblas_stop_device_memory_size_query, &
                              rocblas_set_workspace,rocblas_status_success, &
                              rocblas_status_size_increased,rocblas_status_size_unchanged
    use hipfort_rocsolver, only: rocsolver_dgetrs
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
#if GPU_SUITE_IMPL != 0
    ! Use the public LP64 C ABI: the selected hipFORT getrf generic does not
    ! accept this program's explicit device C pointers.
    interface
        function rocsolver_dgetrf(handle,m,n,a,lda,ipiv,info) bind(C,name='rocsolver_dgetrf') result(status)
            import c_ptr,c_int
            type(c_ptr), value :: handle,a,ipiv,info
            integer(c_int), value :: m,n,lda
            integer(c_int) :: status
        end function
    end interface
#endif
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
    interface
        subroutine make_dense_system(n,nrhs,mat_A,rhs_B)
            import c_double
            integer, intent(in) :: n,nrhs
            real(c_double), intent(out) :: mat_A(n,n),rhs_B(n,nrhs)
        end subroutine
    end interface
    real(c_double), allocatable, target :: mat_A(:,:),rhs_B(:,:)
    integer(c_int), allocatable, target :: ipiv(:)
    integer(c_int), target :: info_getrf=-1
    integer :: info_getrs=-1
#if GPU_SUITE_IMPL != 0
    type(c_ptr) :: handle=c_null_ptr,workspace=c_null_ptr
#endif
#if GPU_SUITE_IMPL == 1
    type(c_ptr) :: d_A=c_null_ptr,d_B=c_null_ptr,d_ipiv=c_null_ptr,d_info=c_null_ptr
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
        count=checked_product(o%n,o%n); count=checked_product(o%n,o%nrhs)
        call require(o%repeat == 1,'solver benchmark requires repeat=1')
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
        allocate(mat_A(o%n,o%n),rhs_B(o%n,o%nrhs),ipiv(o%n),stat=status)
        call check(status,'host allocation')
    end subroutine
    subroutine reset()
        call make_dense_system(o%n,o%nrhs,mat_A,rhs_B)
        ipiv=0; info_getrf=-1; info_getrs=-1
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL != 0
        integer(c_int) :: factor_status,solve_status
        integer(c_size_t) :: bytes
        call check(rocblas_create_handle(handle),'rocblas_create_handle')
        call check(rocblas_start_device_memory_size_query(handle),'start workspace query')
        factor_status=rocsolver_dgetrf(handle,o%n,o%n,c_null_ptr,o%n,c_null_ptr,c_null_ptr)
        solve_status=rocsolver_dgetrs(handle,rocblas_operation_none,o%n,o%nrhs,c_null_ptr,o%n, &
                                     c_null_ptr,c_null_ptr,o%n)
        ! hipFORT receives this host size_t output by reference, not as a C pointer.
        call check(rocblas_stop_device_memory_size_query(handle,bytes),'stop workspace query')
        call require(query_ok(factor_status) .and. query_ok(solve_status),'rocSOLVER workspace query failed')
        call require(bytes >= 0_c_size_t,'invalid workspace size')
        ! rocBLAS treats zero-sized user workspace as automatic; allocate at least
        ! one byte so library-scope execution does not allocate workspace silently.
        bytes=max(1_c_size_t,bytes)
        call check(hipMalloc(workspace,bytes),'allocate solver workspace')
        call check(rocblas_set_workspace(handle,workspace,bytes),'set solver workspace')
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_A,size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate A')
        call check(hipMalloc(d_B,size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate B')
        call check(hipMalloc(d_ipiv,int(o%n,c_size_t)*c_sizeof(info_getrf)),'allocate pivots')
        call check(hipMalloc(d_info,c_sizeof(info_getrf)),'allocate info')
        call restore()
#elif GPU_SUITE_IMPL == 2
        !$omp target enter data map(to: mat_A,rhs_B,ipiv,info_getrf)
        data_present=.true.
        !$omp target data use_device_addr(mat_A,rhs_B,ipiv,info_getrf)
        call check(check_mapped_device(c_loc(mat_A),int(o%device,c_int)),'mapped mat_A device')
        call check(check_mapped_device(c_loc(rhs_B),int(o%device,c_int)),'mapped rhs_B device')
        call check(check_mapped_device(c_loc(ipiv),int(o%device,c_int)),'mapped ipiv device')
        call check(check_mapped_device(c_loc(info_getrf),int(o%device,c_int)),'mapped info_getrf device')
        !$omp end target data
#endif
    end subroutine
#if GPU_SUITE_IMPL != 0
    logical function query_ok(status)
        integer(c_int), intent(in) :: status
        query_ok=status == rocblas_status_success .or. status == rocblas_status_size_increased .or. &
                 status == rocblas_status_size_unchanged
    end function
#endif
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(d_A,c_loc(mat_A),size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'restore A')
        call check(hipMemcpy(d_B,c_loc(rhs_B),size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'restore B')
        call check(hipMemcpy(d_ipiv,c_loc(ipiv),int(o%n,c_size_t)*c_sizeof(info_getrf), &
                             hipMemcpyHostToDevice),'restore pivots')
        call check(hipMemcpy(d_info,c_loc(info_getrf),c_sizeof(info_getrf),hipMemcpyHostToDevice),'restore info')
#elif GPU_SUITE_IMPL == 2
        !$omp target update to(mat_A,rhs_B,ipiv,info_getrf)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        ! The benchmark uses staged getrf/getrs; the teaching CPU example uses DGESV.
        call dgetrf(o%n,o%n,mat_A,o%n,ipiv,info_getrf)
        if (info_getrf == 0) call dgetrs('N',o%n,o%nrhs,mat_A,o%n,ipiv,rhs_B,o%n,info_getrs)
#elif GPU_SUITE_IMPL == 1
        call check(rocsolver_dgetrf(handle,o%n,o%n,d_A,o%n,d_ipiv,d_info),'rocsolver_dgetrf')
        call check(hipDeviceSynchronize(),'getrf completion')
        call check(hipMemcpy(c_loc(info_getrf),d_info,c_sizeof(info_getrf),hipMemcpyDeviceToHost),'getrf info')
        if (info_getrf == 0) then
            call check(rocsolver_dgetrs(handle,rocblas_operation_none,o%n,o%nrhs,d_A,o%n,d_ipiv,d_B,o%n), &
                       'rocsolver_dgetrs')
            call check(hipDeviceSynchronize(),'getrs completion')
            info_getrs=0
        end if
#else
        !$omp target data use_device_addr(mat_A,ipiv,info_getrf)
        call check(rocsolver_dgetrf(handle,o%n,o%n,c_loc(mat_A),o%n,c_loc(ipiv),c_loc(info_getrf)),'rocsolver_dgetrf')
        !$omp end target data
        call check(hipDeviceSynchronize(),'getrf completion')
        !$omp target update from(info_getrf)
        if (info_getrf == 0) then
            !$omp target data use_device_addr(mat_A,rhs_B,ipiv)
            call check(rocsolver_dgetrs(handle,rocblas_operation_none,o%n,o%nrhs,c_loc(mat_A),o%n, &
                                        c_loc(ipiv),c_loc(rhs_B),o%n),'rocsolver_dgetrs')
            !$omp end target data
            call check(hipDeviceSynchronize(),'getrs completion')
            info_getrs=0
        end if
#endif
        ! getrs has no device info output. Synthetic zero requires API success AND
        ! completed work, never merely submitting the solve or skipping it.
    end subroutine
    subroutine synchronize()
        ! Restores use blocking HIP copies / OpenMP updates. operation() already
        ! waits for getrf and getrs so info is valid; avoid a third timed barrier.
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(c_loc(rhs_B),d_B,size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyDeviceToHost),'retrieve solution')
#elif GPU_SUITE_IMPL == 2
        ! Retrieval is timed; explicit unmapping remains in cleanup outside timing.
        !$omp target update from(rhs_B)
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        call check(hipDeviceSynchronize(),'cleanup completion')
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_info),'free info'); call check(hipFree(d_ipiv),'free pivots')
        call check(hipFree(d_B),'free B'); call check(hipFree(d_A),'free A')
        d_A=c_null_ptr; d_B=c_null_ptr; d_ipiv=c_null_ptr; d_info=c_null_ptr
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$omp target exit data map(delete: mat_A,rhs_B,ipiv,info_getrf)
            data_present=.false.
        end if
#endif
#if GPU_SUITE_IMPL != 0
        call check(rocblas_set_workspace(handle,c_null_ptr,0_c_size_t),'detach workspace')
        call check(rocblas_destroy_handle(handle),'destroy handle')
        call check(hipFree(workspace),'free workspace')
        handle=c_null_ptr; workspace=c_null_ptr
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
            ! Original A=n*I+ones and B=2*n: no second n*n matrix is needed.
            residual=maxval(abs(real(o%n,c_double)*rhs_B(:,rhs)+sum(rhs_B(:,rhs))-2.0_c_double*o%n))
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
