program openmp_rocsolver
    use iso_c_binding
    use ieee_arithmetic, only: ieee_is_finite
    use omp_lib
    use hipfort, only: hipSetDevice,hipMalloc,hipFree,hipMemcpy,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use hipfort_rocblas, only: rocblas_create_handle,rocblas_destroy_handle,rocblas_operation_none
    use hipfort_rocsolver, only: rocsolver_dgetrs
    implicit none
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
    interface
        subroutine make_dense_system(n,nrhs,mat_A,rhs_B)
            import c_int,c_double
            integer(c_int), intent(in) :: n,nrhs
            real(c_double), intent(out) :: mat_A(n,n),rhs_B(n,nrhs)
        end subroutine
    end interface
    integer(c_int), parameter :: n=1024,nrhs=16
    real(c_double), allocatable, target :: mat_A(:,:),rhs_B(:,:)
    integer(c_int), allocatable, target :: ipiv(:)
    integer(c_int), target :: info=-1
    type(c_ptr) :: handle=c_null_ptr
    integer :: device,allocation_status,j
    logical :: initial,success=.true.
    real(c_double) :: error,residual

    ! Regions below use this unchanged OpenMP default device.
    device=omp_get_default_device()
    if (device < 0 .or. device >= omp_get_num_devices()) error stop 'OpenMP target unavailable'
    initial=.true.
    !$omp target map(from: initial)
    initial=omp_is_initial_device()
    !$omp end target
    if (initial) error stop 'OpenMP host fallback rejected'
    call record(hipSetDevice(int(device,c_int)),'hipSetDevice')
    if (.not. success) error stop 'HIP device selection failed'
    allocate(mat_A(n,n),rhs_B(n,nrhs),ipiv(n),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    call make_dense_system(n,nrhs,mat_A,rhs_B)
    ipiv=0
    call record(rocblas_create_handle(handle),'rocblas_create_handle')
    if (success) then
        !$omp target data map(to: mat_A,ipiv) map(tofrom: rhs_B,info)
        !$omp target data use_device_addr(mat_A,ipiv,info)
        call record(rocsolver_dgetrf(handle,n,n,c_loc(mat_A),n,c_loc(ipiv),c_loc(info)),'rocsolver_dgetrf')
        !$omp end target data
        call record(hipDeviceSynchronize(),'getrf completion')
        !$omp target update from(info)
        if (success .and. info == 0) then
            !$omp target data use_device_addr(mat_A,rhs_B,ipiv)
            call record(rocsolver_dgetrs(handle,rocblas_operation_none,n,nrhs,c_loc(mat_A),n, &
                                        c_loc(ipiv),c_loc(rhs_B),n),'rocsolver_dgetrs')
            !$omp end target data
        else
            success=.false.
        end if
        ! Complete library work before OpenMP copies back and releases its mapping.
        call record(hipDeviceSynchronize(),'getrs/cleanup completion')
        !$omp end target data
    end if
    if (success) then
        success=all(ieee_is_finite(rhs_B))
        if (success) then
            error=maxval(abs(rhs_B-1.0_c_double)); residual=0
            do j=1,nrhs
                residual=max(residual,maxval(abs(n*rhs_B(:,j)+sum(rhs_B(:,j))-2.0_c_double*n))/(4.0_c_double*n))
            end do
            print *, 'OpenMP rocSOLVER getrf info=',info,', max solution error=',error,', relative residual=',residual
            success=ieee_is_finite(residual) .and. error <= 1.0e-12_c_double .and. residual <= 1.0e-12_c_double
        end if
    end if
    if (c_associated(handle)) call record(rocblas_destroy_handle(handle),'destroy handle')
    deallocate(mat_A,rhs_B,ipiv)
    if (.not. success) error stop 'OpenMP rocSOLVER pipeline or verification failed'
contains
    subroutine record(status,label)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: label
        if (status /= 0) then
            print *, trim(label),': status ',status
            success=.false.
        end if
    end subroutine
end program openmp_rocsolver
