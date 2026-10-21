program solver_gpu
    use iso_c_binding
    use ieee_arithmetic, only: ieee_is_finite
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
    integer(c_int), target :: info=-1
    type(c_ptr) :: d_A=c_null_ptr,d_B=c_null_ptr,d_ipiv=c_null_ptr,d_info=c_null_ptr,handle=c_null_ptr
    integer :: allocation_status,j
    real(c_double) :: error,residual
    logical :: success=.true.

    allocate(mat_A(n,n),rhs_B(n,nrhs),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    call make_dense_system(n,nrhs,mat_A,rhs_B)
    pipeline: block
        if (.not. ok(hipSetDevice(0_c_int),'hipSetDevice')) exit pipeline
        if (.not. ok(hipMalloc(d_A,size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate A')) exit pipeline
        if (.not. ok(hipMalloc(d_B,size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double)),'allocate B')) exit pipeline
        if (.not. ok(hipMalloc(d_ipiv,int(n,c_size_t)*c_sizeof(info)),'allocate pivots')) exit pipeline
        if (.not. ok(hipMalloc(d_info,c_sizeof(info)),'allocate info')) exit pipeline
        if (.not. ok(hipMemcpy(d_A,c_loc(mat_A),size(mat_A,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'copy A')) exit pipeline
        if (.not. ok(hipMemcpy(d_B,c_loc(rhs_B),size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyHostToDevice),'copy B')) exit pipeline
        if (.not. ok(hipMemcpy(d_info,c_loc(info),c_sizeof(info),hipMemcpyHostToDevice),'initialize info')) exit pipeline
        if (.not. ok(rocblas_create_handle(handle),'rocblas_create_handle')) exit pipeline
        if (.not. ok(rocsolver_dgetrf(handle,n,n,d_A,n,d_ipiv,d_info),'rocsolver_dgetrf')) exit pipeline
        if (.not. ok(hipDeviceSynchronize(),'getrf completion')) exit pipeline
        if (.not. ok(hipMemcpy(c_loc(info),d_info,c_sizeof(info),hipMemcpyDeviceToHost),'getrf info')) exit pipeline
        if (info /= 0) then
            print *, 'LU factorization failed; info = ',info
            success=.false.
            exit pipeline
        end if
        ! rocSOLVER getrs returns rocblas_status; it has no LAPACK info argument.
        if (.not. ok(rocsolver_dgetrs(handle,rocblas_operation_none,n,nrhs,d_A,n,d_ipiv,d_B,n), &
                     'rocsolver_dgetrs')) exit pipeline
        if (.not. ok(hipDeviceSynchronize(),'getrs completion')) exit pipeline
        if (.not. ok(hipMemcpy(c_loc(rhs_B),d_B,size(rhs_B,kind=c_size_t)*c_sizeof(0.0_c_double), &
                             hipMemcpyDeviceToHost),'copy solution')) exit pipeline
        success=all(ieee_is_finite(rhs_B))
        if (.not. success) exit pipeline
        error=maxval(abs(rhs_B-1.0_c_double)); residual=0
        do j=1,nrhs
            residual=max(residual,maxval(abs(n*rhs_B(:,j)+sum(rhs_B(:,j))-2.0_c_double*n))/(4.0_c_double*n))
        end do
        print *, 'rocSOLVER getrf info=',info,', max solution error=',error,', relative residual=',residual
        success=ieee_is_finite(residual) .and. error <= 1.0e-12_c_double .and. residual <= 1.0e-12_c_double
    end block pipeline
    if (c_associated(d_A)) call cleanup_check(hipDeviceSynchronize(),'cleanup completion')
    if (c_associated(handle)) call cleanup_check(rocblas_destroy_handle(handle),'destroy handle')
    if (c_associated(d_info)) call cleanup_check(hipFree(d_info),'free info')
    if (c_associated(d_ipiv)) call cleanup_check(hipFree(d_ipiv),'free pivots')
    if (c_associated(d_B)) call cleanup_check(hipFree(d_B),'free B')
    if (c_associated(d_A)) call cleanup_check(hipFree(d_A),'free A')
    deallocate(mat_A,rhs_B)
    if (.not. success) error stop 'rocSOLVER pipeline or verification failed'
contains
    logical function ok(status,label)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: label
        ok=status == 0
        if (.not. ok) then
            print *, trim(label),': status ',status
            success=.false.
        end if
    end function
    subroutine cleanup_check(status,label)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: label
        logical :: ignored
        ignored=ok(status,label)
    end subroutine
end program solver_gpu
