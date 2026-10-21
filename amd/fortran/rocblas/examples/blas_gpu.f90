program blas_gpu
    use iso_c_binding
    use ieee_arithmetic, only: ieee_is_finite
    use hipfort, only: hipSetDevice,hipMalloc,hipFree,hipMemcpy,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use hipfort_rocblas, only: rocblas_create_handle,rocblas_destroy_handle,rocblas_operation_none, &
                              rocblas_dgemm
    implicit none
    integer(c_int), parameter :: m=1024,n=1024,k=1024
    real(c_double), target :: alpha=1.0_c_double,beta=1.0_c_double
    real(c_double), allocatable, target :: mat_A(:,:),mat_B(:,:),mat_C(:,:)
    type(c_ptr) :: d_A=c_null_ptr,d_B=c_null_ptr,d_C=c_null_ptr,handle=c_null_ptr
    real(c_double) :: error
    integer :: allocation_status
    logical :: success=.true.

    allocate(mat_A(m,k),mat_B(k,n),mat_C(m,n),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    mat_A=1; mat_B=1; mat_C=1
    pipeline: block
        if (.not. ok(hipSetDevice(0_c_int),'hipSetDevice')) exit pipeline
        if (.not. ok(hipMalloc(d_A,size(mat_A,kind=c_size_t)*c_sizeof(alpha)),'hipMalloc A')) exit pipeline
        if (.not. ok(hipMalloc(d_B,size(mat_B,kind=c_size_t)*c_sizeof(alpha)),'hipMalloc B')) exit pipeline
        if (.not. ok(hipMalloc(d_C,size(mat_C,kind=c_size_t)*c_sizeof(alpha)),'hipMalloc C')) exit pipeline
        if (.not. ok(hipMemcpy(d_A,c_loc(mat_A),size(mat_A,kind=c_size_t)*c_sizeof(alpha), &
                             hipMemcpyHostToDevice),'copy A')) exit pipeline
        if (.not. ok(hipMemcpy(d_B,c_loc(mat_B),size(mat_B,kind=c_size_t)*c_sizeof(alpha), &
                             hipMemcpyHostToDevice),'copy B')) exit pipeline
        if (.not. ok(hipMemcpy(d_C,c_loc(mat_C),size(mat_C,kind=c_size_t)*c_sizeof(alpha), &
                             hipMemcpyHostToDevice),'copy C')) exit pipeline
        if (.not. ok(rocblas_create_handle(handle),'rocblas_create_handle')) exit pipeline
        if (.not. ok(rocblas_dgemm(handle,rocblas_operation_none,rocblas_operation_none,m,n,k, &
                                 alpha,d_A,m,d_B,k,beta,d_C,m),'rocblas_dgemm')) exit pipeline
        if (.not. ok(hipDeviceSynchronize(),'DGEMM completion')) exit pipeline
        if (.not. ok(hipMemcpy(c_loc(mat_C),d_C,size(mat_C,kind=c_size_t)*c_sizeof(alpha), &
                             hipMemcpyDeviceToHost),'copy result C')) exit pipeline
        success=all(ieee_is_finite(mat_C))
        if (success) then
            error=maxval(abs(mat_C-real(k+1,c_double)))
            print *, 'rocBLAS DGEMM complete; max error = ',error
            success=error <= 1.0e-10_c_double
        end if
    end block pipeline
    ! Complete outstanding work before releasing resources, including failure paths.
    if (c_associated(d_A)) call cleanup_check(hipDeviceSynchronize(),'cleanup completion')
    if (c_associated(handle)) call cleanup_check(rocblas_destroy_handle(handle),'destroy handle')
    if (c_associated(d_C)) call cleanup_check(hipFree(d_C),'free C')
    if (c_associated(d_B)) call cleanup_check(hipFree(d_B),'free B')
    if (c_associated(d_A)) call cleanup_check(hipFree(d_A),'free A')
    deallocate(mat_A,mat_B,mat_C)
    if (.not. success) error stop 'rocBLAS pipeline or verification failed'
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
end program blas_gpu
