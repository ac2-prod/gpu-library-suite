program openacc_cusolver
    use cudafor
    use cusolverDn
    use cublas_v2, only: CUBLAS_OP_N
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    interface
        subroutine make_dense_system(n, nrhs, mat_A, rhs_B)
            implicit none
            integer, intent(in) :: n, nrhs
            real(8), intent(out) :: mat_A(n,n), rhs_B(n,nrhs)
        end subroutine make_dense_system
    end interface

    integer, parameter :: n = 1024, nrhs = 16, lda = n, ldb = n
    integer :: istat, lwork, info_getrf, info_getrs
    type(cusolverDnHandle) :: handle
    real(8), allocatable :: mat_A(:,:), rhs_B(:,:)
    integer, allocatable :: ipiv(:)
    real(8), device, allocatable :: d_work(:)

    allocate(mat_A(n,n), rhs_B(n,nrhs), ipiv(n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    call make_dense_system(n, nrhs, mat_A, rhs_B)
    istat = cusolverDnCreate(handle)
    if (istat /= 0) error stop 'cusolverDnCreate failed'

    !$acc data copyin(mat_A(1:n,1:n)) copy(rhs_B(1:n,1:nrhs)) &
    !$acc     create(ipiv(1:n)) copyout(info_getrf, info_getrs)
        !$acc host_data use_device(mat_A)
            istat = cusolverDnDgetrf_bufferSize( &
                handle, n, n, mat_A, lda, lwork)
    if (istat /= 0) error stop 'cusolverDnDgetrf_bufferSize failed'
        !$acc end host_data
        if (lwork < 0) error stop 'negative workspace size'
        allocate(d_work(max(1,lwork)), stat=allocation_status)
        if (allocation_status /= 0) error stop 'allocation failed'
        !$acc host_data use_device(mat_A, rhs_B, ipiv, &
        !$acc                      info_getrf, info_getrs)
            istat = cusolverDnDgetrf( &
                handle, n, n, mat_A, lda, d_work, ipiv, info_getrf)
    if (istat /= 0) error stop 'cusolverDnDgetrf failed'
        !$acc end host_data
        istat = cudaDeviceSynchronize()
        if (istat /= 0) error stop 'getrf synchronization failed'
        !$acc update self(info_getrf)
        if (info_getrf /= 0) error stop 'getrf failed before solve'
        !$acc host_data use_device(mat_A, rhs_B, ipiv, info_getrs)
            istat = cusolverDnDgetrs( &
                handle, CUBLAS_OP_N, n, nrhs, &
                mat_A, lda, ipiv, rhs_B, ldb, info_getrs)
    if (istat /= 0) error stop 'cusolverDnDgetrs failed'
        !$acc end host_data

        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'
        deallocate(d_work)
    !$acc end data

    print *, "info_getrf = ", info_getrf, ", info_getrs = ", info_getrs
    print *, "rhs_B(1,1) = ", rhs_B(1,1)

    if (info_getrf /= 0 .or. info_getrs /= 0) error stop 'getrf/getrs failed'
    if (.not. all(ieee_is_finite(rhs_B))) error stop 'nonfinite solution'
    if (maxval(abs(rhs_B-1.0d0)) > 1.0d-10) error stop 'solution verification failed'
    block
        integer :: rhs
        real(8) :: residual, scale
        do rhs = 1, nrhs
            residual = maxval(abs(real(n,8)*rhs_B(:,rhs)+sum(rhs_B(:,rhs))-2.0d0*real(n,8)))
            scale = 2.0d0*real(n,8)*(maxval(abs(rhs_B(:,rhs)))+1.0d0)
            if (residual/scale > 1.0d-10) error stop 'residual verification failed'
        end do
    end block
    print *, 'verification PASS'
    istat = cusolverDnDestroy(handle)
    if (istat /= 0) error stop 'cusolverDnDestroy failed'
    deallocate(mat_A, rhs_B, ipiv)
end program openacc_cusolver
