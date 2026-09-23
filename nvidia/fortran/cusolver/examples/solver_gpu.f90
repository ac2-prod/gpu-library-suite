program solver_gpu
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

    integer, parameter :: n = 1024, nrhs = 16
    integer, parameter :: lda = n, ldb = n
    integer :: istat, lwork
    integer :: info_getrf, info_getrs
    type(cusolverDnHandle) :: handle

    real(8), allocatable :: h_A(:,:), h_B(:,:)
    real(8), device, allocatable :: d_A(:,:), d_B(:,:), d_work(:)
    integer, device, allocatable :: d_ipiv(:)
    integer, device :: d_info_getrf, d_info_getrs

    allocate(h_A(n,n), h_B(n,nrhs), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    call make_dense_system(n, nrhs, h_A, h_B)

    allocate(d_A(n,n), d_B(n,nrhs), d_ipiv(n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    istat = cudaMemcpy(d_A, h_A, n*n, cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    istat = cudaMemcpy(d_B, h_B, n*nrhs, cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    istat = cusolverDnCreate(handle)
    if (istat /= 0) error stop 'cusolverDnCreate failed'

    istat = cusolverDnDgetrf_bufferSize( &
        handle, n, n, d_A, lda, lwork)
    if (istat /= 0) error stop 'cusolverDnDgetrf_bufferSize failed'

    if (lwork < 0) error stop 'negative workspace size'
    allocate(d_work(max(1,lwork)), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    istat = cusolverDnDgetrf( &
        handle, n, n, d_A, lda, d_work, d_ipiv, d_info_getrf)
    if (istat /= 0) error stop 'cusolverDnDgetrf failed'

    istat = cudaMemcpy(info_getrf, d_info_getrf, 1, cudaMemcpyDeviceToHost)
    if (istat /= 0 .or. info_getrf /= 0) error stop 'getrf failed before solve'

    istat = cusolverDnDgetrs( &
        handle, CUBLAS_OP_N, n, nrhs, &
        d_A, lda, d_ipiv, d_B, ldb, d_info_getrs)
    if (istat /= 0) error stop 'cusolverDnDgetrs failed'

    istat = cudaMemcpy(h_B, d_B, n*nrhs, cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    istat = cudaMemcpy( &
        info_getrf, d_info_getrf, 1, cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    istat = cudaMemcpy( &
        info_getrs, d_info_getrs, 1, cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    if (info_getrf /= 0 .or. info_getrs /= 0) error stop 'getrf/getrs failed'
    if (.not. all(ieee_is_finite(h_B))) error stop 'nonfinite solution'
    if (maxval(abs(h_B-1.0d0)) > 1.0d-10) error stop 'solution verification failed'
    block
        integer :: rhs
        real(8) :: residual, scale
        do rhs = 1, nrhs
            residual = maxval(abs(real(n,8)*h_B(:,rhs)+sum(h_B(:,rhs))-2.0d0*real(n,8)))
            scale = 2.0d0*real(n,8)*(maxval(abs(h_B(:,rhs)))+1.0d0)
            if (residual/scale > 1.0d-10) error stop 'residual verification failed'
        end do
    end block
    print *, 'verification PASS'
    istat = cusolverDnDestroy(handle)
    if (istat /= 0) error stop 'cusolverDnDestroy failed'

    deallocate(d_work, d_A, d_B, d_ipiv)
    deallocate(h_A, h_B)
end program solver_gpu
