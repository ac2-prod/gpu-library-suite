program solver_cpu
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
    integer :: info
    integer, allocatable :: ipiv(:)
    real(8), allocatable :: mat_A(:,:), rhs_B(:,:)

    allocate(mat_A(n,n), rhs_B(n,nrhs), ipiv(n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    ! 求解対象の密行列Aと右辺Bを生成する
    call make_dense_system(n, nrhs, mat_A, rhs_B)

    call dgesv(n, nrhs, mat_A, lda, ipiv, rhs_B, ldb, info)
    if (info /= 0) error stop 'dgesv failed'

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
    deallocate(mat_A, rhs_B, ipiv)
end program solver_cpu
