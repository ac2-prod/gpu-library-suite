program solver_cpu
    use iso_c_binding, only: c_double
    use ieee_arithmetic, only: ieee_is_finite
    implicit none
    interface
        subroutine make_dense_system(n,nrhs,mat_A,rhs_B)
            import c_double
            integer, intent(in) :: n,nrhs
            real(c_double), intent(out) :: mat_A(n,n),rhs_B(n,nrhs)
        end subroutine
    end interface
    integer, parameter :: n=1024,nrhs=16
    real(c_double), allocatable :: mat_A(:,:),rhs_B(:,:)
    integer, allocatable :: ipiv(:)
    integer :: info,status,j
    real(c_double) :: error,residual

    allocate(mat_A(n,n),rhs_B(n,nrhs),ipiv(n),stat=status)
    if (status /= 0) error stop 'host allocation failed'
    call make_dense_system(n,nrhs,mat_A,rhs_B)
    ! Link common/fortran/make_dense_system.f90 and an explicit LP64 LAPACK provider.
    call dgesv(n,nrhs,mat_A,n,ipiv,rhs_B,n,info)
    if (info /= 0) error stop 'DGESV failed'
    if (.not. all(ieee_is_finite(rhs_B))) error stop 'nonfinite solution'
    error=maxval(abs(rhs_B-1.0_c_double))
    residual=0
    do j=1,nrhs
        residual=max(residual,maxval(abs(n*rhs_B(:,j)+sum(rhs_B(:,j))-2.0_c_double*n))/(4.0_c_double*n))
    end do
    print *, 'CPU DGESV info=',info,', max solution error=',error,', relative residual=',residual
    deallocate(mat_A,rhs_B,ipiv)
    if (.not. ieee_is_finite(residual)) error stop 'nonfinite residual'
    if (error > 1.0e-12_c_double .or. residual > 1.0e-12_c_double) error stop 'solution verification failed'
end program solver_cpu
