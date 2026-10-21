program blas_cpu
    use iso_c_binding, only: c_double
    use ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer, parameter :: m=1024, n=1024, k=1024
    real(c_double), parameter :: alpha=1.0_c_double, beta=1.0_c_double
    real(c_double), allocatable :: mat_A(:,:),mat_B(:,:),mat_C(:,:)
    real(c_double) :: error
    integer :: status

    allocate(mat_A(m,k),mat_B(k,n),mat_C(m,n),stat=status)
    if (status /= 0) error stop 'host allocation failed'
    mat_A=1; mat_B=1; mat_C=1
    ! Link an explicitly selected production BLAS provider (LP64).
    call dgemm('N','N',m,n,k,alpha,mat_A,m,mat_B,k,beta,mat_C,m)
    if (.not. all(ieee_is_finite(mat_C))) error stop 'nonfinite DGEMM result'
    error=maxval(abs(mat_C-real(k+1,c_double)))
    print *, 'CPU DGEMM complete; max error = ',error
    deallocate(mat_A,mat_B,mat_C)
    if (error > 1.0e-10_c_double) error stop 'DGEMM verification failed'
end program blas_cpu
