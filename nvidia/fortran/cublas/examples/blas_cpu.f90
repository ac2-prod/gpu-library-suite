program blas_cpu
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer, parameter :: m = 1024
    integer, parameter :: n = 1024
    integer, parameter :: k = 1024
    real(8) :: alpha, beta
    real(8), allocatable :: &
        mat_A(:,:), mat_B(:,:), mat_C(:,:)

    alpha = 1.0d0
    beta  = 1.0d0

    allocate(mat_A(m,k), mat_B(k,n), mat_C(m,n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    mat_A = 1.0d0
    mat_B = 1.0d0
    mat_C = 1.0d0

    call dgemm('N', 'N', m, n, k, &
               alpha, mat_A, m, &
                      mat_B, k, &
               beta,  mat_C, m)

    if (.not. all(ieee_is_finite(mat_C))) error stop 'nonfinite DGEMM result'
    if (maxval(abs(mat_C - (alpha*real(k,8)+beta))) > 1.0d-10*real(k,8)) &
        error stop 'DGEMM verification failed'
    print *, 'verification PASS'
    deallocate(mat_A, mat_B, mat_C)
end program blas_cpu
