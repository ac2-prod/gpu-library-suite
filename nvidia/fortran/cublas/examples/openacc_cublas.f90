program openacc_cublas
    use cudafor
    use cublas_v2
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer, parameter :: m = 1024
    integer, parameter :: n = 1024
    integer, parameter :: k = 1024
    real(8) :: alpha, beta
    integer :: istat
    type(cublasHandle) :: handle
    real(8), allocatable :: mat_A(:,:), mat_B(:,:), mat_C(:,:)
    alpha = 1.0d0; beta  = 1.0d0

    allocate(mat_A(m,k), mat_B(k,n), mat_C(m,n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    mat_A = 1.0d0; mat_B = 1.0d0; mat_C = 1.0d0

    istat = cublasCreate(handle)
    if (istat /= 0) error stop 'cublasCreate failed'

    !$acc data copyin(mat_A(1:m,1:k), mat_B(1:k,1:n)) &
    !$acc     copy(mat_C(1:m,1:n))
        !$acc host_data use_device(mat_A, mat_B, mat_C)
            istat = cublasDgemm(handle, &
                CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &
                alpha, mat_A, m, mat_B, k, &
                beta,  mat_C, m)
    if (istat /= 0) error stop 'cublasDgemm failed'
        !$acc end host_data
        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'
    !$acc end data
    print *, "mat_C(1,1) = ", mat_C(1,1)
    if (.not. all(ieee_is_finite(mat_C))) error stop 'nonfinite DGEMM result'
    if (maxval(abs(mat_C - (alpha*real(k,8)+beta))) > 1.0d-10*real(k,8)) &
        error stop 'DGEMM verification failed'
    print *, 'verification PASS'
    istat = cublasDestroy(handle)
    if (istat /= 0) error stop 'cublasDestroy failed'
    deallocate(mat_A, mat_B, mat_C)
end program openacc_cublas
