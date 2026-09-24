program blas_gpu
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
    real(8), allocatable :: h_A(:,:), h_B(:,:), h_C(:,:)
    real(8), device, allocatable :: d_A(:,:), d_B(:,:), d_C(:,:)
    alpha = 1.0d0; beta  = 1.0d0

    allocate(h_A(m,k), h_B(k,n), h_C(m,n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    h_A = 1.0d0; h_B = 1.0d0; h_C = 1.0d0

    allocate(d_A(m,k), d_B(k,n), d_C(m,n), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    istat = cudaMemcpy(d_A, h_A, m*k, cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    istat = cudaMemcpy(d_B, h_B, k*n, cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    istat = cudaMemcpy(d_C, h_C, m*n, cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    istat = cublasCreate(handle)
    if (istat /= 0) error stop 'cublasCreate failed'

    istat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &
        alpha, d_A, m, d_B, k, beta, d_C, m)
    if (istat /= 0) error stop 'cublasDgemm failed'

    istat = cudaMemcpy(h_C, d_C, m*n, cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'
    if (.not. all(ieee_is_finite(h_C))) error stop 'nonfinite DGEMM result'
    if (maxval(abs(h_C - (alpha*real(k,8)+beta))) > 1.0d-10*real(k,8)) &
        error stop 'DGEMM verification failed'
    print *, 'verification PASS'
    istat = cublasDestroy(handle)
    if (istat /= 0) error stop 'cublasDestroy failed'
    deallocate(d_A, d_B, d_C)
    deallocate(h_A, h_B, h_C)
end program blas_gpu
