program openacc_cusparse
    use cudafor
    use cusparse
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    interface
        subroutine make_poisson2d_csr(nx, ny, nrow, ncol, nnz, &
            row_offsets, col_indices, values)
            implicit none
            integer, intent(in) :: nx, ny
            integer, intent(out) :: nrow, ncol, nnz
            integer, allocatable, intent(out) ::  &
                row_offsets(:), col_indices(:)
            real(8), allocatable, intent(out) :: values(:)
        end subroutine make_poisson2d_csr
    end interface

    integer, parameter :: nx = 1024, ny = 1024
    real(8), parameter :: alpha = 1.0d0, beta = 1.0d0
    integer :: nrow, ncol, nnz, istat
    integer(8) :: nrow64, ncol64, nnz64, buffer_size
    integer, allocatable :: h_row_offsets(:), h_col_indices(:)
    real(8), allocatable :: h_values(:), h_x(:), h_y(:)
    integer(1), device, allocatable :: d_buffer(:)
    type(cusparseHandle) :: handle
    type(cusparseSpMatDescr) :: mat_A
    type(cusparseDnVecDescr) :: vec_x, vec_y

    call make_poisson2d_csr(nx, ny, nrow, ncol, nnz, &
        h_row_offsets, h_col_indices, h_values)

    nrow64 = nrow; ncol64 = ncol; nnz64 = nnz
    allocate(h_x(ncol), h_y(nrow), stat=allocation_status)
    if (allocation_status /= 0) error stop 'allocation failed'
    h_x = 1.0d0; h_y = 1.0d0
    istat = cusparseCreate(handle)
    if (istat /= 0) error stop 'cusparseCreate failed'



    !$acc data copyin(h_row_offsets(1:nrow+1), h_col_indices(1:nnz), &
    !$acc             h_values(1:nnz), h_x(1:ncol)) copy(h_y(1:nrow))
        !$acc host_data use_device(h_row_offsets, h_col_indices, &
        !$acc                          h_values, h_x, h_y)
            istat = cusparseCreateCsr(mat_A, nrow64, ncol64, nnz64, &
                h_row_offsets, h_col_indices, h_values, &
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, &
                CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F)
    if (istat /= 0) error stop 'cusparseCreateCsr failed'
            istat = cusparseCreateDnVec(vec_x, ncol64, h_x, CUDA_R_64F)
    if (istat /= 0) error stop 'cusparseCreateDnVec failed'
            istat = cusparseCreateDnVec(vec_y, nrow64, h_y, CUDA_R_64F)
    if (istat /= 0) error stop 'cusparseCreateDnVec failed'
        !$acc end host_data

        buffer_size = 0_8
        istat = cusparseSpMV_bufferSize(handle, &
            CUSPARSE_OPERATION_NON_TRANSPOSE, &
            alpha, mat_A, vec_x, beta, vec_y, CUDA_R_64F, &
            CUSPARSE_SPMV_ALG_DEFAULT, buffer_size)
    if (istat /= 0) error stop 'cusparseSpMV_bufferSize failed'

        allocate(d_buffer(max(1_8, buffer_size)), stat=allocation_status)

        if (allocation_status /= 0) error stop 'allocation failed'
        istat = cusparseSpMV(handle, &
            CUSPARSE_OPERATION_NON_TRANSPOSE, &
            alpha, mat_A, vec_x, beta, vec_y, CUDA_R_64F, &
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer)
    if (istat /= 0) error stop 'cusparseSpMV failed'

        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'

        istat = cusparseDestroySpMat(mat_A)
    if (istat /= 0) error stop 'cusparseDestroySpMat failed'
        istat = cusparseDestroyDnVec(vec_x)
    if (istat /= 0) error stop 'cusparseDestroyDnVec failed'
        istat = cusparseDestroyDnVec(vec_y)
    if (istat /= 0) error stop 'cusparseDestroyDnVec failed'
        deallocate(d_buffer)
    !$acc end data

    print *, "h_y(1) = ", h_y(1)

    block
        integer :: ix, iy, p, neighbors
        real(8) :: expected
        if (.not. all(ieee_is_finite(h_y))) error stop 'nonfinite SpMV result'
        do iy = 1, ny
            do ix = 1, nx
                p = (iy-1)*nx+ix
                neighbors = 0
                if (ix > 1) neighbors = neighbors+1
                if (ix < nx) neighbors = neighbors+1
                if (iy > 1) neighbors = neighbors+1
                if (iy < ny) neighbors = neighbors+1
                expected = alpha*real(4-neighbors,8)+beta
                if (abs(h_y(p)-expected) > 1.0d-10) error stop 'SpMV verification failed'
            end do
        end do
    end block
    print *, 'verification PASS'
    istat = cusparseDestroy(handle)
    if (istat /= 0) error stop 'cusparseDestroy failed'
    deallocate(h_row_offsets, h_col_indices, h_values, h_x, h_y)
end program openacc_cusparse
