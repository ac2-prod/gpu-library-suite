include 'mkl_spblas.f90'

program sparse_cpu
    use mkl_spblas
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    interface
        subroutine make_poisson2d_csr( &
            nx, ny, nrow, ncol, nnz, &
            row_offsets, col_indices, values)

            implicit none

            integer, intent(in) :: nx, ny
            integer, intent(out) :: nrow, ncol, nnz

            integer, allocatable, intent(out) :: &
                row_offsets(:), col_indices(:)
            real(8), allocatable, intent(out) :: values(:)
        end subroutine make_poisson2d_csr
    end interface

    integer, parameter :: nx = 1024
    integer, parameter :: ny = 1024
    real(8), parameter :: alpha = 1.0d0
    real(8), parameter :: beta  = 1.0d0
    integer :: nrow, ncol, nnz
    integer :: istat
    integer, allocatable :: row_offsets(:), col_indices(:)
    real(8), allocatable :: values(:), vec_x(:), vec_y(:)
    type(sparse_matrix_t) :: mat_A
    type(matrix_descr) :: descr

    ! 1始まりのCSR配列を生成する
    call make_poisson2d_csr( &
        nx, ny, nrow, ncol, nnz, &
        row_offsets, col_indices, values)

    allocate(vec_x(ncol), vec_y(nrow), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    vec_x = 1.0d0; vec_y = 1.0d0

    descr%type = SPARSE_MATRIX_TYPE_GENERAL
    descr%mode = SPARSE_FILL_MODE_FULL
    descr%diag = SPARSE_DIAG_NON_UNIT

    istat = mkl_sparse_d_create_csr( &
        mat_A, SPARSE_INDEX_BASE_ONE, &
        nrow, ncol, &
        row_offsets(1:nrow), &
        row_offsets(2:nrow + 1), &
        col_indices, values)
    if (istat /= 0) error stop 'mkl_sparse_d_create_csr failed'

    istat = mkl_sparse_d_mv( &
        SPARSE_OPERATION_NON_TRANSPOSE, &
        alpha, mat_A, descr, &
        vec_x, beta, vec_y)
    if (istat /= 0) error stop 'mkl_sparse_d_mv failed'

    block
        integer :: ix, iy, p, neighbors
        real(8) :: expected
        if (.not. all(ieee_is_finite(vec_y))) error stop 'nonfinite SpMV result'
        do iy = 1, ny
            do ix = 1, nx
                p = (iy-1)*nx+ix
                neighbors = 0
                if (ix > 1) neighbors = neighbors+1
                if (ix < nx) neighbors = neighbors+1
                if (iy > 1) neighbors = neighbors+1
                if (iy < ny) neighbors = neighbors+1
                expected = alpha*real(4-neighbors,8)+beta
                if (abs(vec_y(p)-expected) > 1.0d-10) error stop 'SpMV verification failed'
            end do
        end do
    end block
    print *, 'verification PASS'
    istat = mkl_sparse_destroy(mat_A)
    if (istat /= 0) error stop 'mkl_sparse_destroy failed'

    deallocate(row_offsets, col_indices, values)
    deallocate(vec_x, vec_y)
end program sparse_cpu
