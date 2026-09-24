! Canonical five-point Poisson matrix, shifted to one-based Fortran CSR.
subroutine make_poisson2d_csr(nx, ny, nrow, ncol, nnz, row_offsets, col_indices, values)
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    integer, intent(in) :: nx, ny
    integer, intent(out) :: nrow, ncol, nnz
    integer, allocatable, intent(out) :: row_offsets(:), col_indices(:)
    real(8), allocatable, intent(out) :: values(:)
    integer(int64) :: rows64, nnz64
    integer :: ix, iy, p, q, allocation_status
    if (nx < 1 .or. ny < 1) error stop 'invalid Poisson dimensions'
    rows64 = int(nx,int64)*int(ny,int64)
    if (rows64 >= int(huge(nrow),int64)) error stop 'Poisson row count overflows LP64'
    nnz64 = 5_int64*rows64-2_int64*nx-2_int64*ny
    if (nnz64 >= int(huge(nnz),int64)) error stop 'Poisson nnz overflows LP64'
    nrow = int(rows64); ncol = nrow; nnz = int(nnz64)
    allocate(row_offsets(nrow+1), col_indices(nnz), values(nnz), stat=allocation_status)
    if (allocation_status /= 0) error stop 'Poisson allocation failed'
    q = 1
    do iy = 1, ny
        do ix = 1, nx
            p = (iy-1)*nx+ix
            row_offsets(p) = q
            ! Ascending column order: up, left, diagonal, right, down.
            if (iy > 1) call append(p-nx, -1.0d0)
            if (ix > 1) call append(p-1, -1.0d0)
            call append(p, 4.0d0)
            if (ix < nx) call append(p+1, -1.0d0)
            if (iy < ny) call append(p+nx, -1.0d0)
        end do
    end do
    row_offsets(nrow+1) = q
    if (q /= nnz+1) error stop 'Poisson nnz mismatch'
contains
    subroutine append(column, value)
        integer, intent(in) :: column
        real(8), intent(in) :: value
        col_indices(q) = column
        values(q) = value
        q = q+1
    end subroutine append
end subroutine make_poisson2d_csr
