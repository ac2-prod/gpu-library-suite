! Explicit helper input: sorted zero-based CSR for the five-point Poisson matrix.
module amd_poisson_csr
    use iso_c_binding
    implicit none
    private
    public :: make_poisson2d_csr
contains
    subroutine make_poisson2d_csr(nx,ny,nrow,ncol,nnz,row_offsets,col_indices,values)
        integer, intent(in) :: nx,ny
        integer(c_int), intent(out) :: nrow,ncol,nnz
        integer(c_int), allocatable, intent(out) :: row_offsets(:),col_indices(:)
        real(c_double), allocatable, intent(out) :: values(:)
        integer(c_int64_t) :: rows,nonzeros
        integer :: ix,iy,p,j,status
        if (nx<1 .or. ny<1) error stop 'Poisson dimensions must be positive'
        rows=int(nx,c_int64_t)*int(ny,c_int64_t)
        if (rows>=huge(0_c_int)) error stop 'Poisson CSR exceeds LP64'
        nonzeros=5_c_int64_t*rows-2_c_int64_t*nx-2_c_int64_t*ny
        if (nonzeros>huge(0_c_int)) error stop 'Poisson CSR exceeds LP64'
        nrow=int(rows,c_int); ncol=nrow; nnz=int(nonzeros,c_int)
        allocate(row_offsets(nrow+1),col_indices(nnz),values(nnz),stat=status)
        if (status/=0) error stop 'Poisson CSR allocation failed'
        j=0
        do iy=0,ny-1
            do ix=0,nx-1
                p=iy*nx+ix
                row_offsets(p+1)=j
                if (iy>0) then
                    j=j+1; col_indices(j)=p-nx; values(j)=-1
                end if
                if (ix>0) then
                    j=j+1; col_indices(j)=p-1; values(j)=-1
                end if
                j=j+1; col_indices(j)=p; values(j)=4
                if (ix+1<nx) then
                    j=j+1; col_indices(j)=p+1; values(j)=-1
                end if
                if (iy+1<ny) then
                    j=j+1; col_indices(j)=p+nx; values(j)=-1
                end if
            end do
        end do
        row_offsets(nrow+1)=j
        if (j/=nnz) error stop 'Poisson CSR nonzero count mismatch'
    end subroutine
end module amd_poisson_csr
