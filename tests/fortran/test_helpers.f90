program test_helpers
    implicit none
    interface
        subroutine make_poisson2d_csr(nx,ny,nrow,ncol,nnz,row_offsets,col_indices,values)
            integer,intent(in)::nx,ny
            integer,intent(out)::nrow,ncol,nnz
            integer,allocatable,intent(out)::row_offsets(:),col_indices(:)
            real(8),allocatable,intent(out)::values(:)
        end subroutine
        subroutine make_dense_system(n,nrhs,a,b)
            integer,intent(in)::n,nrhs
            real(8),intent(out)::a(n,n),b(n,nrhs)
        end subroutine
    end interface
    integer,allocatable :: rows(:),cols(:)
    real(8),allocatable :: values(:)
    real(8) :: a(3,3),b(3,2)
    integer :: nrow,ncol,nnz,i,j,nx,ny,p,ix,iy,neighbors
    do nx=1,4
        do ny=1,3
            call make_poisson2d_csr(nx,ny,nrow,ncol,nnz,rows,cols,values)
            if (nrow/=nx*ny.or.ncol/=nrow.or.nnz/=5*nx*ny-2*nx-2*ny) error stop 'dimensions'
            if (rows(1)/=1.or.rows(nrow+1)/=nnz+1) error stop 'one-based offsets'
            do iy=1,ny
                do ix=1,nx
                    p=(iy-1)*nx+ix; neighbors=0
                    if(ix>1)neighbors=neighbors+1
                    if(ix<nx)neighbors=neighbors+1
                    if(iy>1)neighbors=neighbors+1
                    if(iy<ny)neighbors=neighbors+1
                    if(sum(values(rows(p):rows(p+1)-1))/=real(4-neighbors,8)) error stop 'stencil'
                    do j=rows(p),rows(p+1)-1
                        if(cols(j)<1.or.cols(j)>ncol)error stop 'column bounds'
                        if(j>rows(p))then
                            if(cols(j)<=cols(j-1))error stop 'column order'
                        end if
                        if(cols(j)==p.and.values(j)/=4.0d0)error stop 'diagonal'
                        if(cols(j)/=p.and.values(j)/=-1.0d0)error stop 'neighbor'
                    end do
                end do
            end do
        end do
    end do
    call make_dense_system(3,2,a,b)
    do j=1,3
        do i=1,3
            if (i==j.and.a(i,j)/=4.0d0)error stop 'dense diagonal'
            if (i/=j.and.a(i,j)/=1.0d0)error stop 'dense off-diagonal'
        end do
    end do
    if(any(b/=6.0d0))error stop 'canonical RHS'
    if(any(matmul(a,[1.0d0,1.0d0,1.0d0])/=b(:,1)))error stop 'known solution'
    print *, 'Fortran helper checks PASS'
end program test_helpers
