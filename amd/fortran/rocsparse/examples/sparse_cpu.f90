program sparse_cpu
    use iso_c_binding
    use ieee_arithmetic
    use amd_poisson_csr
    use aocl_sparse_c
    implicit none
    integer, parameter :: nx=1024,ny=1024
    integer :: status,ix,iy,p,neighbors
    real(c_double) :: max_error,expected
    logical :: failed=.false.
    integer(c_int), allocatable, target :: row_offsets(:),col_indices(:)
    real(c_double), allocatable, target :: values(:),vec_x(:),vec_y(:)
    real(c_double), target :: alpha,beta
    integer(c_int) :: nrow,ncol,nnz
    type(c_ptr) :: mat_A=c_null_ptr
    type(c_ptr) :: descr=c_null_ptr

    call make_poisson2d_csr(nx,ny,nrow,ncol,nnz,row_offsets,col_indices,values)
    allocate(vec_x(ncol),vec_y(nrow),stat=status)
    call check(status,'host allocation')
    vec_x=1; vec_y=1; alpha=1; beta=1
    call check(aoclsparse_create_mat_descr(descr),'aoclsparse_create_mat_descr')
    call check(aoclsparse_set_mat_index_base(descr,aoclsparse_index_base_zero),'AOCL index base')
    call check(aoclsparse_set_mat_type(descr,aoclsparse_matrix_type_general),'AOCL matrix type')
    call check(aoclsparse_set_mat_diag_type(descr,aoclsparse_diag_type_non_unit),'AOCL diagonal type')
    call check(aoclsparse_create_dcsr(mat_A,aoclsparse_index_base_zero,nrow,ncol,nnz, &
               row_offsets,col_indices,values),'aoclsparse_create_dcsr')
    call check(aoclsparse_set_mv_hint(mat_A,aoclsparse_operation_none,descr,1_c_int), &
               'aoclsparse_set_mv_hint')
    call check(aoclsparse_optimize(mat_A),'aoclsparse_optimize')
    call check(aoclsparse_dmv(aoclsparse_operation_none,alpha,mat_A,descr,vec_x,beta,vec_y),'aoclsparse_dmv')

    max_error=0
    call require(all(ieee_is_finite(vec_y)),'nonfinite SpMV result')
    do iy=1,ny
        do ix=1,nx
            p=(iy-1)*nx+ix
            neighbors=0
            if (ix>1) neighbors=neighbors+1
            if (ix<nx) neighbors=neighbors+1
            if (iy>1) neighbors=neighbors+1
            if (iy<ny) neighbors=neighbors+1
            expected=real(5-neighbors,c_double)
            max_error=max(max_error,abs(vec_y(p)-expected))
        end do
    end do
    call release()
    if (failed) error stop 'resource cleanup failed'
    deallocate(row_offsets,col_indices,values,vec_x,vec_y)
    print '(a,es16.8)', 'Poisson SpMV complete; max error = ',max_error
    if (max_error>1.0e-12_c_double) error stop 'SpMV verification failed'
contains
    subroutine check(code,operation)
        integer, intent(in) :: code
        character(*), intent(in) :: operation
        if (code/=0) then
            print *,trim(operation),' failed: ',code
            call release()
            error stop 1
        end if
    end subroutine
    subroutine require(condition,message)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        if (.not.condition) then
            print *,message
            call release()
            error stop 1
        end if
    end subroutine
    subroutine record_cleanup(code)
        integer, intent(in) :: code
        if (code/=0) failed=.true.
    end subroutine
    subroutine release()
        if (c_associated(mat_A)) call record_cleanup(aoclsparse_destroy(mat_A))
        if (c_associated(descr)) call record_cleanup(aoclsparse_destroy_mat_descr(descr))
        mat_A=c_null_ptr; descr=c_null_ptr
    end subroutine
end program sparse_cpu
