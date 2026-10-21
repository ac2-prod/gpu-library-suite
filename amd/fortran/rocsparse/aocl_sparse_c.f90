! Explicit LP64 AOCL Sparse C ABI. Compile and link this module as an input.
! API: https://github.com/amd/aocl-sparse/tree/master/library/include
module aocl_sparse_c
    use iso_c_binding
    implicit none
    private
    public :: aoclsparse_operation_none,aoclsparse_index_base_zero,aoclsparse_matrix_type_general
    public :: aoclsparse_diag_type_non_unit,aoclsparse_create_mat_descr,aoclsparse_destroy_mat_descr
    public :: aoclsparse_set_mat_index_base,aoclsparse_set_mat_type,aoclsparse_set_mat_diag_type
    public :: aoclsparse_create_dcsr,aoclsparse_set_mv_hint,aoclsparse_optimize,aoclsparse_dmv,aoclsparse_destroy
    integer(c_int), parameter :: aoclsparse_operation_none=111_c_int
    integer(c_int), parameter :: aoclsparse_index_base_zero=0_c_int
    integer(c_int), parameter :: aoclsparse_matrix_type_general=0_c_int
    integer(c_int), parameter :: aoclsparse_diag_type_non_unit=0_c_int
    interface
        integer(c_int) function aoclsparse_create_mat_descr(descr) bind(C)
            import c_int,c_ptr
            type(c_ptr), intent(out) :: descr
        end function
        integer(c_int) function aoclsparse_destroy_mat_descr(descr) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: descr
        end function
        integer(c_int) function aoclsparse_set_mat_index_base(descr,base) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: descr
            integer(c_int), value :: base
        end function
        integer(c_int) function aoclsparse_set_mat_type(descr,kind) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: descr
            integer(c_int), value :: kind
        end function
        integer(c_int) function aoclsparse_set_mat_diag_type(descr,kind) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: descr
            integer(c_int), value :: kind
        end function
        integer(c_int) function aoclsparse_create_dcsr(matrix,base,m,n,nnz,row,col,val) bind(C)
            import c_int,c_ptr,c_double
            type(c_ptr), intent(out) :: matrix
            integer(c_int), value :: base,m,n,nnz
            integer(c_int), intent(in) :: row(*),col(*)
            real(c_double), intent(in) :: val(*)
        end function
        integer(c_int) function aoclsparse_set_mv_hint(matrix,op,descr,calls) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: matrix,descr
            integer(c_int), value :: op,calls
        end function
        integer(c_int) function aoclsparse_optimize(matrix) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: matrix
        end function
        integer(c_int) function aoclsparse_dmv(op,alpha,matrix,descr,x,beta,y) bind(C)
            import c_int,c_ptr,c_double
            integer(c_int), value :: op
            real(c_double), intent(in) :: alpha,beta,x(*)
            type(c_ptr), value :: matrix,descr
            real(c_double), intent(inout) :: y(*)
        end function
        integer(c_int) function aoclsparse_destroy(matrix) bind(C)
            import c_int,c_ptr
            type(c_ptr), intent(inout) :: matrix
        end function
    end interface
end module aocl_sparse_c
