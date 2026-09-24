! Test-only adapter module to controlled C fixtures, NOT the oneMKL module/ABI.
module mkl_spblas
    use iso_c_binding
    implicit none
    integer, parameter :: SPARSE_INDEX_BASE_ONE=1,SPARSE_OPERATION_NON_TRANSPOSE=10
    integer, parameter :: SPARSE_MATRIX_TYPE_GENERAL=20,SPARSE_FILL_MODE_FULL=30,SPARSE_DIAG_NON_UNIT=40
    type, bind(C) :: sparse_matrix_t
        type(c_ptr) :: handle
    end type
    type, bind(C) :: matrix_descr
        integer(c_int) :: type,mode,diag
    end type
    interface
        integer(c_int) function mkl_sparse_d_create_csr(matrix,indexing,rows,columns, &
                                                       row_start,row_end,col,values) bind(C)
            import sparse_matrix_t,c_int,c_double
            type(sparse_matrix_t) :: matrix
            integer(c_int), value :: indexing,rows,columns
            integer(c_int) :: row_start(*),row_end(*),col(*)
            real(c_double) :: values(*)
        end function
        integer(c_int) function c_hint(matrix,operation,descr,count) bind(C,name='mkl_sparse_set_mv_hint')
            import c_ptr,c_int,matrix_descr
            type(c_ptr), value :: matrix
            integer(c_int), value :: operation,count
            type(matrix_descr), value :: descr
        end function
        integer(c_int) function c_optimize(matrix) bind(C,name='mkl_sparse_optimize')
            import c_ptr,c_int
            type(c_ptr), value :: matrix
        end function
        integer(c_int) function c_mv(operation,alpha,matrix,descr,x,beta,y) bind(C,name='mkl_sparse_d_mv')
            import c_ptr,c_int,c_double,matrix_descr
            integer(c_int), value :: operation
            real(c_double), value :: alpha,beta
            type(c_ptr), value :: matrix
            type(matrix_descr), value :: descr
            real(c_double) :: x(*),y(*)
        end function
        integer(c_int) function c_destroy(matrix) bind(C,name='mkl_sparse_destroy')
            import c_ptr,c_int
            type(c_ptr), value :: matrix
        end function
    end interface
contains
    integer function mkl_sparse_set_mv_hint(matrix,operation,descr,count)
        type(sparse_matrix_t) :: matrix
        integer :: operation,count
        type(matrix_descr) :: descr
        mkl_sparse_set_mv_hint=c_hint(matrix%handle,operation,descr,count)
    end function
    integer function mkl_sparse_optimize(matrix)
        type(sparse_matrix_t) :: matrix
        mkl_sparse_optimize=c_optimize(matrix%handle)
    end function
    integer function mkl_sparse_d_mv(operation,alpha,matrix,descr,x,beta,y)
        integer :: operation
        real(c_double) :: alpha,beta,x(*),y(*)
        type(sparse_matrix_t) :: matrix
        type(matrix_descr) :: descr
        mkl_sparse_d_mv=c_mv(operation,alpha,matrix%handle,descr,x,beta,y)
    end function
    integer function mkl_sparse_destroy(matrix)
        type(sparse_matrix_t) :: matrix
        mkl_sparse_destroy=c_destroy(matrix%handle)
    end function
end module
