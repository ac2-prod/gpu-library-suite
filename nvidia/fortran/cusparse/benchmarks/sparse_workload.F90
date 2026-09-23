#if GPU_SUITE_IMPL == 0
include 'mkl_spblas.f90'
#endif
module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL == 0
    use mkl_spblas
#else
    use cudafor
    use cusparse
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
    interface
        subroutine make_poisson2d_csr(nx,ny,nrow,ncol,nnz,row_offsets,col_indices,values)
            integer, intent(in) :: nx,ny
            integer, intent(out) :: nrow,ncol,nnz
            integer, allocatable, intent(out) :: row_offsets(:),col_indices(:)
            real(8), allocatable, intent(out) :: values(:)
        end subroutine
    end interface
    integer, allocatable :: row_offsets(:),col_indices(:)
    real(c_double), allocatable :: values(:),vec_x(:),vec_y(:)
    integer :: nrow,ncol,nnz
#if GPU_SUITE_IMPL == 0
    type(sparse_matrix_t) :: mat_A
    type(matrix_descr) :: descr
#else
    type(cusparseHandle) :: handle
    type(cusparseSpMatDescr) :: mat_A
    type(cusparseDnVecDescr) :: x_descr,y_descr
    integer(c_int8_t), device, allocatable :: workspace(:)
    integer(c_int64_t) :: buffer_size
#endif
#if GPU_SUITE_IMPL == 1
    integer, device, allocatable :: d_row(:),d_col(:)
    real(c_double), device, allocatable :: d_values(:),d_x(:),d_y(:)
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: rows,nonzeros
        rows=checked_product(o%nx,o%ny)
        call require(rows<int(huge(0),c_int64_t),'CSR rows overflow LP64')
        nonzeros=5*rows-2_c_int64_t*o%nx-2_c_int64_t*o%ny
        call require(nonzeros<int(huge(0),c_int64_t),'CSR nonzeros overflow LP64')
#if GPU_SUITE_IMPL != 0
        call check(cudaSetDevice(o%device),'cudaSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
        call make_poisson2d_csr(o%nx,o%ny,nrow,ncol,nnz,row_offsets,col_indices,values)
        allocate(vec_x(ncol),vec_y(nrow),stat=status)
        call check(status,'host allocation')
    end subroutine
    subroutine reset()
        vec_x=1; vec_y=1
    end subroutine
    subroutine setup()
        integer :: status
#if GPU_SUITE_IMPL == 0
        descr%type=SPARSE_MATRIX_TYPE_GENERAL
        descr%mode=SPARSE_FILL_MODE_FULL
        descr%diag=SPARSE_DIAG_NON_UNIT
        call check(mkl_sparse_d_create_csr(mat_A,SPARSE_INDEX_BASE_ONE,nrow,ncol, &
                   row_offsets(1:nrow),row_offsets(2:nrow+1),col_indices,values),'mkl_sparse_d_create_csr')
        call check(mkl_sparse_set_mv_hint(mat_A,SPARSE_OPERATION_NON_TRANSPOSE,descr,o%repeat),'mkl_sparse_set_mv_hint')
        call check(mkl_sparse_optimize(mat_A),'mkl_sparse_optimize')
#else
#if GPU_SUITE_IMPL == 1
        allocate(d_row(nrow+1),d_col(nnz),d_values(nnz),d_x(ncol),d_y(nrow),stat=status)
        call check(status,'device allocation')
        call restore()
#else
        !$acc enter data copyin(row_offsets,col_indices,values,vec_x,vec_y)
        data_present=.true.
#endif
        call check(cusparseCreate(handle),'cusparseCreate')
#if GPU_SUITE_IMPL == 1
        call check(cusparseCreateCsr(mat_A,int(nrow,c_int64_t),int(ncol,c_int64_t),int(nnz,c_int64_t), &
                   d_row,d_col,d_values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ONE,CUDA_R_64F), &
                   'cusparseCreateCsr')
        call check(cusparseCreateDnVec(x_descr,int(ncol,c_int64_t),d_x,CUDA_R_64F),'cusparseCreateDnVec x')
        call check(cusparseCreateDnVec(y_descr,int(nrow,c_int64_t),d_y,CUDA_R_64F),'cusparseCreateDnVec y')
#else
        !$acc host_data use_device(row_offsets,col_indices,values,vec_x,vec_y)
        status=cusparseCreateCsr(mat_A,int(nrow,c_int64_t),int(ncol,c_int64_t),int(nnz,c_int64_t), &
                   row_offsets,col_indices,values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ONE,CUDA_R_64F)
        call check(status,'cusparseCreateCsr')
        status=cusparseCreateDnVec(x_descr,int(ncol,c_int64_t),vec_x,CUDA_R_64F)
        call check(status,'cusparseCreateDnVec x')
        status=cusparseCreateDnVec(y_descr,int(nrow,c_int64_t),vec_y,CUDA_R_64F)
        call check(status,'cusparseCreateDnVec y')
        !$acc end host_data
#endif
        buffer_size=0
        call check(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, &
                   o%alpha,mat_A,x_descr,o%beta,y_descr,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,buffer_size), &
                   'cusparseSpMV_bufferSize')
        call require(buffer_size>=0,'negative cuSPARSE workspace size')
        allocate(workspace(max(1_c_int64_t,buffer_size)),stat=status)
        call check(status,'cuSPARSE workspace allocation')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(d_row,row_offsets,nrow+1,cudaMemcpyHostToDevice),'copy CSR offsets')
        call check(cudaMemcpy(d_col,col_indices,nnz,cudaMemcpyHostToDevice),'copy CSR columns')
        call check(cudaMemcpy(d_values,values,nnz,cudaMemcpyHostToDevice),'copy CSR values')
        call check(cudaMemcpy(d_x,vec_x,ncol,cudaMemcpyHostToDevice),'copy x')
        call check(cudaMemcpy(d_y,vec_y,nrow,cudaMemcpyHostToDevice),'copy initial y')
#elif GPU_SUITE_IMPL == 2
        !$acc update device(vec_x,vec_y)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        call check(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,o%alpha,mat_A,descr,vec_x,o%beta,vec_y), &
                   'mkl_sparse_d_mv')
#else
        ! Descriptors retain translated addresses; do not wrap SpMV in host_data.
        call check(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,o%alpha,mat_A,x_descr, &
                   o%beta,y_descr,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,workspace),'cusparseSpMV')
#endif
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(cudaDeviceSynchronize(),'cudaDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(vec_y,d_y,nrow,cudaMemcpyDeviceToHost),'copy y result')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data copyout(vec_y) delete(row_offsets,col_indices,values,vec_x)
            data_present=.false.
        else
            !$acc update self(vec_y)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 0
        call check(mkl_sparse_destroy(mat_A),'mkl_sparse_destroy')
#else
        call check(cusparseDestroyDnVec(y_descr),'cusparseDestroyDnVec y')
        call check(cusparseDestroyDnVec(x_descr),'cusparseDestroyDnVec x')
        call check(cusparseDestroySpMat(mat_A),'cusparseDestroySpMat')
        deallocate(workspace)
        call check(cusparseDestroy(handle),'cusparseDestroy')
#if GPU_SUITE_IMPL == 1
        deallocate(d_row,d_col,d_values,d_x,d_y)
#else
        if (data_present) then
            !$acc exit data delete(row_offsets,col_indices,values,vec_x,vec_y)
            data_present=.false.
        end if
#endif
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        integer :: ix,iy,p,neighbors,j,updates
        real(c_double) :: expected
        logical :: finite
        getrf=0; getrs=0; metrics=0; scale=0
        updates=1
        if (o%scope==0) updates=o%repeat
        finite=all(ieee_is_finite(vec_y))
        do iy=1,o%ny
            do ix=1,o%nx
                p=(iy-1)*o%nx+ix; neighbors=0
                if (ix>1) neighbors=neighbors+1
                if (ix<o%nx) neighbors=neighbors+1
                if (iy>1) neighbors=neighbors+1
                if (iy<o%ny) neighbors=neighbors+1
                expected=1
                do j=1,updates
                    expected=o%alpha*real(4-neighbors,c_double)+o%beta*expected
                end do
                call require(ieee_is_finite(expected),'nonfinite analytic SpMV reference')
                scale=max(scale,abs(expected))
                if (finite) metrics(1)=max(metrics(1),abs(vec_y(p)-expected))
            end do
        end do
        if (.not. finite) metrics(1)=nan_value()
    end subroutine
    subroutine host_cleanup()
        deallocate(row_offsets,col_indices,values,vec_x,vec_y)
    end subroutine
end module gpu_suite_workload
