module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
    use amd_poisson_csr
#if GPU_SUITE_IMPL == 0
    use aocl_sparse_c
#else
    use hipfort, only: hipMalloc,hipFree,hipMemcpy,hipSetDevice,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use rocsparse_v2_c
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
    integer(c_int), allocatable, target :: row_offsets(:),col_indices(:)
    real(c_double), allocatable, target :: values(:),vec_x(:),vec_y(:)
    real(c_double), target :: alpha,beta
    integer(c_int) :: nrow,ncol,nnz
    type(c_ptr) :: mat_A=c_null_ptr
#if GPU_SUITE_IMPL == 0
    type(c_ptr) :: descr=c_null_ptr
#else
    type(c_ptr) :: handle=c_null_ptr,x_descr=c_null_ptr,y_descr=c_null_ptr,spmv=c_null_ptr
    type(c_ptr) :: d_row=c_null_ptr,d_col=c_null_ptr,d_values=c_null_ptr,d_x=c_null_ptr,d_y=c_null_ptr
    type(c_ptr) :: workspace=c_null_ptr
    integer(c_size_t) :: buffer_size=0
    integer(c_int), target :: alg=rocsparse_spmv_alg_csr_rowsplit
    integer(c_int), target :: op=rocsparse_operation_none,datatype=rocsparse_datatype_f64_r
#endif
    logical :: data_present=.false.
#if GPU_SUITE_IMPL == 2
    interface
        integer(c_int) function gpu_suite_f_check_mapped_device(pointer,device) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: pointer
            integer(c_int), value :: device
        end function
    end interface
#endif
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: rows,nonzeros
#if GPU_SUITE_IMPL == 2
        logical :: on_host
#endif
        rows=checked_product(o%nx,o%ny)
        call require(rows<int(huge(0_c_int),c_int64_t),'CSR rows overflow LP64')
        nonzeros=5*rows-2_c_int64_t*o%nx-2_c_int64_t*o%ny
        call require(nonzeros<int(huge(0_c_int),c_int64_t),'CSR nonzeros overflow LP64')
        call require(storage_size(0_c_int)==32,'AOCL/rocSPARSE requires 32-bit CSR indices')
        alpha=o%alpha; beta=o%beta
#if GPU_SUITE_IMPL != 0
        call check(hipSetDevice(int(o%device,c_int)),'hipSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call require(omp_get_default_device()>=0 .and. omp_get_default_device()<omp_get_num_devices(), &
                     'OpenMP default device unavailable')
        on_host=.true.
        !$omp target map(from:on_host)
        on_host=omp_is_initial_device()
        !$omp end target
        call require(.not.on_host,'OpenMP target fell back to host')
#endif
        call make_poisson2d_csr(o%nx,o%ny,nrow,ncol,nnz,row_offsets,col_indices,values)
        allocate(vec_x(ncol),vec_y(nrow),stat=status)
        call check(status,'host allocation')
        vec_x=1
    end subroutine
    subroutine reset()
        vec_y=1
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 0
        call check(aoclsparse_create_mat_descr(descr),'aoclsparse_create_mat_descr')
        call check(aoclsparse_set_mat_index_base(descr,aoclsparse_index_base_zero),'AOCL index base')
        call check(aoclsparse_set_mat_type(descr,aoclsparse_matrix_type_general),'AOCL matrix type')
        call check(aoclsparse_set_mat_diag_type(descr,aoclsparse_diag_type_non_unit),'AOCL diagonal type')
        call check(aoclsparse_create_dcsr(mat_A,aoclsparse_index_base_zero,nrow,ncol,nnz, &
                   row_offsets,col_indices,values),'aoclsparse_create_dcsr')
        call check(aoclsparse_set_mv_hint(mat_A,aoclsparse_operation_none,descr,1_c_int), &
                   'aoclsparse_set_mv_hint')
        call check(aoclsparse_optimize(mat_A),'aoclsparse_optimize')
#else
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_row,size(row_offsets,kind=c_size_t)*c_sizeof(row_offsets(1))),'allocate row offsets')
        call check(hipMalloc(d_col,size(col_indices,kind=c_size_t)*c_sizeof(col_indices(1))),'allocate columns')
        call check(hipMalloc(d_values,size(values,kind=c_size_t)*c_sizeof(values(1))),'allocate values')
        call check(hipMalloc(d_x,size(vec_x,kind=c_size_t)*c_sizeof(vec_x(1))),'allocate x')
        call check(hipMalloc(d_y,size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1))),'allocate y')
        call check(hipMemcpy(d_row,c_loc(row_offsets(1)),size(row_offsets,kind=c_size_t)*c_sizeof(row_offsets(1)), &
                   hipMemcpyHostToDevice),'copy CSR offsets')
        call check(hipMemcpy(d_col,c_loc(col_indices(1)),size(col_indices,kind=c_size_t)*c_sizeof(col_indices(1)), &
                   hipMemcpyHostToDevice),'copy CSR columns')
        call check(hipMemcpy(d_values,c_loc(values(1)),size(values,kind=c_size_t)*c_sizeof(values(1)), &
                   hipMemcpyHostToDevice),'copy CSR values')
        call check(hipMemcpy(d_x,c_loc(vec_x(1)),size(vec_x,kind=c_size_t)*c_sizeof(vec_x(1)), &
                   hipMemcpyHostToDevice),'copy x')
        call restore()
#else
        call check(rocsparse_create_handle(handle),'rocsparse_create_handle')
        !$omp target enter data map(to:row_offsets,col_indices,values,vec_x,vec_y)
        data_present=.true.
        !$omp target data use_device_addr(row_offsets,col_indices,values,vec_x,vec_y)
        d_row=c_loc(row_offsets(1)); d_col=c_loc(col_indices(1)); d_values=c_loc(values(1))
        d_x=c_loc(vec_x(1)); d_y=c_loc(vec_y(1))
        !$omp end target data
        call check(gpu_suite_f_check_mapped_device(d_row,int(o%device,c_int)),'CSR offsets device mapping')
        call check(gpu_suite_f_check_mapped_device(d_col,int(o%device,c_int)),'CSR columns device mapping')
        call check(gpu_suite_f_check_mapped_device(d_values,int(o%device,c_int)),'CSR values device mapping')
        call check(gpu_suite_f_check_mapped_device(d_x,int(o%device,c_int)),'x device mapping')
        call check(gpu_suite_f_check_mapped_device(d_y,int(o%device,c_int)),'y device mapping')
#endif
#if GPU_SUITE_IMPL == 1
        call check(rocsparse_create_handle(handle),'rocsparse_create_handle')
#endif
        call check(rocsparse_create_csr_descr(mat_A,int(nrow,c_int64_t),int(ncol,c_int64_t),int(nnz,c_int64_t), &
                   d_row,d_col,d_values,rocsparse_indextype_i32,rocsparse_indextype_i32, &
                   rocsparse_index_base_zero,rocsparse_datatype_f64_r),'rocsparse_create_csr_descr')
        call check(rocsparse_create_dnvec_descr(x_descr,int(ncol,c_int64_t),d_x,rocsparse_datatype_f64_r), &
                   'rocsparse_create_dnvec_descr x')
        call check(rocsparse_create_dnvec_descr(y_descr,int(nrow,c_int64_t),d_y,rocsparse_datatype_f64_r), &
                   'rocsparse_create_dnvec_descr y')
        call check(rocsparse_create_spmv_descr(spmv),'rocsparse_create_spmv_descr')
        call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_alg,c_loc(alg),c_sizeof(alg),c_null_ptr), &
                   'SpMV algorithm')
        call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_operation,c_loc(op),c_sizeof(op),c_null_ptr), &
                   'SpMV operation')
        call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_scalar_datatype, &
                   c_loc(datatype),c_sizeof(datatype),c_null_ptr),'SpMV scalar type')
        call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_compute_datatype, &
                   c_loc(datatype),c_sizeof(datatype),c_null_ptr),'SpMV compute type')
        buffer_size=0
        call check(rocsparse_v2_spmv_buffer_size(handle,spmv,mat_A,x_descr,y_descr, &
                   rocsparse_v2_spmv_stage_analysis,buffer_size,c_null_ptr),'SpMV analysis workspace size')
        call require(buffer_size>=0,'SpMV workspace size overflow')
        if (buffer_size>0) call check(hipMalloc(workspace,buffer_size),'allocate analysis workspace')
        call check(rocsparse_v2_spmv(handle,spmv,c_loc(alpha),mat_A,x_descr,c_loc(beta),y_descr, &
                   rocsparse_v2_spmv_stage_analysis,buffer_size,workspace,c_null_ptr),'SpMV analysis')
        if (c_associated(workspace)) call check(hipFree(workspace),'free analysis workspace')
        workspace=c_null_ptr
        call check(rocsparse_v2_spmv_buffer_size(handle,spmv,mat_A,x_descr,y_descr, &
                   rocsparse_v2_spmv_stage_compute,buffer_size,c_null_ptr),'SpMV compute workspace size')
        call require(buffer_size>=0,'SpMV workspace size overflow')
        if (buffer_size>0) call check(hipMalloc(workspace,buffer_size),'allocate compute workspace')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(d_y,c_loc(vec_y(1)),size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1)), &
                   hipMemcpyHostToDevice),'restore initial y')
#elif GPU_SUITE_IMPL == 2
        !$omp target update to(vec_y)
#endif
    end subroutine
    subroutine operation(repetitions,timed)
        integer, intent(in) :: repetitions
        logical, intent(in) :: timed
        integer :: repetition
        if (timed) call start_timer()
        do repetition=1,repetitions
#if GPU_SUITE_IMPL == 0
        call check(aoclsparse_dmv(aoclsparse_operation_none,alpha,mat_A,descr,vec_x,beta,vec_y),'aoclsparse_dmv')
#else
        ! The descriptors retain device addresses until cleanup; no per-repeat synchronization.
        call check(rocsparse_v2_spmv(handle,spmv,c_loc(alpha),mat_A,x_descr,c_loc(beta),y_descr, &
                   rocsparse_v2_spmv_stage_compute,buffer_size,workspace,c_null_ptr),'SpMV compute')
#endif
        end do
        if (timed .or. o%scope /= 0) call synchronize()
        if (timed) call end_timer()
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(hipDeviceSynchronize(),'hipDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(c_loc(vec_y(1)),d_y,size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1)), &
                   hipMemcpyDeviceToHost),'copy y result')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            ! Match C/C++ structured target-data exit before the end timestamp.
            !$omp target exit data map(from:vec_y) map(delete:row_offsets,col_indices,values,vec_x)
            data_present=.false.
        else
            !$omp target update from(vec_y)
            call check(hipDeviceSynchronize(),'OpenMP result completion')
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 0
        call check(aoclsparse_destroy(mat_A),'aoclsparse_destroy')
        call check(aoclsparse_destroy_mat_descr(descr),'aoclsparse_destroy_mat_descr')
        mat_A=c_null_ptr; descr=c_null_ptr
#else
        call check(rocsparse_destroy_spmv_descr(spmv),'rocsparse_destroy_spmv_descr')
        call check(rocsparse_destroy_dnvec_descr(y_descr),'rocsparse_destroy_dnvec_descr y')
        call check(rocsparse_destroy_dnvec_descr(x_descr),'rocsparse_destroy_dnvec_descr x')
        call check(rocsparse_destroy_spmat_descr(mat_A),'rocsparse_destroy_spmat_descr')
        if (c_associated(workspace)) call check(hipFree(workspace),'free compute workspace')
        workspace=c_null_ptr
        call check(rocsparse_destroy_handle(handle),'rocsparse_destroy_handle')
        spmv=c_null_ptr; y_descr=c_null_ptr; x_descr=c_null_ptr; mat_A=c_null_ptr; handle=c_null_ptr
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_y),'free y')
        call check(hipFree(d_x),'free x')
        call check(hipFree(d_values),'free values')
        call check(hipFree(d_col),'free columns')
        call check(hipFree(d_row),'free row offsets')
#else
        if (data_present) then
            !$omp target exit data map(delete:row_offsets,col_indices,values,vec_x,vec_y)
            data_present=.false.
        end if
#endif
        d_y=c_null_ptr; d_x=c_null_ptr; d_values=c_null_ptr; d_col=c_null_ptr; d_row=c_null_ptr
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
                    expected=alpha*real(4-neighbors,c_double)+beta*expected
                end do
                call require(ieee_is_finite(expected),'nonfinite analytic SpMV reference')
                scale=max(scale,abs(expected))
                if (finite) metrics(1)=max(metrics(1),abs(vec_y(p)-expected))
            end do
        end do
        if (.not.finite) metrics(1)=nan_value()
    end subroutine
    subroutine host_cleanup()
        deallocate(row_offsets,col_indices,values,vec_x,vec_y)
    end subroutine
end module gpu_suite_workload
