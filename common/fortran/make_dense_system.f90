! Same canonical system as the C/C++ examples; column-major, FP64.
subroutine make_dense_system(n, nrhs, mat_A, rhs_B)
    implicit none
    integer, intent(in) :: n, nrhs
    real(8), intent(out) :: mat_A(n,n), rhs_B(n,nrhs)
    integer :: i
    if (n < 1 .or. nrhs < 1) error stop 'invalid dense system dimensions'
    mat_A = 1.0d0
    do i = 1, n
        mat_A(i,i) = real(n,8)+1.0d0
    end do
    rhs_B = 2.0d0*real(n,8)
end subroutine make_dense_system
