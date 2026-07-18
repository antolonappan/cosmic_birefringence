module tools_fast_omp
  implicit none
contains

  pure integer function equation_count(nob) result(n)
    integer, intent(in) :: nob
    n = 4*nob*nob - 2*nob
  end function equation_count

  pure logical function skip_equation(i,j,im,jm)
    integer, intent(in) :: i,j,im,jm
    skip_equation = (im == jm .and. i == j)
  end function skip_equation

  pure subroutine r_matrix(ai, aj, inverse, r)
    real(8), intent(in) :: ai, aj
    logical, intent(in) :: inverse
    real(8), intent(out) :: r(2,2)
    real(8) :: d, a, amp
    d = cos(ai)*cos(aj); a = sin(ai)*sin(aj)
    if (inverse) then
      amp = 2d0/(cos(2d0*ai)+cos(2d0*aj))
      r = reshape([d,-a,-a,d],[2,2])*amp
    else
      r = reshape([d,a,a,d],[2,2])
    end if
  end subroutine r_matrix

  pure subroutine r_vector(ai, aj, r)
    real(8), intent(in) :: ai, aj
    real(8), intent(out) :: r(2)
    r = [cos(ai)*sin(aj), -sin(ai)*cos(aj)]
  end subroutine r_vector

  pure subroutine blocks_unprimed(ai, aj, bi, bj, ablock, bblock)
    real(8), intent(in) :: ai,aj,bi,bj
    real(8), intent(out) :: ablock(3), bblock(2)
    real(8) :: rv(2), rvb(2), ri(2,2), rb(2,2), rm(2,2)
    call r_vector(ai,aj,rv); call r_matrix(ai,aj,.true.,ri)
    call r_vector(ai+bi,aj+bj,rvb); call r_matrix(ai+bi,aj+bj,.false.,rb)
    rm = matmul(ri,rb)
    ablock(1:2) = -matmul(rv,ri); ablock(3)=1d0
    bblock = rvb - matmul(rv,rm)
  end subroutine blocks_unprimed

  pure subroutine blocks_primed(ai, aj, bi, bj, psi, dust_a, ablock, bblock)
    real(8), intent(in) :: ai,aj,bi,bj,psi,dust_a
    real(8), intent(out) :: ablock(3), bblock(2)
    real(8) :: x, rv(2), dv(2), dm(2,2), lm(2,2), li(2,2), lv(2)
    real(8) :: rb(2,2), rvb(2), minusv(2), determinant
    x = dust_a*sin(4d0*psi)
    call r_vector(ai,aj,rv)
    dv = [cos(ai)*cos(aj), -sin(ai)*sin(aj)]
    dm = reshape([-cos(ai)*sin(aj), cos(aj)*sin(ai), &
                  -cos(aj)*sin(ai), cos(ai)*sin(aj)],[2,2])
    call r_matrix(ai,aj,.false.,lm)
    lv = rv; lv(1) = lv(1) + x*(dv(1)+dv(2))
    lm(:,1) = lm(:,1) + x*(dm(:,1)+dm(:,2))
    determinant = lm(1,1)*lm(2,2)-lm(1,2)*lm(2,1)
    li(1,1)=lm(2,2)/determinant; li(2,2)=lm(1,1)/determinant
    li(1,2)=-lm(1,2)/determinant; li(2,1)=-lm(2,1)/determinant
    minusv = -matmul(lv,li)
    ablock = [minusv(1),minusv(2),1d0]
    call r_vector(ai+bi,aj+bj,rvb); call r_matrix(ai+bi,aj+bj,.false.,rb)
    bblock = rvb + matmul(minusv,rb)
  end subroutine blocks_primed

  pure real(8) function select_a(k,a) result(value)
    integer, intent(in) :: k
    real(8), intent(in) :: a(:)
    integer :: n
    n=size(a)
    select case(n)
    case(1)
      value=a(1)
    case(2)
      if(k<23)then; value=a(1); else; value=a(2); endif
    case(3)
      if(k<8)then
        value=a(1)
      elseif(k<23)then
        value=a(2)
      else
        value=a(3)
      endif
    case(4)
      if(k<4)then
        value=a(1)
      elseif(k<8)then
        value=a(2)
      elseif(k<23)then
        value=a(3)
      else
        value=a(4)
      endif
    case(5)
      if(k<6)then
        value=a(1)
      elseif(k<13)then
        value=a(2)
      elseif(k<21)then
        value=a(3)
      elseif(k<38)then
        value=a(4)
      else
        value=a(5)
      endif
    case(6)
      if(k<4)then
        value=a(1)
      elseif(k<8)then
        value=a(2)
      elseif(k<13)then
        value=a(3)
      elseif(k<19)then
        value=a(4)
      elseif(k<38)then
        value=a(5)
      else
        value=a(6)
      endif
    case default
      value=0d0
    end select
  end function select_a

  subroutine build_matrices(nob,alpha,beta,psi,a,k,turnoff,model_dust,amat,bmat)
    integer, intent(in) :: nob,k,turnoff
    real(8), intent(in) :: alpha(:),beta(:),psi(:),a(:)
    logical, intent(in) :: model_dust
    real(8), intent(out) :: amat(equation_count(nob),3*equation_count(nob))
    real(8), intent(out) :: bmat(equation_count(nob),2*equation_count(nob))
    integer :: i,j,im,jm,v,ci,cj
    real(8) :: bi,bj,ab(3),bb(2),av
    amat=0d0; bmat=0d0; v=0
    if(model_dust) av=select_a(k,a)
    do i=0,nob-1; do j=0,nob-1; do im=0,1; do jm=0,1
      if(skip_equation(i,j,im,jm)) cycle
      v=v+1; ci=im*nob+i+1; cj=jm*nob+j+1
      if(size(beta)==1)then; bi=2d0*beta(1);bj=bi
      else;bi=2d0*beta(i+1);bj=2d0*beta(j+1);endif
      if(model_dust .and. i>=turnoff .and. j>=turnoff)then
        call blocks_primed(2d0*alpha(ci),2d0*alpha(cj),bi,bj,psi(k+1),av,ab,bb)
      else
        call blocks_unprimed(2d0*alpha(ci),2d0*alpha(cj),bi,bj,ab,bb)
      endif
      amat(v,3*v-2:3*v)=ab; bmat(v,2*v-1:2*v)=bb
    enddo;enddo;enddo;enddo
  end subroutine build_matrices

  subroutine likelihood_core(clo,clt,cov,ln_det,nob,alpha,beta,psi,a,turnoff,model_dust,prob)
    real(8), intent(in) :: clo(:,:),clt(:,:),cov(:,:,:),alpha(:),beta(:),psi(:),a(:)
    logical, intent(in) :: ln_det,model_dust
    integer, intent(in) :: nob,turnoff
    real(8), intent(out) :: prob
    integer :: nb,neq,k,info
    real(8) :: contribution
    external dpotrf,dpotrs
    nb=size(clo,1); neq=equation_count(nob); prob=0d0
    !$omp parallel do default(shared) private(k,info,contribution) reduction(+:prob)
    do k=0,nb-1
      block
        integer :: i
        real(8) :: am(neq,3*neq),bm(neq,2*neq),v(neq),cm(neq,neq),rhs(neq,1)
        call build_matrices(nob,alpha,beta,psi,a,k,turnoff,model_dust,am,bm)
        v=matmul(am,clo(k+1,:))-matmul(bm,clt(k+1,:))
        cm=matmul(matmul(am,cov(k+1,:,:)),transpose(am))
        rhs(:,1)=v
        call dpotrf('L',neq,cm,neq,info)
        if(info==0)then
          contribution=0d0
          if(ln_det) contribution=2d0*sum(log([(cm(i,i),i=1,neq)]))
          call dpotrs('L',neq,1,cm,neq,rhs,neq,info)
          contribution=contribution+dot_product(v,rhs(:,1))
          prob=prob+contribution
        else
          prob=prob+huge(1d0)
        endif
      end block
    enddo
    !$omp end parallel do
    prob=-0.5d0*prob
  end subroutine likelihood_core

  subroutine likelihood_prob_impl(clo,clt,cov,ln_det,nob,alpha,beta,prob)
    real(8), intent(in) :: clo(:,:),clt(:,:),cov(:,:,:),alpha(:),beta(:)
    logical, intent(in) :: ln_det
    integer, intent(in) :: nob
    real(8), intent(out) :: prob
    real(8) :: dummy(1)
    dummy=0d0
    call likelihood_core(clo,clt,cov,ln_det,nob,alpha,beta,dummy,dummy,0,.false.,prob)
  end subroutine likelihood_prob_impl

  subroutine likelihood_prob_model_eb_impl(clo,clt,cov,ln_det,nob,alpha,beta,psi,a,turnoff,prob)
    real(8), intent(in) :: clo(:,:),clt(:,:),cov(:,:,:),alpha(:),beta(:),psi(:),a(:)
    logical, intent(in) :: ln_det
    integer, intent(in) :: nob,turnoff
    real(8), intent(out) :: prob
    call likelihood_core(clo,clt,cov,ln_det,nob,alpha,beta,psi,a,turnoff,.true.,prob)
  end subroutine likelihood_prob_model_eb_impl
end module tools_fast_omp

subroutine likelihood_prob(clo,clt,cov,ln_det,nob,alpha,beta,prob)
  use tools_fast_omp, only: likelihood_prob_impl
  implicit none
  real(8), intent(in) :: clo(:,:),clt(:,:),cov(:,:,:),alpha(:),beta(:)
  logical, intent(in) :: ln_det
  integer, intent(in) :: nob
  real(8), intent(out) :: prob
  call likelihood_prob_impl(clo,clt,cov,ln_det,nob,alpha,beta,prob)
end subroutine likelihood_prob

subroutine likelihood_prob_model_eb(clo,clt,cov,ln_det,nob,alpha,beta,psi,a,turnoff,prob)
  use tools_fast_omp, only: likelihood_prob_model_eb_impl
  implicit none
  real(8), intent(in) :: clo(:,:),clt(:,:),cov(:,:,:),alpha(:),beta(:),psi(:),a(:)
  logical, intent(in) :: ln_det
  integer, intent(in) :: nob,turnoff
  real(8), intent(out) :: prob
  call likelihood_prob_model_eb_impl(clo,clt,cov,ln_det,nob,alpha,beta,psi,a,turnoff,prob)
end subroutine likelihood_prob_model_eb
