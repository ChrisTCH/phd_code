
program strfunc

  use config_module
  use random_module, only : initgen, randomu, randomz
  use fits_module  , only : fits_get_dims, fits_get_data, fits_get_data_1d   &
                          , fits_put_data, fits_put_data_2d, fits_put_data_1d

  implicit none

  character(len = 255) :: filename, datafile
  character(len = 255), dimension(4) :: fl

  integer(kind=8) :: lb, le
  integer               :: status, length, plan, nlens, grainsize = 32
  integer               :: i, j, k, l, s, p, ib, jb, kb, ie, je, ke, li, lj
  integer, dimension(:), allocatable :: seed
  integer, dimension(3) :: dims, ndim, pb, pe
  real   , dimension(3) :: xc, xd, xb, xe, bb, dv, mx
  real                  :: fc, lx, ly, ll, qq, dq, pr
  integer, save         :: kpar = 0, npar = 1

  real, dimension(:,:,:), allocatable :: tmp, dn, vx, vy, vz, bx, by, bz, abb, abvel
  real, dimension(:,:)  , allocatable :: sf, cn, rb, rd
  real, dimension(:,:)  , allocatable, save :: lsf, lcn
  real, dimension(:)    , allocatable :: ls

  logical         :: scalar = .true.
  logical(kind=4) :: result

  real, parameter :: pi = 3.1415926535897931159979634685442

!$  integer :: omp_get_num_threads, omp_get_thread_num

!$omp threadprivate(lsf,lcn,kpar)
!
!-------------------------------------------------------------------------------
!
  call system_clock(lb)

! read parameters from config.in
!
  write( *, "('TASK    : ',a)" ) "computing structure functions"
  write( *, "('INFO    : ',a)" ) "reading parameters"
  call read_config_file

  write( *, "('INDIR   : ',a)" ) trim(inp_dir)
  write( *, "('OUDIR   : ',a)" ) trim(out_dir)
  write( *, "('FRAME   : ',a)" ) trim(frame)
  write( *, "('FIELD   : ',a)" ) trim(field)

! determine if scalar or vector and fill filenames
!
  select case (field)
  case("dens", "logd")
    fl(1) = "dens.fits.gz"
  case("abb")
    fl(1) = "bb.fits.gz"
  case("abvel")
    fl(1) = "abvel.fits.gz"
  case("velo", "veld")
    fl(1) = "velx.fits.gz"
    fl(2) = "vely.fits.gz"
    fl(3) = "velz.fits.gz"
    fl(4) = "dens.fits.gz"
    scalar = .false.
  case("magn", "magd")
    fl(1) = "magx.fits.gz"
    fl(2) = "magy.fits.gz"
    fl(3) = "magz.fits.gz"
    fl(4) = "dens.fits.gz"
    scalar = .false.
  case default
    write (*,'("WARNING: field ",a4," is not implemented!")') trim(field)
    stop
  end select

  write( *, "('INFO    : ',a)" ) "reading variables"

! get field dimensions
!
  write(datafile, '(a)') trim(inp_dir) // trim(fl(1))
  call fits_get_dims(datafile, dims, status)

  if (scalar) then
    allocate(dn(dims(1), dims(2), dims(3)))
    allocate(abb(dims(1), dims(2), dims(3)))
    allocate(abvel(dims(1), dims(2), dims(3)))
  else
    if (field .eq. 'veld') &
      allocate(dn(dims(1), dims(2), dims(3)))
    allocate(vx(dims(1), dims(2), dims(3)))
    allocate(vy(dims(1), dims(2), dims(3)))
    allocate(vz(dims(1), dims(2), dims(3)))
  endif

  if (frame .eq. "loc") then
    allocate(bx (dims(1), dims(2), dims(3)))
    allocate(by (dims(1), dims(2), dims(3)))
    allocate(bz (dims(1), dims(2), dims(3)))
  endif

! read data
!
  if (scalar) then
    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(1))
    write( *, "('FILE    : ',a)" ) trim(fl(1))
    call fits_get_data(datafile, dn, status)
 
write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(1))
    write( *, "('FILE    : ',a)" ) trim(fl(1))
    call fits_get_data(datafile, abb, status)

write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(1))
    write( *, "('FILE    : ',a)" ) trim(fl(1))
    call fits_get_data(datafile, abvel, status)

if (field .eq. 'logd') then
      do k = 1, dims(3)
        do j = 1, dims(2)
          do i = 1, dims(1)
            qq = alog(dn(i,j,k))
            dn(i,j,k) = qq
          enddo
        enddo
      enddo
    endif
  else
    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(1))
    write( *, "('FILE    : ',a)" ) trim(fl(1))
    call fits_get_data(datafile, vx, status)

    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(2))
    write( *, "('FILE    : ',a)" ) trim(fl(2))
    call fits_get_data(datafile, vy, status)

    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(3))
    write( *, "('FILE    : ',a)" ) trim(fl(3))
    call fits_get_data(datafile, vz, status)

    if (field .eq. 'veld' .or. field .eq. 'magd') then
      write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(fl(4))
      write( *, "('FILE    : ',a)" ) trim(fl(4))
      call fits_get_data(datafile, dn, status)

      do k = 1, dims(3)
        do j = 1, dims(2)
          do i = 1, dims(1)
            qq = dn(i,j,k)**(1./3.)

            vx(i,j,k) = qq*vx(i,j,k)
            vy(i,j,k) = qq*vy(i,j,k)
            vz(i,j,k) = qq*vz(i,j,k)
          enddo
        enddo
      enddo
    endif
  endif

  if (frame .eq. "loc") then
! read magnetic field
!
    filename = 'magx.fits.gz'
    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(filename)
    write( *, "('FILE    : ',a)" ) trim(filename)
    call fits_get_data(datafile, bx, status)

    filename = 'magy.fits.gz'
    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(filename)
    write( *, "('FILE    : ',a)" ) trim(filename)
    call fits_get_data(datafile, by, status)

    filename = 'magz.fits.gz'
    write(datafile, '(a)') trim(inp_dir) // trim(sub_dir) // trim(filename)
    write( *, "('FILE    : ',a)" ) trim(filename)
    call fits_get_data(datafile, bz, status)
  endif

! limit maxlen to the half of the box size
!
  maxlen = min(maxlen, dims(1)/2)

  length = nint(sqrt(3.0)*maxlen) + 1

  if (frame .eq. "loc") then
    allocate(sf(length,length))
    allocate(cn(length,length))
  else
    allocate(sf (maxexp,length))
    allocate(cn (maxexp,length))
  endif
  allocate(rb(3,nvecs))
  allocate(rd(3,nvecs))
  allocate(ls(length))

  sf = 0.0
  cn = 0.0
  mx = 2.*maxlen/dims
  ndim = dims - 1
  pr = 100.0 / nshots

!$omp parallel
!$  npar = omp_get_num_threads()
!$omp end parallel
  allocate(seed(1:npar))
  do i = 1, npar
    call random_number(qq)
    seed(i) = 123456789*qq
  enddo
  call initgen(npar, seed)

  write( *, "('TASK    : ',a,' (',1pe7.1,' points)')" ) "calculating", 1.*nshots*nvecs

  if (frame .eq. "loc") then

!! SCALAR
!!
    if (scalar) then

!$omp parallel
!
! allocate thread private arrays
!
      if (frame .eq. "loc") then
        allocate(lsf(length,length))
        allocate(lcn(length,length))
      else
        allocate(lsf(maxexp,length))
        allocate(lcn(maxexp,length))
      endif

! get number of thread
!
!$    kpar = omp_get_thread_num()

! iterate over random positions
!
      do p = 1, nshots

! reset local arrays to zero
!
        lsf = 0.
        lcn = 0.

!$omp single
        write(*,"('PROGRESS: ',f6.2,' % done',a1,$)") pr*p, char(13)

! generate random numbers (only the first thread reaching this step)
!
!         call random_number(rb)
!         call random_number(rd)
!$omp end single

!$omp do private(j)
        do i = 1, 3
          do j = 1, nvecs
            rb(i,j) = randomu(kpar)
            rd(i,j) = randomz(kpar)
          enddo
        enddo
!$omp end do

!$omp do private(i,xb,xd,xe,ll,pb,pe,bb,lx,ly,li,lj,dq,qq)
        do i = 1, nvecs

          xb = rb(:,i)
          xd = mx * rd(:,i)
          xe = xb + xd

          xd = dims * xd
          ll = sqrt(sum(xd**2))
          xd = xd / ll

! convert positions to indices
!
          pb = nint(ndim * (xb - floor(xb))) + 1
          pe = nint(ndim * (xe - floor(xe))) + 1

! calculate direction of the local magnetic field
!
          bb(1) = bx(pb(1),pb(2),pb(3)) + bx(pe(1),pe(2),pe(3))
          bb(2) = by(pb(1),pb(2),pb(3)) + by(pe(1),pe(2),pe(3))
          bb(3) = bz(pb(1),pb(2),pb(3)) + bz(pe(1),pe(2),pe(3))
          bb = bb / sqrt(sum(bb**2))

! calculate projection of l on the direction of B
!
          lx = sum(bb*xd)
          ly = ll*sqrt(max(0.0, 1.0 - lx*lx))
          lx = ll*abs(lx)

          li = int(lx) + 1
          lj = int(ly) + 1

! calculate the increment of analyzed field
!
          dq = dn(pe(1),pe(2),pe(3)) - dn(pb(1),pb(2),pb(3))

! add to structure function
!
          qq = dq*dq
          lsf(li,lj) = lsf(li,lj) + qq
          lcn(li,lj) = lcn(li,lj) + 1.

        enddo

! sum local values together
!
!$omp critical
        sf = sf + lsf
        cn = cn + lcn
!$omp end critical
      enddo

      if (allocated(lsf)) deallocate(lsf)
      if (allocated(lcn)) deallocate(lcn)
!$omp end parallel

    else  ! end SCALAR, start VECTOR

!$omp parallel
!
! allocate thread private arrays
!
      if (frame .eq. "loc") then
        allocate(lsf(length,length))
        allocate(lcn(length,length))
      else
        allocate(lsf(maxexp,length))
        allocate(lcn(maxexp,length))
      endif

! get number of thread
!
!$    kpar = omp_get_thread_num()

! iterate over random positions
!
      do p = 1, nshots

! reset local arrays to zero
!
        lsf = 0.
        lcn = 0.

!$omp single
        write(*,"('PROGRESS: ',f6.2,' % done',a1,$)") pr*p, char(13)

! generate random numbers (only the first thread reaching this step)
!
!         call random_number(rb)
!         call random_number(rd)
!$omp end single

!$omp do private(j)
        do i = 1, 3
          do j = 1, nvecs
            rb(i,j) = randomu(kpar)
            rd(i,j) = randomz(kpar)
          enddo
        enddo
!$omp end do

!$omp do private(i,xb,xd,xe,ll,pb,pe,bb,lx,ly,li,lj,dv,qq)
        do i = 1, nvecs

          xb = rb(:,i)
          xd = mx * rd(:,i)
          xe = xb + xd

          xd = dims * xd
          ll = sqrt(sum(xd**2))
          xd = xd / ll

! convert positions to indices
!
          pb = nint(ndim * (xb - floor(xb))) + 1
          pe = nint(ndim * (xe - floor(xe))) + 1

! calculate direction of the local magnetic field
!
          bb(1) = bx(pb(1),pb(2),pb(3)) + bx(pe(1),pe(2),pe(3))
          bb(2) = by(pb(1),pb(2),pb(3)) + by(pe(1),pe(2),pe(3))
          bb(3) = bz(pb(1),pb(2),pb(3)) + bz(pe(1),pe(2),pe(3))
          bb = bb / sqrt(sum(bb**2))

! calculate projection of l on the direction of B
!
          lx = sum(bb*xd)
          ly = ll*sqrt(max(0.0, 1.0 - lx*lx))
          lx = ll*abs(lx)

          li = int(lx) + 1
          lj = int(ly) + 1

! calculate the increment of analyzed field
!
          dv(1) = vx(pe(1),pe(2),pe(3)) - vx(pb(1),pb(2),pb(3))
          dv(2) = vy(pe(1),pe(2),pe(3)) - vy(pb(1),pb(2),pb(3))
          dv(3) = vz(pe(1),pe(2),pe(3)) - vz(pb(1),pb(2),pb(3))

! add to structure function
!
          qq = sum(dv*dv)
          lsf(li,lj) = lsf(li,lj) + qq
          lcn(li,lj) = lcn(li,lj) + 1.

        enddo

!$omp critical
        sf = sf + lsf
        cn = cn + lcn
!$omp end critical
      enddo

      if (allocated(lsf)) deallocate(lsf)
      if (allocated(lcn)) deallocate(lcn)
!$omp end parallel

    endif ! end VECTOR

  else ! frame .eq. "glo"

!! SCALAR
!!
    if (scalar) then

!$omp parallel
!
! allocate thread private arrays
!
      if (frame .eq. "loc") then
        allocate(lsf(length,length))
        allocate(lcn(length,length))
      else
        allocate(lsf(maxexp,length))
        allocate(lcn(maxexp,length))
      endif

! get number of thread
!
!$    kpar = omp_get_thread_num()

! iterate over random positions
!
      do p = 1, nshots

! reset local arrays to zero
!
        lsf = 0.
        lcn = 0.

!$omp single
        write(*,"('PROGRESS: ',f6.2,' % done',a1,$)") pr*p, char(13)

! generate random numbers (only the first thread reaching this step)
!
!         call random_number(rb)
!         call random_number(rd)
!$omp end single

!$omp do private(j)
        do i = 1, 3
          do j = 1, nvecs
            rb(i,j) = randomu(kpar)
            rd(i,j) = randomz(kpar)
          enddo
        enddo
!$omp end do

!$omp do private(i,xb,xd,xe,ll,pb,pe,bb,lx,ly,li,lj,dq,qq)
        do i = 1, nvecs

          xb = rb(:,i)
          xd = mx * rd(:,i)
          xe = xb + xd

          xd = dims * xd
          ll = sqrt(sum(xd**2))
          xd = xd / ll

! convert positions to indices
!
          pb = nint(ndim * (xb - floor(xb))) + 1
          pe = nint(ndim * (xe - floor(xe))) + 1

! calculate the increment of analyzed field
!
          dq = dn(pe(1),pe(2),pe(3)) - dn(pb(1),pb(2),pb(3))

! add to structure function
!
          lj = int(ll) + 1

          do li = 1, maxexp
            qq = abs(dq)**li
            lsf(li,lj) = lsf(li,lj) + qq
            lcn(li,lj) = lcn(li,lj) + 1.
          enddo

        enddo

!$omp critical
        sf = sf + lsf
        cn = cn + lcn
!$omp end critical
      enddo

      if (allocated(lsf)) deallocate(lsf)
      if (allocated(lcn)) deallocate(lcn)
!$omp end parallel

    else  ! end SCALAR, start VECTOR

!$omp parallel
!
! allocate thread private arrays
!
      if (frame .eq. "loc") then
        allocate(lsf(length,length))
        allocate(lcn(length,length))
      else
        allocate(lsf(maxexp,length))
        allocate(lcn(maxexp,length))
      endif

! get number of thread
!
!$    kpar = omp_get_thread_num()

! iterate over random positions
!
      do p = 1, nshots

! reset local arrays to zero
!
        lsf = 0.
        lcn = 0.

!$omp single
        write(*,"('PROGRESS: ',f6.2,' % done',a1,$)") pr*p, char(13)

! generate random numbers (only the first thread reaching this step)
!
!         call random_number(rb)
!         call random_number(rd)
!$omp end single

!$omp do private(j)
        do i = 1, 3
          do j = 1, nvecs
            rb(i,j) = randomu(kpar)
            rd(i,j) = randomz(kpar)
          enddo
        enddo
!$omp end do

!$omp do private(i,xb,xd,xe,ll,pb,pe,bb,lx,ly,li,lj,dv,qq)
        do i = 1, nvecs

          xb = rb(:,i)
          xd = mx * rd(:,i)
          xe = xb + xd

          xd = dims * xd
          ll = sqrt(sum(xd**2))

! convert positions to indices
!
          pb = nint(ndim * (xb - floor(xb))) + 1
          pe = nint(ndim * (xe - floor(xe))) + 1

! calculate the increment of analyzed field
!
          dv(1) = vx(pe(1),pe(2),pe(3)) - vx(pb(1),pb(2),pb(3))
          dv(2) = vy(pe(1),pe(2),pe(3)) - vy(pb(1),pb(2),pb(3))
          dv(3) = vz(pe(1),pe(2),pe(3)) - vz(pb(1),pb(2),pb(3))

! add to structure function
!
          lj = int(ll) + 1

          do li = 1, maxexp
            qq = sqrt(sum(dv*dv))**li
            lsf(li,lj) = lsf(li,lj) + qq
            lcn(li,lj) = lcn(li,lj) + 1.
          enddo

        enddo

!$omp critical
        sf = sf + lsf
        cn = cn + lcn
!$omp end critical
      enddo

      if (allocated(lsf)) deallocate(lsf)
      if (allocated(lcn)) deallocate(lcn)
!$omp end parallel

    endif ! end VECTOR

  endif ! GLOBAL

! average
!
  where(cn .ge. 1)
    sf = sf / cn
  endwhere

! generate separation length
!
  do i = 1, length
    ls(i) = 1.*(i-1)
  enddo

!
!
  write(*,"(a)") ''

! write data
!
  write( *, "('INFO    : ',a)" ) 'writing correlations to file'
  filename = 'corr.fits.gz'
  write(datafile, '("!",a)') trim(out_dir) // frame // '_' // trim(field) // '_' // trim(filename)
  call fits_put_data_2d(datafile, sf, status)

  write( *, "('INFO    : ',a)" ) 'writing countings to file'
  filename = 'coun.fits.gz'
  write(datafile, '("!",a)') trim(out_dir) // frame // '_' // trim(field) // '_' // trim(filename)
  call fits_put_data_2d(datafile, cn, status)

  write( *, "('INFO    : ',a)" ) 'writing separation length to file'
  filename = 'lsep.fits.gz'
  write(datafile, '("!",a)') trim(out_dir) // frame // '_' // trim(field) // '_' // trim(filename)
  call fits_put_data_1d(datafile, ls, status)

! deallocate local variables
!
  if (allocated(dn)) deallocate(dn)
  if (allocated(bx)) deallocate(bx)
  if (allocated(by)) deallocate(by)
  if (allocated(bz)) deallocate(bz)
  if (allocated(sf)) deallocate(sf)
  if (allocated(cn)) deallocate(cn)
  if (allocated(rb)) deallocate(rb)
  if (allocated(rd)) deallocate(rd)
  if (allocated(ls)) deallocate(ls)
  if (allocated(seed)) deallocate(seed)

! calculate time statistics
!
  call system_clock(le)
  qq = 0.000001*(le-lb)
  write( *, "('INFO    : calculated in ',1f12.3,' sec',a)" ) qq
  write( *, "('INFO    : ',1pe9.3,' points per second')" ) 1.*nshots*nvecs/qq
  write( *, "('INFO    : END')" )

end program
