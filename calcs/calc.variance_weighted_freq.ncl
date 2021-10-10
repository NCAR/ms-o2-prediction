load "~mclong/p/util/ncl2/analysis.interface.ncl"

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin
  eopt = True
  eopt@req = (/"VARNAME","FILE_IN","FILE_OUT"/)
  eopt@opt = (/"FLAGS","GRID_FILE"/)

  env = shell_inputs(eopt)
  print(env)

;------------------------------------------------------------
;---- read variable
;------------------------------------------------------------
  OPT = True
  OPT@add_coords = True
  x  = read_var(env@FILE_IN,env@VARNAME,env@DIMSUB,env@DIMCOORD,OPT)

  dimsize = dimsizes(x)
  nd = dimsizes(dimsize)

;------------------------------------------------------------
;---- specify calculation options
;------------------------------------------------------------
  ;; http://www.ncl.ucar.edu/Document/Functions/Built-in/specx_anal.shtml
  ;; (iopt) detrending: 0 = remove mean, 1 = remove linear trend
  iopt = 1
  ;; (jave) smoothing: odd number (>= 3)
  ;;     jave < 3: Do no smoothing. spcx contains raw spectra estimates (periodogram).
  ;;     jave >= 3: Average jave periodogram estimates together utilizing modified
  ;;                daniell smoothing  (good stability but may lead to large bias).
  ;;                All weights are 1/jave except weight(1) and weight(jave) which are 1/(2*jave).   ;;                This is the recommended option.
  ;;                It is this number which has the most impact on the degrees of freedom.
  jave = 5
  ;; (pct) tapering: (0.0 <= pct <= 1.0);
  ;;                pct = 0.0, no tapering will be done.
  ;;                If pct = 1.0, the whole series is effected.
  ;;                A value of 0.10 is common (tapering should always be done).
  pct = 0.1

;------------------------------------------------------------
;---- define output vars
;------------------------------------------------------------
  N = dimsize(0)
  print("N = "+N)
  df = 1.d0/N
  print("df = "+df)

  dimsize(0) = N/2

  ;;--- freq
  ;; frq:    A one-dimensional array of length N/2 representing frequency (cycles/time).
  freq = new(dimsize(0),double)
  freq!0 = "freq"

  ;; bw:     spectral band width (scalar)
  bw = new(1,double)
  bw!0 = "scalar"
  bw@long_name = "Spectral bandwidth"

  dof = new(1,double)
  dof!0 = "scalar"
  dof@long_name = "degrees of freedom"

  ;;--- spectrum
  ;; spcx:   1D array of length N/2.
  ;;         spcx(0) - spectral estimate at frequency = (1/N)  [N=dimsizes(x)]
  ;;         spcx(N/2-1)- spectral estimate at frequency = 0.5
  ;;         These spectra have been normalized so that the area under the curve:
  ;;         (spcx(0)+spcx(N/2-1))*(df/2) + SUM{spcx(1:N/2-2)*df}
  ;;         equals the variance of the detrended series, where df=(1/N)=frequency spacing.
  ;;         The units are variance/(unit frequency interval).
  spcx = new(dimsize,double)
  spcx!0 = "freq"
  spcx!1 = x!1
  spcx!2 = x!2

  spcx@long_name = "Spectrum"
  if isatt(x,"long_name")
      spcx@long_name = spcx@long_name+" "+x@long_name
  end if
  if isatt(x,"units")
      spcx@units = "("+x@units+")^2"
  end if
  spcx@note = "spectra have been normalized so that the area under the curve: "+ \
              "(spcx(0)+spcx(N/2-1))*(df/2) + SUM{spcx(1:N/2-2)*df} = variance of detrended series"

  ;;--- mean
  ;; xavei:  average of the x series on input (scalar)
  xavg = new(dimsize(1:nd-1),double)
  xavg!0 = x!1
  xavg!1 = x!2
  xavg@long_name = "Mean"
  if isatt(x,"long_name")
      xavg@long_name = xavg@long_name+" "+x@long_name
  end if
  if isatt(x,"units")
      xavg@units = x@units
  end if

  ;;--- variance
  ;; xvari:  variance of the x series on input (scalar)
  xvari = new(dimsize(1:nd-1),double)
  xvari!0 = x!1
  xvari!1 = x!2
  xvari@long_name = "Total variance"
  if isatt(x,"long_name")
      xvari@long_name = xvari@long_name+" "+x@long_name
  end if
  if isatt(x,"units")
      xvari@units = "("+x@units+")^2"
  end if

  ;; xvaro:  variance of the x series after detrending (scalar)
  xvaro = new(dimsize(1:nd-1),double)
  xvaro!0 = x!1
  xvaro!1 = x!2
  xvaro@long_name = "Variance"
  if isatt(x,"long_name")
      xvaro@long_name = xvaro@long_name+" "+x@long_name
  end if
  if isatt(x,"units")
      xvaro@units = "("+x@units+")^2"
  end if

  ;;--- lag1 autocorrelation
  ;; xlag1:  lag one auto correlation of the x series after detrending (scalar)
  xcor = new(dimsize(1:nd-1),double)
  xcor!0 = x!1
  xcor!1 = x!2
  xcor@long_name = "Lag 1 autocorrelation"

  ;;--- detrending slope
  ;; xslope: least squares slope of linear trend (if iopt = 1) of the x series (scalar)
  xslp = new(dimsize(1:nd-1),double)
  xslp!0 = x!1
  xslp!1 = x!2
  xslp@long_name = "Linear slope (from detrending)"
  if isatt(x,"long_name")
      xslp@long_name = xslp@long_name+" "+x@long_name
  end if

  ;;--- confidence metrics
  ;; splt = specx_ci(sdof,0.95,0.99)
  ;; splt(0,:) - input spectrum
  ;; splt(1,:) - Markov "Red Noise" spectrum
  ;; splt(2,:) - lower confidence bound for Markov
  ;; splt(3,:) - upper confidence bound for Markov
  rednoise = new(dimsize,double)
  rednoise!0 = "freq"
  rednoise!1 = x!1
  rednoise!2 = x!2
  rednoise@long_name = "Markov 'red noise' spectrum"

  spcx_ci_lo = new(dimsize,double)
  spcx_ci_lo!0 = "freq"
  spcx_ci_lo!1 = x!1
  spcx_ci_lo!2 = x!2
  spcx_ci_lo@long_name = "low confidence bound from Markov"

  spcx_ci_hi = new(dimsize,double)
  spcx_ci_hi!0 = "freq"
  spcx_ci_hi!1 = x!1
  spcx_ci_hi!2 = x!2
  spcx_ci_hi@long_name = "upper confidence bound from Markov"

;------------------------------------------------------------
;---- compute spectra
;------------------------------------------------------------
  if nd .eq. 3
      q = 0
      do j = 0,dimsize(1)-1
          do i = 0,dimsize(2)-1
              if any(ismissing(x(:,j,i)))
                  continue
              end if

              sdof := specx_anal(x(:,j,i),iopt,jave,pct)

              spcx(:,j,i) = (/sdof@spcx/)
              if q .eq. 0
                  freq = (/sdof@frq/)
                  bw   = (/sdof@bw/)
                  dof  = (/sdof/)
              else if q .gt. 0
                  if .not. all(freq.eq.sdof@frq)
                      print("freak freq")
                      status_exit(1)
                  end if
              end if
              end if
              xavg(j,i)   = (/sdof@xavei/)
              xvari(j,i)  = (/sdof@xvari/)
              xvaro(j,i)  = (/sdof@xvaro/)
              xcor(j,i)   = (/sdof@xlag1/)
              xslp(j,i)   = (/sdof@xslope/)

              ;splt = specx_ci(sdof,0.95,0.99)
              ;rednoise(:,j,i)   = (/splt(1,:)/)
              ;spcx_ci_lo(:,j,i) = (/splt(2,:)/)
              ;spcx_ci_hi(:,j,i) = (/splt(3,:)/)

              q = q + 1
          end do
      end do
  end if

  ;; write the frequency coordinate
  freq&freq = freq
  spcx&freq = freq
  rednoise&freq = freq
  spcx_ci_lo&freq = freq
  spcx_ci_hi&freq = freq

  ;; add a "period" field
  period = 1./freq
  period!0 = "freq"
  period&freq = freq

  ;; compute variance check
  varspc_int = new(dimsize(1:nd-1),double)
  varspc_int!0 = x!1
  varspc_int!1 = x!2
  varspc_int@long_name = "Variance reconstructed from spectra"
  varspc_int = (/(spcx(0,:,:)+spcx(N/2-1,:,:))*(df/2.) + dim_sum_n(spcx(1:N/2-2,:,:)*df,0)/)

  ;; compute cumsum variance
  varspc_cumint = new(dimsize,double)
  varspc_cumint!0 = "freq"
  varspc_cumint!1 = x!1
  varspc_cumint!2 = x!2
  varspc_cumint@long_name = "Cumulative variance"

  varspc_cumint(0,:,:) = (/spcx(0,:,:)*(df/2.)/)
  do i = 1,N/2-2
      varspc_cumint(i,:,:) = (/varspc_cumint(i-1,:,:) + spcx(i,:,:)*df/)
  end do
  varspc_cumint(N/2-1,:,:) = (/varspc_cumint(N/2-2,:,:) + spcx(N/2-1,:,:)*(df/2.)/)

  ;; compute variance-weighted average
  varw_avg_p = new(dimsize(1:nd-1),double)
  varw_avg_p!0 = x!1
  varw_avg_p!1 = x!2
  varw_avg_p@long_name = "Variance weighted period"

  denom = ( (freq(0)*spcx(0,:,:) + freq(N/2-1)*spcx(N/2-1,:,:))*(df/2.) \
          + dim_sum_n( \
          conform(spcx(1:N/2-2,:,:),freq(1:N/2-2),0) * spcx(1:N/2-2,:,:) * df, 0) )
  denom = where(denom .eq. 0.,denom@_FillValue,denom)
  varw_avg_p = (/varspc_int/denom/)

;------------------------------------------------------------
;---- write output file
;------------------------------------------------------------
  system("rm -f " + env@FILE_OUT)
  fo = addfile(env@FILE_OUT,"c")

  fo->spcx = spcx
  fo->freq = freq
  fo->bw   = bw
  fo->dof  = dof
  fo->xavg = xavg
  fo->xvari = xvari
  fo->xvaro = xvaro
  fo->xcor = xcor
  fo->xslp = xslp
  fo->period = period

  fo->rednoise   = rednoise
  fo->spcx_ci_lo = spcx_ci_lo
  fo->spcx_ci_hi = spcx_ci_hi

  fo->varspc_int = varspc_int
  fo->varw_avg_p = varw_avg_p
  fo->varspc_cumint = varspc_cumint

  document_calc(fo,env)

  if .not.ismissing(env@ABNORMAL_EXIT)
      system("rm -f " + env@ABNORMAL_EXIT)
  end if
  exit
end

status_exit(1)
