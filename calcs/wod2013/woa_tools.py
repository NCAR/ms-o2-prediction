#! /usr/bin/env python
import os
from subprocess import call
import xarray as xr
import numpy as np

woa_pth = '/glade/p/ncgd0033/obs/woa2013v2/1x1d'

#----------------------------------------------------
#--- function
#----------------------------------------------------
def clm_code(mon):
    # 13: jfm, 14: amp, 15: jas, 16: ond
    if 1 <= mon <= 3:
        return 13
    elif 4 <= mon <= 6:
        return 14
    elif 7 <= mon <= 9:
        return 15
    elif 10 <= mon <= 12:
        return 16

#----------------------------------------------------
#--- function
#----------------------------------------------------

def regrid_to_sigma_coords(file_in,file_out,file_sigma,clobber=False):
    '''
    regrid to sigma coords
    '''

    if os.path.exists(file_out) and not clobber:
        return

    from remap_z import calc_z_to_sigma

    sigma_start = 24.475
    sigma_stop = 26.975
    sigma_delta = 0.05

    calc_z_to_sigma.compute(file_in,file_out,
                            file_in_sigma = file_sigma,
                            sigma_varname = 'PD',
                            convert_from_pd = True,
                            sigma_start = sigma_start,
                            sigma_stop = sigma_stop,
                            sigma_delta = sigma_delta,
                            zname = 'depth',
                            dzname = 'dz',
                            kmtname = '')

#----------------------------------------------------
#--- function
#----------------------------------------------------

def add_potential_density(file_in_s,file_in_t,file_out,clobber=False):
    import seawater as sw
    if os.path.exists(file_out) and not clobber:
        return

    dst = xr.open_dataset(file_in_t,decode_times=False,decode_coords=False)
    dss = xr.open_dataset(file_in_s,decode_times=False,decode_coords=False)

    na = np.newaxis
    pressure = sw.eos80.pres(dst.depth.values[na,:,na,na],dst.lat.values[na,na,:,na])
    dsd = dst.rename({'t_an':'PD'})

    dsd.PD.values = sw.eos80.pden(s=dss.s_an.values,
                                  t=dst.t_an.values,
                                  p=pressure,pr=0.)
    dsd.to_netcdf(file_out)

#----------------------------------------------------
#--- function
#----------------------------------------------------

def make_merged_clim(odir,clobber=False):
    '''
    blend monthly and seasonal WOA data
    '''

    files_out = {}
    for v in ['t','s','o','n','p','i','O','A']:

        if v in ['t','s']:
            file_tmplt = woa_pth+'/woa13_decav_%s%02d_01v2.nc'
        elif v in ['o','p','n','i','O','A']:
            file_tmplt = woa_pth+'/woa13_all_%s%02d_01.nc'
        else:
            print('no file template defined')
            exit(1)

        file_out = odir+'/woa13v2_seas_mon_merged_%s_01.nc'%v
        files_out[v] = file_out
        if os.path.exists(file_out) and not clobber:
            continue

        dso_list = []
        file_ann = file_tmplt%(v,0)
        ds_ann = xr.open_dataset(file_ann,decode_times = False,
                                 decode_coords = False)
        for m in range(1,13):
            print('month %d'%m)
            file_mon = file_tmplt%(v,m)
            file_clm = file_tmplt%(v,clm_code(m))

            #-- read mon data
            ds_mon = xr.open_dataset(file_mon,
                                   decode_times = False,
                                   decode_coords = False)
            #-- read climatology
            ds_clm = xr.open_dataset(file_clm,
                                     decode_times = False,
                                     decode_coords = False)

            if len(ds_clm) == len(ds_mon):
                print('using annual means')
                ds_clm = ds_ann.copy()

            #-- overwrite upper ocean with monthly data
            dso = ds_ann.copy()
            dso = dso.drop([vi for vi in ds_clm.variables if vi not in [v+'_an']])

            nk = len(ds_mon.depth)
            for vi in dso.variables:
                if dso[vi].ndim == 4:
                    print('blending %s'%vi)
                    dso[vi].values[:,0:nk,:,:] = ds_mon[vi].values[:]

            dso.load()
            dso_list.append(dso)

        combined = xr.concat(dso_list,'time')
        combined['dz'] = ds_clm.depth_bnds.diff(dim='nbounds')[:,0]

        #-- produce netcdf file
        combined.to_netcdf(file_out)
        print('wrote %s'%file_out)

    return files_out

if __name__ == '__main__':

    odir = os.path.join(os.environ['SCRATCH'],'wod2013','woa2013v2')
    if not os.path.exists(odir): call(['mkdir','-p',odir])

    files_out = make_merged_clim(odir,clobber=False)

    file_sigma = files_out['s'].replace('_s_','_PD_')
    add_potential_density(files_out['s'],files_out['t'],file_sigma,clobber=False)

    for v,file_in in files_out.items():
        file_out = file_in.replace('.nc','.sigma.nc')
        regrid_to_sigma_coords(file_in,file_out,file_sigma,clobber=False)
