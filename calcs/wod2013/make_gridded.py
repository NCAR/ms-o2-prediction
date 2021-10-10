#! /usr/bin/env python
import os
import sys
from subprocess import call
from datetime import datetime
from netcdftime import utime
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

import grid_tools
from wod_info import wodvars_attrs,standard_z,standard_z_depth_bounds
from wod_info import woa_shortnames,wodvars_convert_units
from calendar import monthrange

from remap_z import calc_z_to_sigma
calc_z_to_sigma.verbose = False

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def compute_anom_wrt_woa(variables,file_in,file_clm,file_out,max_depth,
                         output_on_sigma=False):
    #-- read dataset
    print('-'*80)
    print('gridded dataset')
    ds = xr.open_dataset(file_in,decode_times=False)
    print(ds)
    print

    #-- read climatology
    woa = {}
    for v in variables:
        #-- load the data
        dsi = xr.open_dataset(file_clm[v],decode_times=False)

        #-- change names
        dsi = dsi.rename({woa_shortnames[v]+'_an':v})

        #-- convert units
        if v in wodvars_convert_units:
            dsi[v] = dsi[v] * wodvars_convert_units[v]['factor']
            dsi[v].attrs['units'] = wodvars_convert_units[v]['new_units']

        #-- merge
        woa = xr.merge((woa,dsi))

    if output_on_sigma:
        xr.testing.assert_equal(woa.sigma,ds.sigma)
    else:
        if max_depth < 6000.:
            woa = woa.sel(depth=slice(0,max_depth))
        xr.testing.assert_equal(woa.depth,ds.depth)

    print('-'*80)
    print('WOA climatology')
    print(woa)
    print

    #-- unique years
    year = np.unique(ds.year)
    print('-'*80)
    print('unique years')
    print(year)
    print

    #-- compute anomalies
    file_cat = []
    for y in year:
        #-- index the right year
        nx = np.where(ds.year==y)[0]
        dsy = ds.isel(time=slice(nx[0],nx[-1]+1))

        #-- compute anomaly
        dsa = (dsy - woa).mean(dim='time')
        dsa['year'] = dsy.year.isel(time=-1,drop=False).expand_dims('time').astype(np.int32)
        dsa.time.values[0] = dsy.time.isel(time=-1).values
        dsa['area'] = ds.area
        dsa['dz'] = ds.dz

        #-- change type and include total count
        for v in variables:
            dsa[v] = dsa[v].expand_dims('time').astype(np.float32)
            dsa[v+'_cnt'] = dsy[v+'_cnt'].sum(dim='time').expand_dims('time').astype(np.int32)

        #-- copy attributes
        for v in dsa.variables:
            dsa[v].attrs = dsy[v].attrs

        #-- write file
        file_tmp = file_out+'.%04d.tmp'%y
        print('writing file: %s'%file_tmp)
        dsa.to_netcdf(file_tmp,unlimited_dims='time')
        file_cat.append(file_tmp)

    ds = xr.open_mfdataset(file_cat,concat_dim='time',data_vars='minimal')
    ds.to_netcdf(file_out,unlimited_dims='time',engine='h5netcdf')
    call('rm -f '+' '.join(file_cat),shell=True)

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def grid_stations(variables,files_in,file_out,max_depth,yr0,yr1,
                  output_on_sigma=False,
                  sigma_interp_method='remap_z'):

    na = np.newaxis
    if not any(sigma_interp_method == s for s in ['linear','remap_z']):
        print('sigma_interp_method "%s" not recognized'%sigma_interp_method)
        exit(1)

    #---------------------------------------------------------------------------
    #-- generate time axis
    #---------------------------------------------------------------------------

    cdftime = utime(wodvars_attrs['time']['units'],
                    calendar=wodvars_attrs['time']['calendar'])
    date = [datetime(y,m,monthrange(y,m)[1])
            for y in range(yr0,yr1+1)
            for m in range(1,13)]
    year = np.array([d.year for d in date])
    month = np.array([d.month for d in date])
    time = cdftime.date2num(date)

    nt = len(time)

    #---------------------------------------------------------------------------
    #-- generate grid
    #---------------------------------------------------------------------------

    islatlon = True
    nx = 360
    ny = 180
    grid = grid_tools.generate_latlon_grid(nx=nx,ny=ny,lon0=-180.)
    grid = grid.rename({'latitude':'lat','longitude':'lon'})

    corner_w_lon = grid.xv.values[:,:,0]
    corner_e_lon = grid.xv.values[:,:,1]
    corner_s_lat = grid.yv.values[:,:,0]
    corner_n_lat = grid.yv.values[:,:,3]

    #---------------------------------------------------------------------------
    #-- make output dataset
    #---------------------------------------------------------------------------

    print('constructing output dataset')

    standard_z_selection = (standard_z<=max_depth)

    #-- depth levels
    if output_on_sigma:
        sigma_start = 24.475
        sigma_stop = 26.975
        dsigma = 0.05

        depth_coord_name = 'sigma'
        dz_name = ''
        sigma_edges = np.arange(sigma_start,sigma_stop+dsigma,dsigma)
        sigma = np.round(100.*np.average(np.vstack((sigma_edges[0:-1],sigma_edges[1:])),axis=0))/100.
        depth_coord = sigma

        klev_out = len(sigma_edges)-1
        dz = np.diff(standard_z_depth_bounds[standard_z_selection,:],axis=1)[:,0]
        z_edges = np.concatenate(([0.],np.cumsum(dz)))

    else:
        depth_coord_name = 'depth'
        dz_name = 'dz'
        depth_coord = standard_z[standard_z_selection]
        dz = np.diff(standard_z_depth_bounds[standard_z_selection,:],axis=1)[:,0]

    nz = len(depth_coord)

    #-- add coordinates
    coords = {'time':time,'lat':grid.lat,'lon':grid.lon,
              depth_coord_name:depth_coord}
    dso = xr.Dataset(coords=coords)

    if dz_name:
        dso[dz_name] = xr.DataArray(dz,dims=(depth_coord_name))
    dso['month'] = xr.DataArray(month,dims=('time')).astype(np.int32)
    dso['year'] = xr.DataArray(year,dims=('time')).astype(np.int32)
    dso['area'] = grid.area

    #-- derived variables?
    variables_derived = []
    if 'temperature' in variables and 'salinity' in variables:
        import seawater as sw
        variables_derived = ['potential_density']

    isderived = [False for v in variables] + [True for v in variables_derived]
    variables = variables + variables_derived

    #-- add variables
    for v in variables:
        attrs = {'long_name':v,'units':''}
        if v in wodvars_attrs:
            attrs = wodvars_attrs[v]

        dso[v] = xr.DataArray(np.zeros((nt,nz,ny,nx)).astype(np.float64),
                              dims=('time',depth_coord_name,'lat','lon'),
                              attrs=attrs)

        attrs['long_name'] = attrs['long_name']+' std. dev.'
        dso[v+'_std'] = xr.DataArray(np.zeros((nt,nz,ny,nx)).astype(np.float64),
                                     dims=('time',depth_coord_name,'lat','lon'),
                                     attrs=attrs)

        attrs['long_name'] = attrs['long_name']+' count'
        attrs['units'] = 'count'
        dso[v+'_cnt'] = xr.DataArray(np.zeros((nt,nz,ny,nx)).astype(np.float64),
                                     dims=('time',depth_coord_name,'lat','lon'),
                                     attrs=attrs)

    #-- add attributes to coordinates
    for v in ['time',depth_coord_name,'lat','lon']:
        if v in wodvars_attrs:
            dso[v].attrs = wodvars_attrs[v]

    print('-'*80)
    print('output dataset')
    print(dso)
    print

    #---------------------------------------------------------------------------
    #-- perform gridding of anomalies
    #---------------------------------------------------------------------------

    #-- loop over files
    for f in files_in:
        if not os.path.exists(f):
            print('missing %s'%f)
            continue

        #-- open dataset
        print('-'*80)
        print('reading %s'%f)
        ds = xr.open_dataset(f,decode_times=False)

        cdftime = utime(ds.time.attrs['units'],
                        calendar=ds.time.attrs['calendar'])

        SDATE = cdftime.num2date(ds.time.values)
        SYEAR = np.array([d.year for d in SDATE])

        if not ((yr0 <=SYEAR) & (SYEAR<=yr1)).any():
            print('year range: %d-%d'%(SYEAR.min(),SYEAR.max()))
            print('no data in range %d-%d'%(yr0,yr1))
            continue
        ncast = len(ds.cast)

        print(ds)

        #-- loop over casts
        for i in range(ncast):
            #-- read coordinates of cast file
            slon = ds.lon.values[i]
            slon = np.where(slon == 180., -180., slon)
            slat = ds.lat.values[i]
            sdate = SDATE[i]

            if not ((yr0 <=sdate.year) & (sdate.year<=yr1)):
                continue

            #-- index into gridded array
            L = np.where((sdate.year == year) & (sdate.month == month))[0]
            if len(L) == 0:
                print('\nERROR: bad time point')
                exit(1)

            if islatlon:
                J,I = np.where((corner_w_lon <= slon) & (slon < corner_e_lon) &
                               (corner_s_lat <= slat) & (slat < corner_n_lat))
            else:
                J,I = grid_tools.index_point_on_grid(slon,slat,grid.xv,grid.yv)

            if J.size == 0 or I.size == 0:
                print('\nWARNING: bad lat/lon point')
                print('index = %d'%i)
                print('cast = %d'%ds.cast.values[i])
                print('lat = %.4f'%slat)
                print('lon = %.4f'%slon)
                continue

            #-- compute average
            if output_on_sigma:
                depth = ds.depth.values
                pressure = sw.eos80.pres(depth=depth,
                                         lat=ds.lat.values[i])
                pd = sw.eos80.pden(s=ds['salinity'].values[i,:],
                                   t=ds['temperature'].values[i,:],
                                   p=pressure,pr=0.)

                if np.isnan(pd).all():
                    continue

                pd = pd - 1000.
                pd_sort_index = np.argsort(pd)


                if sigma_interp_method == 'remap_z':
                    kmt = np.where(~np.isnan(pd))[0][-1]+1
                    thickness,z = calc_z_to_sigma.remap_z_type(klev_out,
                                                               kmt,
                                                               z_edges,
                                                               depth[na,:,na,na],
                                                               pd[na,:,na,na],
                                                               sigma_edges)
                    thickness = calc_z_to_sigma.squeeze_and_nanify(thickness)
                    z = calc_z_to_sigma.squeeze_and_nanify(z)

            for iv,v in enumerate(variables):

                #-- read values
                if isderived[iv]:
                    if v == 'potential_density':
                        pressure = sw.eos80.pres(depth=ds.depth.values,
                                                 lat=ds.lat.values[i])
                        x = sw.eos80.pden(s=ds['salinity'].values[i,:],
                                          t=ds['temperature'].values[i,:],
                                          p=pressure,pr=0.)
                    elif v == 'THICKNESS':
                        x = thickness

                    elif v == 'Z':
                        x = z
                else:
                    x = ds[v].values[i,:]

                if output_on_sigma and not any(v == s for s in ['THICKNESS','Z']):
                    if sigma_interp_method == 'linear':
                        finterp = interp1d(pd[pd_sort_index],x[pd_sort_index],
                                           assume_sorted=True,
                                           bounds_error=False)
                        x = finterp(sigma)
                    elif sigma_interp_method == 'remap_z':
                        _,x = calc_z_to_sigma.remap_z_type(klev_out,
                                                           kmt,
                                                           z_edges,
                                                           x[na,:,na,na],
                                                           pd[na,:,na,na],
                                                           sigma_edges)
                        x = calc_z_to_sigma.squeeze_and_nanify(x)

                #-- count non-missing
                cnt = ~np.isnan(x)*1

                #-- set temporary array
                xbar = dso[v].values[L,:,J,I]
                n = dso[v+'_cnt'].values[L,:,J,I]
                s2 = dso[v+'_std'].values[L,:,J,I]

                #-- compute mean & variance
                n += cnt
                dev = np.where(np.isnan(x),0.,x - xbar)
                xbar += np.divide(dev, n, where=n>0)
                dev2 = np.where(np.isnan(x),0.,x - xbar)
                s2 += dev*dev2

                #-- write back to output dataset
                dso[v].values[L,:,J,I] = xbar
                dso[v+'_cnt'].values[L,:,J,I] = n
                dso[v+'_std'].values[L,:,J,I] = s2

                del xbar
                del n
                del s2

            percent_complete = i*100./ncast
            if sys.stdout.isatty():
                print "\r{:0.2f} %".format(percent_complete),
            else:
                if i%(ncast/200) == 0:
                    print "{}: {:0.2f} %".format(
                        datetime.now().strftime("%Y %m %d %H:%M:%S"),
                        percent_complete)
        print
        ds.close()

    #---------------------------------------------------------------------------
    #-- finalize
    #---------------------------------------------------------------------------

    print('finalizing')

    #-- apply normalizations and fill values
    for v in variables:
        #-- count
        n = dso[v+'_cnt'].values

        #-- normalize variance
        var = dso[v+'_std'].values
        dso[v+'_std'].values = np.where(n > 2, np.divide(var,(n-1),where=n-1>0), np.nan)
        dso[v+'_std'].values = np.sqrt(dso[v+'_std'].values)

        #-- set missing values
        var = dso[v].values
        dso[v].values = np.where(n == 0, np.nan, var)

    #-- output single precisions
    for v in variables:

        attrs = dso[v].attrs
        if v in wodvars_convert_units:
            dso[v] = dso[v] * wodvars_convert_units[v]['factor']
            attrs['units'] = wodvars_convert_units[v]['new_units']
        dso[v] = dso[v].astype(np.float32)
        dso[v].attrs = attrs

        attrs = dso[v+'_cnt'].attrs
        dso[v+'_cnt'] = dso[v+'_cnt'].astype(np.int32)
        dso[v+'_cnt'].attrs = attrs

        attrs = dso[v+'_std'].attrs
        dso[v+'_std'] = dso[v+'_std'].astype(np.float32)
        if v in wodvars_convert_units:
            dso[v+'_std'] = dso[v+'_std'] * wodvars_convert_units[v]['factor']
            attrs['units'] = wodvars_convert_units[v]['new_units']
        dso[v+'_std'].attrs = attrs

    #-- write to file
    dso.to_netcdf(file_out,unlimited_dims='time')

#-------------------------------------------------------------------------------
#-- main
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import json

    #-- set defaults
    control_defaults = {
        'variables': [],
        'files_in': [],
        'file_out': [],
        'yr0': None,
        'yr1': None,
        'max_depth': None,
        'output_on_sigma':False,
        'sigma_interp_method':'remap_z'}

    p = argparse.ArgumentParser(description='compute gridded anomalies')
    p.add_argument('json_control',
                   default=control_defaults)

    p.add_argument('-f',dest='json_as_file',
                   action='store_true',default=False,
                   help='Interpret input as a file name')

    args = p.parse_args()
    if not args.json_as_file:
        control_in = json.loads(args.json_control)
    else:
        with open(args.json_control,'r') as fp:
            control_in = json.load(fp)

    control = control_defaults
    control.update(control_in)

    #-- begin
    grid_stations(variables=control['variables'],
                  files_in=control['files_in'],
                  file_out=control['file_out'],
                  max_depth=control['max_depth'],
                  yr0=control['yr0'],
                  yr1=control['yr1'],
                  output_on_sigma=control['output_on_sigma'],
                  sigma_interp_method=control['sigma_interp_method'])
