from __future__ import absolute_import, division, print_function

import os
from subprocess import call

import yaml
import importlib
from collections import OrderedDict

import numpy as np
import xarray as xr
import pandas as pd
import cftime

import calc

grid_file = '/glade/work/mclong/grids/pop-grid-g16.nc'
year_range_clim = slice(1964,2014)

dirf = './fig'
if not os.path.exists(dirf):
    call(['mkdir','-p',dirf])

dirt = '/glade/scratch/mclong/calcs/o2-prediction'
if not os.path.exists(dirt):
    call(['mkdir','-p',dirt])


xr_open_ds = {'chunks' : {'time':1},
              'decode_coords' : False,
              'decode_times' : False}
xr.set_options(enable_cftimeindex=True)

ypm = np.array([31,28,31,30,31,30,31,31,30,31,30,31])/365

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def make_time(year_range):
    from itertools import product
    return [cftime.DatetimeNoLeap(year, month, 1) for year, month in
            product(range(year_range[0], year_range[1]+1), range(1, 13))]


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def open_collection(base_dataset,
                    variables,
                    op,
                    isel_name='',
                    isel={},
                    clobber=False):


    if isel and not isel_name:
        raise ValueError('need isel_name with isel')

    operators = {'ann': calc.compute_ann_mean,
                 'monclim': calc.compute_mon_climatology,
                 'monanom': calc.compute_mon_anomaly}

    if isinstance(op,str) and op in operators:
        operator = operators[op]
    else:
        raise ValueError(f'{op} unknown')


    with open('collections.yml') as f:
        spec = yaml.load(f)

    if base_dataset not in spec:
        raise ValueError(f'Unknown dataset: {base_dataset}')

    spec = spec[base_dataset]
    data_mod = importlib.import_module(spec['source'])

    if operator:
        collection_file_base = f'{dirt}/{base_dataset}.{op}'
    else:
        collection_file_base = f'{dirt}/{base_dataset}'

    if isel:
        collection_file_base = f'{collection_file_base}.{isel_name}'

    ds = xr.Dataset()
    for v in variables:

        collection_file = f'{collection_file_base}.{v}.zarr'

        if clobber:
            call(['rm','-frv',collection_file])

        if os.path.exists(collection_file):
            print(f'reading {collection_file}')
            dsi = xr.open_zarr(collection_file,decode_times=False,decode_coords=False)

        else:
            dsm = data_mod.open_dataset(variable_list=v,**spec['open_dataset'])

            if isel:
                dsm = dsm.isel(**isel)

            dsi = operator(dsm)

            print(f'writing {collection_file}')
            dsi.to_zarr(collection_file)

        ds = xr.merge((ds,dsi))

    return ds


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def annmean_collection(base_dataset,
                       variables,
                       isel={},
                       isel_name='',
                       clobber=False):


    if isel and not isel_name:
        raise ValueError('need isel_name with isel')

    with open('collections.yml') as f:
        spec = yaml.load(f)

    if base_dataset not in spec:
        raise ValueError(f'Unknown dataset: {base_dataset}')

    spec = spec[base_dataset]
    data_mod = importlib.import_module(spec['source'])

    ds = xr.Dataset()
    for v in variables:

        if isel:
            collection_file = f'{dirt}/{base_dataset}.ann.{isel_name}.{v}.zarr'
        else:
            collection_file = f'{dirt}/{base_dataset}.ann.{v}.zarr'

        if clobber:
            call(['rm','-frv',collection_file])

        if os.path.exists(collection_file):
            print(f'reading {collection_file}')
            dsi = xr.open_zarr(collection_file,decode_times=False,decode_coords=False)

        else:
            dsm = data_mod.open_dataset(variable_list=v,**spec['open_dataset'])

            if isel:
                dsm = dsm.isel(**isel)

            dsi = calc.compute_ann_mean(dsm)

            print(f'writing {collection_file}')
            dsi.to_zarr(collection_file)

        ds = xr.merge((ds,dsi))
    return ds


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------
def compute_ann_mean_old(dsm,wgt):

    grid_vars = [v for v in dsm.variables if 'time' not in dsm[v].dims]
    variables = [v for v in dsm.variables if 'time' in dsm[v].dims and v not in ['time','time_bound']]

    # save attrs
    attrs = {v:dsm[v].attrs for v in dsm.variables}
    encoding = {v:dsm[v].encoding for v in dsm.variables}

    # groupby.sum() does not seem to handle missing values correctly: yields 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values for each variable
    valid = {v : dsm[v].groupby('time.year').mean(dim='time').notnull().rename({'year':'time'}) for v in variables}
    ones = dsm.drop(grid_vars).where(dsm.isnull()).fillna(1.).where(dsm.notnull()).fillna(0.)

    # compute the annual means
    ds = (dsm.drop(grid_vars) * wgt).groupby('time.year').sum('time').rename({'year':'time'},inplace=True)
    ones_out = (ones * wgt).groupby('time.year').sum('time').rename({'year':'time'},inplace=True)
    ones_out = ones_out.where(ones_out>0.)

    # renormalize to appropriately account for missing values
    ds = ds / ones_out

    # put the grid variables back
    ds = xr.merge((ds,dsm.drop([v for v in dsm.variables if v not in grid_vars])))

    # apply the valid-values mask
    for v in variables:
        ds[v] = ds[v].where(valid[v])

    # put the attributes back
    for v in ds.variables:
        ds[v].attrs = attrs[v]

    # put the encoding back
    for v in ds.variables:
        ds[v].encoding = encoding[v]

    return ds

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def region_box(ds=None):
    m = region_mask(ds,masked_area=False)
    if len(m.region) != 1:
        raise ValueError('Region > 1 not yet implemented')

    lat = np.concatenate((np.array([(m.where(m>0) * m.TLAT).min().values]),
                          np.array([(m.where(m>0) * m.TLAT).max().values])))
    lon = np.concatenate((np.array([(m.where(m>0) * m.TLONG).min().values]),
                          np.array([(m.where(m>0) * m.TLONG).max().values])))

    y = [lat[0], lat[0], lat[1], lat[1], lat[0]]
    x = [lon[0], lon[1], lon[1], lon[0], lon[0]]
    return x,y


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def region_mask(ds=None,masked_area=True):
    if ds is None:
        ds = xr.open_dataset(grid_file,decode_coords=False)
    TLAT = ds.TLAT
    TLONG = ds.TLONG
    KMT = ds.KMT
    TAREA = ds.TAREA

    nj,ni = KMT.shape

    #-- define the mask logic
    M = xr.DataArray(np.ones(KMT.shape),dims=('nlat','nlon'))
    region_defs = OrderedDict([
        ( 'CalCOFI', M.where((25 <= TLAT) & (TLAT <= 38) &
                             (360-126<=TLONG) & (TLONG <= 360-115)) )
        ])

    #-- do things different if z_t is present
    if 'z_t' not in ds.variables:
        mask3d = xr.DataArray(np.ones(((len(region_defs),)+KMT.shape)),
                    dims=('region','nlat','nlon'),
                    coords={'region':list(region_defs.keys()),
                            'TLAT':TLAT,
                            'TLONG':TLONG})
        for i,mask_logic in enumerate(region_defs.values()):
            mask3d.values[i,:,:] = mask_logic.fillna(0.)
        mask3d = mask3d.where(KMT>0)

    else:
        z_t = ds.z_t
        nk = len(z_t)
        ONES = xr.DataArray(np.ones((nk,nj,ni)),dims=('z_t','nlat','nlon'),coords={'z_t':z_t})
        K = xr.DataArray(np.arange(0,len(z_t)),dims=('z_t'))
        MASK = K * ONES
        MASK = MASK.where(MASK <= KMT-1)
        MASK.values = np.where(MASK.notnull(),1.,0.)

        mask3d = xr.DataArray(np.ones(((len(region_defs),)+z_t.shape+KMT.shape)),
                            dims=('region','z_t','nlat','nlon'),
                            coords={'region':list(region_defs.keys()),
                                    'TLAT':TLAT,
                                    'TLONG':TLONG})

        for i,mask_logic in enumerate(region_defs.values()):
            mask3d.values[i,:,:,:] = ONES * mask_logic.fillna(0.)
        mask3d = mask3d.where(MASK==1.)

    if masked_area:
        area_total = (mask3d * TAREA).sum(['nlat','nlon'])
        mask3d = (mask3d * TAREA) / area_total.where(area_total > 0)
        for i in range(len(region_defs)):
            valid = mask3d.isel(region=i).sum(['nlat','nlon'])
            valid = valid.where(valid>0)
            #np.testing.assert_allclose(valid[~np.isnan(valid)],np.ones(len(z_t))[~np.isnan(valid)])

    return mask3d

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def regional_mean(ds,masked_weights=None,mask_z_level=0.):
    if masked_weights is None:
        masked_weights = region_mask(ds,masked_area=True)

    save_attrs = {v:ds[v].attrs for v in ds.variables}

    dsr = xr.Dataset()


    valid = masked_weights.sum(['nlat','nlon'])
    if 'z_t' in ds.variables:
        validk = valid.sel(z_t=mask_z_level,method='nearest')

    for v in ds.variables:
        if ds[v].dims[-2:] == ('nlat','nlon'):
            if 'z_t' in ds[v].dims or 'z_t' not in ds.variables:
                dsr[v] = (ds[v] * masked_weights).sum(['nlat','nlon']).where(valid>0)
            else:
                dsr[v] = (ds[v] * masked_weights.sel(z_t=mask_z_level,method='nearest')).sum(['nlat','nlon']).where(validk>0)
            dsr[v].attrs = save_attrs[v]
        else:
            dsr[v] = ds[v]

    return dsr

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def xcorr(x,y,dim=None):
    valid = (x.notnull() & y.notnull())
    N = valid.sum(dim=dim)

    x = x.where(valid)
    y = y.where(valid)
    x_dev = x - x.mean(dim=dim)
    y_dev = y - y.mean(dim=dim)

    cov = (x_dev * y_dev).sum(dim=dim) / N
    covx = (x_dev ** 2).sum(dim=dim) / N
    covy = (y_dev ** 2).sum(dim=dim) / N
    return ( cov / np.sqrt(covx * covy) )


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def rmsd(x,y,dim=None):
    valid = (x.notnull() & y.notnull())
    N = valid.sum(dim=dim)
    return np.sqrt(((x-y)**2).sum(dim=dim) / N )

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def open_ann_fosi(anomaly=True):
    #-- open the dataset
    xr_open_ds = { #'chunks' : {'time':1},  # chunking breaks "rolling" method
                  'decode_coords' : False,
                  'decode_times' : False}

    case = 'g.e11_LENS.GECOIAF.T62_g16.009'
    file_in = f'/glade/work/yeager/{case}/budget_O2_npac_{case}.0249-0316.nc'
    ds = xr.open_dataset(file_in,**xr_open_ds)

    #-- convert units
    ds = conform_budget_dataset(ds)

    grid = ds.drop([v for v in ds.variables if 'time' in ds[v].dims])

    #-- interpret time: make time into "year"
    offset = cftime.date2num(cftime.DatetimeGregorian(1699,1,1),
                             ds.time.attrs['units'],
                             ds.time.attrs['calendar'])
    ds['date'] = cftime.num2date(ds.time+offset,
                                 ds.time.attrs['units'],
                                 ds.time.attrs['calendar'])
    ds.time.values = [d.year*1. for d in ds.date.values]

    #-- make into an anomaly
    if anomaly:
        for v in ds.variables:
            if 'time' in ds[v].dims and v != 'time':
                attrs = ds[v].attrs
                ds[v] = ds[v] - ds[v].sel(time=year_range_clim).mean('time')
                ds[v].attrs = attrs
    return ds

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def open_fosi_grid():
    #-- open the dataset
    xr_open_ds = { #'chunks' : {'time':1},  # chunking breaks "rolling" method
                  'decode_coords' : False,
                  'decode_times' : False}

    case = 'g.e11_LENS.GECOIAF.T62_g16.009'
    file_in = f'/glade/work/yeager/{case}/budget_O2_npac_{case}.0249-0316.nc'
    ds = xr.open_dataset(file_in,**xr_open_ds)
    return ds.drop([v for v in ds.variables if 'time' in ds[v].dims])


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def open_ann_dple():
    from glob import glob
    xr_open_ds = {'chunks' : {'S':1},
              'decode_coords' : False,
              'decode_times' : False}

    files = glob('/glade/p_old/decpred/CESM-DPLE/postproc/O2_budget_npac/CESM-DPLE.O2_*.annmean.anom.nc')
    varnames = [f[f.find('.O2_')+1:f.find('.annmean')] for f in files]

    dp = xr.Dataset()
    for v,f in zip(varnames,files):
        dsi = xr.open_dataset(f,**xr_open_ds)
        dsi.rename({'anom':v,'S':'time'},inplace=True)
        dp = xr.merge((dp,dsi))

    dp = xr.merge((dp,open_fosi_grid()))
    return conform_budget_dataset(dp)

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def conform_budget_dataset(ds):
        nmols_to_molm2yr = 1e-9 * 365. * 86400. / ds.TAREA * 1e4
        mol_to_molm2 = 1 / ds.TAREA * 1e4
        long_name = {'O2_lat_adv_res' : 'Lateral advection',
                     'O2_vert_adv_res' : 'Vertical advection',
                     'O2_dia_vmix' : 'Vertical mixing (diabatic)',
                     'O2_adi_vmix' : 'Vertical mixing (adiabatic)',
                     'O2_lat_mix' : 'Lateral mixing',
                     'O2_rhs_tend' : 'Total tendency',
                     'O2_sms' : 'Source/sink',
                     'O2_adv' : 'Total advection',
                     'O2_zint' : 'O$_2$ inventory'}
        for v in ds.variables:
            if 'O2_' in v:
                attrs = ds[v].attrs
                if v == 'O2_zint':
                    ds[v] = (ds[v] * mol_to_molm2).where(ds.KMT > 0)
                    new_units = 'mol m$^{-2}$'
                else:
                    ds[v] = (ds[v] * nmols_to_molm2yr).where(ds.KMT > 0)
                    new_units = 'mol m$^{-2}$ yr$^{-1}$'
                ds[v].attrs = attrs
                ds[v].attrs['units'] = new_units

        #-- add some new fields
        ds['O2_sms'] = ds.O2_prod - ds.O2_cons
        ds['O2_sms'].attrs = ds.O2_cons.attrs

        ds['O2_adv'] = ds.O2_lat_adv_res + ds.O2_vert_adv_res
        ds['O2_adv'].attrs = ds.O2_lat_adv_res.attrs

        for v,l in long_name.items():
            ds[v].attrs['long_name'] = l

        return ds


#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def dataview(forecast_lead,apply_region_mask=False):
    ds = open_ann_fosi(anomaly=True)
    dp = open_ann_dple()

    if hasattr(forecast_lead, '__iter__'):
        dpi = dp.sel(L=slice(forecast_lead[0],forecast_lead[1])).mean(dim='L')
    else:
        dpi = dp.sel(L=forecast_lead)
    dpi.time.values = dpi.time.values + np.mean(forecast_lead)

    time_slice = slice(np.max((ds.time[0],dpi.time[0])),
                       np.min((ds.time[-1],dpi.time[-1])))

    dsi = ds.sel(time=time_slice)
    dpi = dpi.sel(time=time_slice)


    #-- if this is a forecast window, apply running mean
    if hasattr(forecast_lead, '__iter__'):
        save_attrs = {v:dsi[v].attrs for v in dsi.variables}
        N = np.diff(forecast_lead)[0] + 1
        dsi = dsi.rolling(time=N,center=True).mean()
        for v in dsi.variables:
            dsi[v].attrs = save_attrs[v]
        # chunk it
        dsi = dsi.chunk({'time':1})

    if apply_region_mask:
        masked_weights = region_mask(dsi,masked_area=True)
        dsi = regional_mean(dsi,masked_weights=masked_weights).compute()
        dpi = regional_mean(dpi,masked_weights=masked_weights).compute()

    if not np.array_equal(dsi.time, dpi.time):
        raise ValueError('Time coords do not match.')

    return {'fosi':dsi,'dp':dpi}

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def load_pdo(year_range=None,apply_ann_filter=False):
    '''read pdo from JSON file:
        https://www.ncdc.noaa.gov/teleconnections/pdo/data.json
    '''
    import json
    with open('data/pdo-data.json','r') as f:
        pdo_data = json.load(f)
    year = xr.DataArray([float(d[0:4]) for d in pdo_data['data'].keys()],dims='time')
    mon = xr.DataArray([float(d[4:6]) for d in pdo_data['data'].keys()],dims='time')
    time = xr.DataArray([cftime.DatetimeNoLeap(y, m, 1) for y, m in zip(year.values,mon.values)],dims='time')
    data = xr.DataArray([float(d) for d in pdo_data['data'].values()],dims='time',coords={'time':time})

    ds = xr.Dataset({'PDO':data,'year':year,'mon':mon})
    if year_range is not None:
        nx = np.where((year_range[0]<=year) & (year <= year_range[1]))[0]
        ds = ds.isel(time=nx)

    if apply_ann_filter:
        save_attrs = {v:ds[v].attrs for v in ds.variables}
        N = 12
        ds = ds.rolling(time=N,center=True).mean()

    return ds

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def load_npgo(year_range=None,apply_ann_filter=False):
    df = pd.read_table('data/npgo.txt',names=['year','mon','NPGO'],comment='#',delimiter='\s+')


    year = xr.DataArray(df.year.values,dims='time')
    mon = xr.DataArray(df.mon.values,dims='time')
    time = xr.DataArray([cftime.DatetimeNoLeap(y, m, 1) for y, m in zip(year.values,mon.values)],dims='time')
    data = xr.DataArray(df.NPGO.values,dims='time',coords={'time':time})

    ds = xr.Dataset({'NPGO':data,'year':year,'mon':mon})
    if year_range is not None:
        nx = np.where((year_range[0]<=year) & (year <= year_range[1]))[0]
        ds = ds.isel(time=nx)

    if apply_ann_filter:
        save_attrs = {v:ds[v].attrs for v in ds.variables}
        N = 12
        ds = ds.rolling(time=N,center=True).mean()
    return ds

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def interp3d(coord_field,ds,new_levels,dim,**kwargs):
    '''kludged function for interpolation
    '''

    method = kwargs.pop('method','linear')
    if method == 'linear':
        from metpy.interpolate import interpolate_1d
        interp_func = interpolate_1d
    elif method == 'log':
        from metpy.interpolate import log_interpolate_1d
        interp_func = log_interpolate_1d

    newdim = new_levels.dims[0]

    dso = xr.Dataset()
    for v in ds.variables:

        if dim not in ds[v].dims:
            dso[v] = ds[v]
        else:

            dims_in = ds[v].dims
            if len(dims_in) == 1: continue

            interp_axis = dims_in.index(dim)
            dims_out = list(dims_in)
            dims_out[interp_axis] = newdim

            dso[v] = xr.DataArray(
                interp_func(new_levels.values,
                            coord_field.values,ds[v].values,axis=interp_axis),
                dims=dims_out,attrs=ds[v].attrs)

    return dso

#------------------------------------------------------------------------------------
#-- function
#------------------------------------------------------------------------------------

def interp_to_pd(ds):
    '''interpolate onto sigma coordinates'''

    sigma = xr.DataArray(np.array([1.026]),dims='sigma')

    grid_vars = [v for v in ds.variables if 'time' not in ds[v].dims]+['time_bound']

    dso = xr.Dataset()
    for i in range(len(ds.time)):
        print(f'interpolating time level {i+1}')
        dsi = ds.isel(time=i).drop(grid_vars).expand_dims('time')
        dsoi = interp3d(dsi.PD,dsi,sigma,dim='z_t')
        if i > 0:
            dso = xr.concat((dso,dsoi),dim='time')
        else:
            dso = dsoi
        dso = dso.chunk({'time':1})
    #-- put grid variables back
    dso = xr.merge((dso,ds.drop([v for v in ds.variables if v not in grid_vars])))

    return dso
